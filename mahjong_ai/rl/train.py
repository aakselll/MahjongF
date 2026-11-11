"""Trainer RL reworké : PPO on-policy mono-processus, 3 phases oracle.

Usage typique (1 siège RL vs 3 SL figés, tonpuu) :

    python -m mahjong_ai.rl.train \
        --sl-discard checkpoints/sl_tonpusen_light/discard.pt \
        --sl-frozen   checkpoints/sl_tonpusen_light \
        --grp-checkpoint checkpoints/grp_tonpusen.pt \
        --output checkpoints/rl_v3/rl_discard.pt \
        --total-steps 500000 --n-envs 16 --batch-games 32

Pourquoi ce fichier remplace training/train_rl_suphx.py (cf. ANALYSE_RL.md) :
    P1  buffer replay off-policy -> ratio PPO invalide  : désormais chaque
        batch est collecté puis jeté, ratio = 1 au 1er passage ;
    P2  3 pas de gradient par update -> ~30x plus ici (minibatches) ;
    P3  self-play symétrique par défaut -> désormais 1 siège RL vs 3 SL
        figés, mean_rank < 2.50 = progrès mesurable en continu ;
    P4  ancre KL 0.05 vers le SL -> désactivée par défaut ;
    P5  entropie adaptative qui écrase la policy loss -> off par défaut,
        mode adaptatif recalibré (cible = entropie mesurée au départ) ;
    P6  forwards batch-1 + têtes SL non batchées + DFS look-ahead mort ->
        envs vectorisés, forwards groupés par modèle, look-ahead coupé.

Métriques clés dans les logs :
    mr    mean_rank du siège RL sur la fenêtre (2.50 = niveau SL) ;
    dng%  % de défausses RL qui feraient ron (chute en phase 1 = la policy
          exploite l'oracle) ;
    rdev  |ratio-1| au 1er minibatch (doit être ~0.000) ;
    H     entropie de la policy ; ||W_o|| usage des canaux oracle.
"""
import argparse
import collections
import os
import time

import numpy as np
import torch

from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner
from mahjong_ai.rl.ppo import compute_advantages, ppo_update


# ---------------------------------------------------------------------- #
# Schedule 3 phases (papier §3.3)
# ---------------------------------------------------------------------- #
def get_phase_and_gamma(step, total_steps, oracle_frac=0.30,
                        transition_frac=0.40):
    """Phase 1 : oracle complet (gamma=1). Phase 2 : dropout linéaire
    1 -> 0. Phase 3 : agent normal (gamma=0), LR réduit."""
    p1_end = int(total_steps * oracle_frac)
    p2_end = int(total_steps * (oracle_frac + transition_frac))
    if step < p1_end:
        return 1, 1.0
    if step < p2_end:
        frac = (step - p1_end) / max(p2_end - p1_end, 1)
        return 2, max(0.0, 1.0 - frac)
    return 3, 0.0


PHASE_POLICY_LR = {1: 1.0, 2: 1.0, 3: 0.1}
PHASE_CRITIC_LR = {1: 1.0, 2: 0.5, 3: 0.1}


def phase_lr_mult(phase, no_oracle=False):
    """Multiplicateurs (policy, critic) du LR pour la phase.

    [NO-ORACLE] En regime sans oracle, TOUT l'entrainement se fait en
    conditions de deploiement (gamma=0) : conserver le x0.1 de la
    phase 3 figerait la policy (cf. run rl_lookahead_v1 : pl~0, clip~0,
    kl~0 sur 75k steps). On force donc 1.0 partout.
    """
    if no_oracle:
        return 1.0, 1.0
    return PHASE_POLICY_LR[phase], PHASE_CRITIC_LR[phase]


# ---------------------------------------------------------------------- #
# Fenêtre glissante de stats de jeu
# ---------------------------------------------------------------------- #
class StatsWindow:
    """Agrège les stats des `capacity` dernières parties (métrique de
    progrès : en mode opponent, c'est le niveau du siège RL vs 3 SL)."""

    def __init__(self, capacity=400):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)

    def add(self, stats):
        self.deque.append(stats)

    def reset(self):
        self.deque.clear()

    def format(self):
        n = len(self.deque)
        if n == 0:
            return None
        ranks = [s['rank_of_seat0'] for s in self.deque]
        c = collections.Counter(ranks)
        mean_rank = float(np.mean(ranks))
        # IC95 approx sur le mean_rank (bootstrap normal)
        se = float(np.std(ranks)) / max(np.sqrt(n), 1e-9)
        n_rounds = sum(s['n_rounds'] for s in self.deque)
        n_tsumo = sum(s['n_tsumo'] for s in self.deque)
        n_ron = sum(s['n_ron'] for s in self.deque)
        n_exh = sum(s['n_exhaustive'] for s in self.deque)
        n_ab = sum(s['n_abortive'] for s in self.deque)
        total_ends = max(n_tsumo + n_ron + n_exh + n_ab, 1)
        n_rl = sum(s['n_rl_discards'] for s in self.deque)
        n_dng = sum(s['n_danger_discards'] for s in self.deque)
        n_unh = sum(s['n_unhandled'] for s in self.deque)
        line = (f"n={n} [{c.get(1,0)}/{c.get(2,0)}/{c.get(3,0)}/{c.get(4,0)}] "
                f"wr={c.get(1,0)/n:.3f} mr={mean_rank:.3f}+-{1.96*se:.3f} "
                f"rounds={n_rounds/n:.2f} "
                f"tsumo%={100*n_tsumo/total_ends:.0f} "
                f"ron%={100*n_ron/total_ends:.0f} "
                f"exh%={100*n_exh/total_ends:.0f} "
                f"ab%={100*n_ab/total_ends:.0f}")
        if n_rl:
            line += f" dng%={100.0*n_dng/max(n_rl,1):.2f}"
        if n_unh:
            line += f" !!UNHANDLED={n_unh}"
        return line


# ---------------------------------------------------------------------- #
# Entraînement
# ---------------------------------------------------------------------- #
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[train] GPU : {torch.cuda.get_device_name(0)}")
    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    h_auto = str(args.h_target).strip().lower() == 'auto'
    h_target = None if h_auto else float(args.h_target)
    h_calib = []
    alpha = args.alpha_init
    if args.entropy_mode == 'fixed':
        alpha = args.alpha_fixed

    agent = PPOAgent(
        args.sl_discard, args.sl_frozen, device=device,
        policy_lr=args.lr, critic_lr=args.critic_lr,
        oracle_init=args.oracle_init, oracle_scale=args.oracle_init_scale,
    )

    start_step = 0
    start_phase = 1
    if args.resume and os.path.isfile(args.resume):
        ck = agent.load(args.resume)
        start_step = int(ck.get('step', 0) or 0)
        alpha = float(ck.get('alpha', alpha))
        start_phase = int(ck.get('phase', 1))
        print(f"[train] resume : step={start_step} alpha={alpha:.4f} "
              f"phase={start_phase}", flush=True)

    if args.kl_ref:
        ref_ck = torch.load(args.kl_ref, map_location=device,
                            weights_only=False)
        agent.load_kl_ref(ref_ck['policy'])
        print(f"[train] ancre KL rechargée depuis {args.kl_ref}", flush=True)

    runner = VecRunner(
        agent, n_envs=args.n_envs, seed=args.seed,
        grp_path=args.grp_checkpoint,
        compute_lookahead=args.compute_lookahead,
        danger_labels=(args.danger_penalty > 0 or args.danger_labels),
        mode=args.mode, device=device,
    )

    window = StatsWindow(capacity=args.stats_window)
    base_out = args.output[:-3] if args.output.endswith('.pt') else args.output

    step = start_step
    n_updates = 0
    n_grad_steps = 0
    current_phase, _ = get_phase_and_gamma(
        step, args.total_steps, args.oracle_frac, args.transition_frac)
    critic_warmup_left = (args.critic_warmup_updates
                          if args.adv_mode == 'critic' else 0)
    recent_entropies = []
    t_start = time.time()
    n_games_total = 0
    last_eval_step = start_step

    print(f"[train] schedule : oracle={args.oracle_frac*100:.0f}% "
          f"transition={args.transition_frac*100:.0f}% "
          f"normal={(1-args.oracle_frac-args.transition_frac)*100:.0f}%",
          flush=True)
    if args.no_oracle:
        print("[train] NO-ORACLE : gamma=0 des le step 0, multiplicateurs "
              "LR forces a 1.0 (policy et critic), filtre IW desactive",
              flush=True)
    print(f"[train] mode={args.mode} n_envs={args.n_envs} "
          f"batch_games={args.batch_games} minibatch={args.minibatch} "
          f"epochs={args.ppo_epochs} kl_coef={args.kl_coef} "
          f"entropy={args.entropy_mode}", flush=True)
    if args.adv_mode == 'critic':
        print(f"[train] [V2] ADV-MODE critic : baseline V(s) asymétrique "
              f"(critic 283c voit l'oracle complet, policy jamais), "
              f"warmup={args.critic_warmup_updates} updates, "
              f"repli batch_mean si EV < {args.min_ev}", flush=True)
    if args.kl_coef > 0:
        print(f"[train] [V2] ANCRE KL vers le SL de départ : "
              f"coef={args.kl_coef} (réf = policy à l'init = SL)",
              flush=True)
    if args.target_kl > 0:
        print(f"[train] [V2] EARLY-STOP KL : cible={args.target_kl} "
              f"(epochs coupées si KL moyenne > "
              f"{1.5 * args.target_kl:.4f})", flush=True)
    if args.danger_labels and args.danger_penalty == 0:
        print("[train] [V2] DANGER-LABELS : dng% (deal-in potentiel) "
              "loggé sans pénalité de reward", flush=True)
    print(f"[train] SANITY ||W_o||={agent.oracle_weight_norm():.4f} "
          f"(init {agent.oracle_init_norm:.4f})", flush=True)

    while step < args.total_steps:
        phase, gamma_t = get_phase_and_gamma(
            step, args.total_steps, args.oracle_frac, args.transition_frac)
        if args.no_oracle:
            gamma_t = 0.0        # [NO-ORACLE] deploiement partout
        if phase != current_phase:
            print(f"[step {step:7d}] PHASE {current_phase} -> {phase} "
                  f"(gamma={gamma_t:.3f})", flush=True)
            current_phase = phase
            window.reset()
            recent_entropies.clear()

        # ---------------- collecte ON-POLICY ---------------- #
        t0 = time.time()
        episodes, game_stats = runner.collect_batch(
            args.batch_games, gamma_t,
            opponent_sample=(args.opponent_mode == 'sample'),
            danger_penalty=args.danger_penalty,
            grp_terminal_correction=not args.no_grp_terminal_correction,
            return_rho=args.return_rho)
        t_collect = time.time() - t0
        for st in game_stats:
            if st is not None:
                window.add(st)
        n_games_total += len(episodes)

        transitions = [tr for ep in episodes for tr in ep]
        if len(transitions) > args.max_batch_transitions:
            sub = np.random.choice(
                len(transitions), args.max_batch_transitions, replace=False)
            transitions = [transitions[i] for i in sub]
        if len(transitions) < 16:
            print(f"[train] batch trop petit ({len(transitions)} "
                  f"transitions), skip", flush=True)
            step += len(transitions)
            continue

        # ---------------- avantages ---------------- #
        warm = critic_warmup_left > 0
        adv_mode_now = ('critic' if (args.adv_mode == 'critic'
                                     and not warm) else 'batch_mean')
        _, adv_info = compute_advantages(
            agent, transitions, mode=adv_mode_now,
            reward_scale=args.reward_scale, min_ev=args.min_ev)

        # ---------------- update PPO ---------------- #
        metrics = ppo_update(
            agent, transitions,
            epochs=args.ppo_epochs, minibatch=args.minibatch,
            clip_eps=args.clip_eps, value_clip=args.value_clip,
            value_coef=args.value_coef,
            entropy_coef=(alpha if args.entropy_mode != 'off' else 0.0),
            kl_coef=args.kl_coef, grad_clip=args.grad_clip,
            reward_scale=args.reward_scale,
            iw_threshold=(None if args.no_oracle else (
                args.iw_threshold if phase == 3
                and args.iw_threshold > 0 else None)),
            warmup=warm,
            lr_policy=args.lr * phase_lr_mult(phase, args.no_oracle)[0],
            lr_critic=args.critic_lr * phase_lr_mult(
                phase, args.no_oracle)[1],
            target_kl=(args.target_kl if args.target_kl > 0 else None),
        )
        if metrics is None:
            continue
        if warm:
            critic_warmup_left -= 1
            if critic_warmup_left == 0:
                print(f"[step {step}] fin du warmup critic -> la policy "
                      f"s'entraîne avec A = G_t - V(s)", flush=True)

        step += len(transitions)
        n_updates += 1
        n_grad_steps += metrics['n_grad_steps']

        # ---------------- alpha adaptatif ---------------- #
        ent = metrics['entropy']
        if ent is not None:
            if h_target is None and args.entropy_mode == 'adaptive':
                h_calib.append(ent)
                if len(h_calib) >= args.h_calib_updates:
                    h_target = float(np.mean(h_calib)) * args.h_target_mult
                    print(f"[train] h_target auto = {h_target:.4f}",
                          flush=True)
            recent_entropies.append(ent)
            if len(recent_entropies) > 50:
                recent_entropies.pop(0)
            if (args.entropy_mode == 'adaptive' and h_target is not None
                    and len(recent_entropies) >= 3):
                h_bar = float(np.mean(recent_entropies))
                alpha = alpha + args.beta * (h_target - h_bar)
                alpha = float(np.clip(alpha, 0.0, args.alpha_max))

        # ---------------- logs ---------------- #
        if step % args.log_every < len(transitions):
            elapsed = time.time() - t_start
            wo = agent.oracle_weight_norm()
            rdev = metrics['ratio_dev']
            ev_str = (f" ev={adv_info['ev']:+.3f}"
                      f"({'critic' if adv_info['used_critic'] else 'mean'})"
                      ) if args.adv_mode == 'critic' else ''
            stop_str = ' KL-STOP' if metrics.get('early_stop') else ''
            print(f"[step {step:7d} | ph{phase} g={gamma_t:.2f}] "
                  f"pl={metrics['policy_loss']:+.4f} "
                  f"vl={metrics['value_loss']:.4f} "
                  f"H={metrics['entropy']:.3f} "
                  f"a={alpha:.4f} "
                  f"rdev={rdev:.4f} kl~{metrics['approx_kl']:+.4f} "
                  f"clip={metrics['clip_frac']:.2f} "
                  f"pgn={metrics['pg_norm']:.2f}{ev_str}{stop_str} "
                  f"||W_o||={wo:.3f} "
                  f"({elapsed:.0f}s, {n_games_total} games, "
                  f"{n_grad_steps} grad-steps)", flush=True)
            # [V2] accord top-1 vs SL (drift) + variance brute de
            # l'avantage (ce que la baseline n'a PAS absorbé)
            agr = agent.sl_agreement(transitions)
            agr_str = 'n/a' if agr is None else f"{agr:.1%}"
            if agr is not None and agr < 0.85:
                agr_str += '!!'
            print(f"    | DRIFT ret_mean={adv_info['ret_mean']:+.2f} "
                  f"ret_std={adv_info['ret_std']:.2f} "
                  f"advvar={adv_info['adv_var_raw']:.1f} "
                  f"agr={agr_str} "
                  f"upd={n_updates} collect={t_collect:.0f}s "
                  f"({len(transitions)} tr)", flush=True)
            game_line = window.format()
            if game_line:
                print(f"    | GAME: {game_line}", flush=True)
            n_unh = sum(s.get('n_unhandled', 0) for s in window.deque)
            if n_unh and not args.allow_unhandled:
                raise RuntimeError(
                    f"requêtes non gérées par le runner ({n_unh} sur la "
                    f"fenêtre) : parties invalides -- reporter le bug")

        # ---------------- eval inline (déploiement, gamma=0) ---------- #
        if (args.eval_every > 0 and args.eval_games > 0
                and step - last_eval_step >= args.eval_every):
            last_eval_step = step
            _, eval_stats = runner.collect_batch(
                args.eval_games, gamma_t,
                deterministic=True, collect=False, gamma_override=0.0)
            if eval_stats:
                ranks = [s['rank_of_seat0'] for s in eval_stats]
                print(f"    | EVAL gamma=0 argmax : n={len(ranks)} "
                      f"mean_rank={float(np.mean(ranks)):.3f} "
                      f"(2.50 = niveau SL)", flush=True)

        # ---------------- checkpoints ---------------- #
        if step % args.save_every < len(transitions):
            _save(agent, base_out, step, alpha, phase, args, metrics)
            _purge(base_out, args.keep_last_n)

    _save(agent, base_out, step, alpha, current_phase, args, None,
          final=True)
    print(f"[train] terminé ({step} steps, {n_updates} updates, "
          f"{n_grad_steps} pas de gradient). "
          f"Checkpoint : {base_out}_latest.pt", flush=True)


def _save(agent, base_out, step, alpha, phase, args, metrics, final=False):
    step_path = f"{base_out}_step{step:07d}.pt"
    agent.save(step_path, step=step, alpha=alpha, phase=phase)
    agent.save(f"{base_out}_latest.pt", step=step, alpha=alpha, phase=phase)
    if final:
        agent.save(f"{base_out}_final.pt", step=step, alpha=alpha,
                   phase=phase)
    print(f"[step {step}] save {step_path} + latest"
          + (" + final" if final else ""), flush=True)


def _purge(base_path, keep_n):
    if keep_n <= 0:
        return
    import glob
    files = sorted(glob.glob(f"{base_path}_step*.pt"))
    for f in files[:-keep_n]:
        try:
            os.remove(f)
        except OSError:
            pass


# ---------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.
                                RawDescriptionHelpFormatter)
    # --- chemins ---
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--grp-checkpoint', default=None,
                   help="prédicteur GRP ; sans lui, reward = score de "
                        "manche /1000")
    p.add_argument('--output', required=True)
    p.add_argument('--resume', default=None)
    p.add_argument('--kl-ref', default=None,
                   help="checkpoint de la policy qui sert d'ancre KL "
                        "(défaut : SL de départ)")

    # --- adversaires / collection ---
    p.add_argument('--mode', choices=['opponent', 'selfplay'],
                   default='opponent',
                   help="opponent : 1 siège RL vs 3 SL figés (défaut, "
                        "recommandé) ; selfplay : 4 sièges RL (papier)")
    p.add_argument('--opponent-mode', choices=['argmax', 'sample'],
                   default='argmax')
    p.add_argument('--n-envs', type=int, default=16,
                   help="nb de parties jouées en interleaved (batching "
                        "des forwards)")
    p.add_argument('--batch-games', type=int, default=32,
                   help="parties collectées par itération PPO")
    p.add_argument('--max-batch-transitions', type=int, default=4096)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--compute-lookahead', action='store_true',
                   help="réactive le DFS look-ahead (canaux de toute "
                        "façon morts, cf. ANALYSE_RL.md -- très lent)")

    # --- schedule oracle (papier §3.3) ---
    p.add_argument('--total-steps', type=int, default=1000000)
    p.add_argument('--oracle-frac', type=float, default=0.30)
    p.add_argument('--transition-frac', type=float, default=0.40)
    p.add_argument('--no-oracle', action='store_true',
                   help="[NO-ORACLE] coupe l'oracle des le step 0 (gamma=0 "
                        "partout) ET force les multiplicateurs LR a 1.0 "
                        "(policy et critic). Les frontieres de phases ne "
                        "servent plus qu'aux logs. Rend aussi le resume "
                        "avec un --total-steps plus grand sans danger.")
    p.add_argument('--oracle-init', choices=['zero', 'kaiming'],
                   default='kaiming')
    p.add_argument('--oracle-init-scale', type=float, default=0.25)
    p.add_argument('--iw-threshold', type=float, default=5.0,
                   help="IW rejection en phase 3 (0 = off). "
                        "Correction off-policy du dropout oracle (papier "
                        "§3.3) : sans oracle (--no-oracle), collection = "
                        "entraînement, le filtre n'a plus de sens et est "
                        "automatiquement désactivé.")

    # --- PPO ---
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--critic-lr', type=float, default=1e-3)
    p.add_argument('--ppo-epochs', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=512)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--value-clip', type=float, default=0.2)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--grad-clip', type=float, default=0.5)
    p.add_argument('--adv-mode', choices=['batch_mean', 'critic'],
                   default='batch_mean',
                   help="baseline des avantages : batch_mean (papier) ou "
                        "critic asymétrique V(s) (oracle complet)")
    p.add_argument('--critic-warmup-updates', type=int, default=30)
    p.add_argument('--reward-scale', type=float, default=10.0)
    p.add_argument('--min-ev', type=float, default=0.02)
    p.add_argument('--return-rho', type=float, default=0.85)

    # --- régularisation ---
    p.add_argument('--kl-coef', type=float, default=0.0,
                   help="ancre KL vers --kl-ref (DÉFAUT 0). [V2] avec la "
                        "baseline critic, 0.10 est le point de départ "
                        "recommandé (monter à 0.20 si H grimpe ou agr "
                        "< 90%% avant 100k steps). L'échec v10 (0.05 "
                        "annulait l'apprentissage) date de l'ancien "
                        "pipeline off-policy, sans early-stop ni critic.")
    p.add_argument('--target-kl', type=float, default=0.0,
                   help="[V2] early-stop PPO : les epochs restantes sont "
                        "coupées si KL moyenne > 1.5x cible (0 = off ; "
                        "0.015 recommandé). Limite la dérive PAR update, "
                        "complémentaire de l'ancre KL (dérive cumulée).")
    p.add_argument('--entropy-mode',
                   choices=['off', 'fixed', 'adaptive'], default='off')
    p.add_argument('--alpha-fixed', type=float, default=0.0)
    p.add_argument('--alpha-init', type=float, default=0.0)
    p.add_argument('--alpha-max', type=float, default=0.05)
    p.add_argument('--beta', type=float, default=0.001)
    p.add_argument('--h-target', type=str, default='auto')
    p.add_argument('--h-target-mult', type=float, default=1.0)
    p.add_argument('--h-calib-updates', type=int, default=5)

    # --- récompenses ---
    p.add_argument('--danger-penalty', type=float, default=0.0,
                   help="pénalité sur la défausse d'une tuile qui ferait "
                        "ron (label exact depuis les mains cachées)")
    p.add_argument('--danger-labels', action='store_true',
                   help="[V2] calcule les labels danger (dng%% = deal-in "
                        "potentiel dans les logs GAME) SANS pénalité de "
                        "reward — instrumentation gratuite, coût CPU "
                        "marginal")
    p.add_argument('--no-grp-terminal-correction', action='store_true',
                   help="désactive la correction terminale "
                        "(R_vrai - Phi_final) du shaping GRP")

    # --- logs / checkpoints ---
    p.add_argument('--log-every', type=int, default=8000)
    p.add_argument('--save-every', type=int, default=50000)
    p.add_argument('--keep-last-n', type=int, default=3)
    p.add_argument('--stats-window', type=int, default=400)
    p.add_argument('--eval-every', type=int, default=0,
                   help="éval argmax gamma=0 tous les N steps (0 = off)")
    p.add_argument('--eval-games', type=int, default=48)
    p.add_argument('--allow-unhandled', action='store_true')
    return p


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == '__main__':
    main()
