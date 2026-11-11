"""
Boucle principale d'entrainement RL Suphx.

[FIDELITE PAPIER §3.3] Oracle guiding en 3 phases :
    Phase 1 (oracle)     : gamma_t = 1 constant, LR normal
    Phase 2 (dropout)    : gamma_t : 1 -> 0 lineaire
    Phase 3 (normal)     : gamma_t = 0, LR / 10, IW rejection

[NIVEAU 2 / FIX IPC 2026-10-14] Les transitions passent par une mp.Queue
    au lieu d'une liste proxy Manager :
    - l'ancien chemin (4 workers ecrivant dans une ListProxy a chaque
      partie) tuait le serveur Manager sous Windows au bout de ~75 min
      de charge IPC ;
    - la Queue est le mecanisme prevu pour le transfert volumineux
      inter-processus (pipes dedies, backpressure via maxsize) ;
    - le Manager ne porte plus que l'etat leger : poids de policy
      publies, stats de jeu. Volume faible, frequence moderee.

[FIX CRASH] try/except autour de la boucle d'entrainement avec
    sauvegarde d'URGENCE des poids (step reel inclus). Si un crash IPC
    ou autre survient a 300k steps, le run est recuperable via --resume
    au lieu d'etre perdu integralement.

[FIX ALPHA] Cap sur le coefficient d'entropie (--alpha-max, defaut 0.15).
    Sans lui, quand H_target est inatteignable (une policy discard
    competente plafonne a H~1.7), alpha croit sans fin : sur le run
    600k precedent, alpha a fini a 0.41 et le bonus d'entropie dominait
    la policy loss par ~7x -> policy aplatie -> deal-ins -> 34 pourcent
    de 4es places contre 25 attendu (mean_rank 2.71 vs SL).
    Le papier ne plafonne pas alpha mais couvre 25M steps ; a notre
    echelle le cap est la protection necessaire. 0 = pas de cap.

[FIX RESUME] agent.save(path, step=step) partout : le checkpoint contient
    le step reel. Plus besoin du regex sur le nom de fichier pour le
    retrouver au resume.

[STATS] stats_proxy partage pour monitorer win rate, distribution de
    rangs, longueur des parties, distribution des fins de manche.

Usage :
    py -3.14 -m mahjong_ai.training.train_rl_suphx `
        --sl-discard checkpoints/sl_tonpusen_light/discard.pt `
        --sl-frozen checkpoints/sl_tonpusen_light `
        --grp-checkpoint checkpoints/grp_tonpusen.pt `
        --total-steps 500000 --n-workers 4 `
        --h-target 1.6 --alpha-max 0.15 --value-coef 0.0 `
        --oracle-frac 0.10 --transition-frac 0.30 `
        --save-every 25000 --keep-last-n 3 --publish-every 5000 `
        --output checkpoints/rl_tonpusen_v2/rl_discard.pt
"""
import argparse
import collections
import gc
import multiprocessing as mp
import os
import threading
import time

import numpy as np
import torch

try:                      # garde memoire optionnelle : pip install psutil
    import psutil
except ImportError:
    psutil = None

from mahjong_ai.agents.oracle_agent import OracleDiscardAgent
from mahjong_ai.training.parameter_server import (
    ParameterServer, publish_to_state,
)
from mahjong_ai.training.self_play_worker import worker_loop


# ---------------------------------------------------------------------- #
# Stats helpers
# ---------------------------------------------------------------------- #
STATS_KEYS_INIT = {
    'n_games': 0,
    'n_1st': 0, 'n_2nd': 0, 'n_3rd': 0, 'n_4th': 0,
    'sum_rank': 0.0,
    'n_rounds_total': 0,
    'n_honba_total': 0,
    'n_tsumo': 0, 'n_ron': 0, 'n_exhaustive': 0,
    'n_abortive': 0,
    'n_riichi': 0,
    'n_pon': 0, 'n_chi': 0, 'n_kan': 0,
    'n_ron_refused': 0,
    'n_sl_errors': 0,       # doit rester 0 (tete SL figee en erreur)
    'n_unhandled': 0,       # doit rester 0 (requete non-DISCARD inattendue)
    'n_rl_discards': 0,     # [DANGER] decisions du siege RL (si penalite)
    'n_danger_discards': 0,  # [DANGER] ... dont la tuile ferait ron
}


def reset_stats_proxy(stats_proxy):
    """Remet les compteurs a zero (utilise au changement de phase)."""
    for k, v in STATS_KEYS_INIT.items():
        stats_proxy[k] = v


def format_game_stats(stats_proxy):
    """Formate la ligne GAME pour les logs."""
    n_games = stats_proxy.get('n_games', 0)
    if n_games == 0:
        return None
    n1 = stats_proxy.get('n_1st', 0)
    n2 = stats_proxy.get('n_2nd', 0)
    n3 = stats_proxy.get('n_3rd', 0)
    n4 = stats_proxy.get('n_4th', 0)
    mean_rank = stats_proxy.get('sum_rank', 0.0) / n_games
    win_rate = n1 / n_games
    mean_rounds = stats_proxy.get('n_rounds_total', 0) / n_games
    mean_honba = stats_proxy.get('n_honba_total', 0) / n_games
    n_tsumo = stats_proxy.get('n_tsumo', 0)
    n_ron = stats_proxy.get('n_ron', 0)
    n_exh = stats_proxy.get('n_exhaustive', 0)
    n_ab = stats_proxy.get('n_abortive', 0)
    total_ends = n_tsumo + n_ron + n_exh + n_ab
    tsumo_pct = 100 * n_tsumo / max(total_ends, 1)
    ron_pct = 100 * n_ron / max(total_ends, 1)
    exh_pct = 100 * n_exh / max(total_ends, 1)
    ab_pct = 100 * n_ab / max(total_ends, 1)

    # [STATS] Taux par manche (4 joueurs confondus). n_rounds_total compte
    # les manches de toutes les parties.
    n_rounds_total = max(stats_proxy.get('n_rounds_total', 0), 1)
    riichi_rnd = stats_proxy.get('n_riichi', 0) / n_rounds_total
    calls_rnd = (stats_proxy.get('n_pon', 0) + stats_proxy.get('n_chi', 0)
                 + stats_proxy.get('n_kan', 0)) / n_rounds_total
    refused_rnd = stats_proxy.get('n_ron_refused', 0) / n_rounds_total

    line = (
        f"n={n_games} "
        f"[{n1}/{n2}/{n3}/{n4}] "
        # NB : sans --opponent-sl, wr/mean_rank concernent le siege 0 ; les 4
        # sieges partagent la meme policy -> 0.25 / 2.50 par symetrie, PAS
        # une mesure de progres. Avec --opponent-sl, ce sont les stats du
        # siege RL (siege tire au hasard) contre 3 SL : 2.50 = niveau SL,
        # < 2.50 = le RL progresse (bruit ~ +-0.07 pour 1000 parties).
        f"wr={win_rate:.3f} "
        f"mean_rank={mean_rank:.2f} "
        f"rounds={mean_rounds:.2f} "
        f"honba={mean_honba:.2f} "
        f"tsumo%={tsumo_pct:.0f} "
        f"ron%={ron_pct:.0f} "
        f"exh%={exh_pct:.0f} "
        f"ab%={ab_pct:.0f} "
        f"riichi/rnd={riichi_rnd:.2f} "
        f"calls/rnd={calls_rnd:.2f} "
        f"ron_refus/rnd={refused_rnd:.2f}"
    )
    n_rl = stats_proxy.get('n_rl_discards', 0)
    if n_rl:
        # [DANGER] % de defausses du siege RL qui feraient ron (tenpai
        # adverse, mains cachees). En phase 1 (oracle complet), la policy
        # peut lire le danger -> ce % doit CHUTER = preuve d'exploitation
        # de l'oracle. Reference SL : le niveau de la ligne de depart.
        line += (f" dng%={100 * stats_proxy.get('n_danger_discards', 0) / n_rl:.2f}")
    n_err = stats_proxy.get('n_sl_errors', 0)
    n_unh = stats_proxy.get('n_unhandled', 0)
    if n_err or n_unh:
        line += f" !!SL_ERR={n_err} !!UNHANDLED={n_unh}"
    return line


def _purge_old_checkpoints(base_path, keep_n):
    """Supprime les checkpoints _step* en gardant les keep_n plus recents."""
    import glob
    pattern = f"{base_path}_step*.pt"
    files = sorted(glob.glob(pattern))
    if len(files) > keep_n:
        for f in files[:-keep_n]:
            try:
                os.remove(f)
                print(f"[purge] supprime {f}", flush=True)
            except OSError:
                pass


# ---------------------------------------------------------------------- #
# Garde memoire
# ---------------------------------------------------------------------- #
def apply_mem_guard(rb_local, rb_lock, buf_cap, mem_pct, limit_pct, floor):
    """Reduit durablement la capacite du buffer si la RAM depasse le seuil.

    buf_cap : liste a 1 element (capacite en parties, lue par le thread de
              drain). Ne remonte jamais d'elle-meme.
    Retourne (action, n_supprimees, capacite) ; action = None (RAM ok),
    'shrunk' (buffer reduit) ou 'at_floor' (deja au plancher).
    """
    if mem_pct is None or mem_pct < limit_pct:
        return None, 0, buf_cap[0]
    with rb_lock:
        new_cap = max(floor, len(rb_local) // 2)
        if new_cap < buf_cap[0]:
            buf_cap[0] = new_cap
            n_drop = max(0, len(rb_local) - new_cap)
            del rb_local[:n_drop]
            return 'shrunk', n_drop, new_cap
    return 'at_floor', 0, buf_cap[0]


# ---------------------------------------------------------------------- #
# Drift tracking helpers
# ---------------------------------------------------------------------- #
def compute_drift_metrics(batch):
    """Calcule les metriques de suivi de derive depuis un batch."""
    if not batch:
        return None

    # [FIX] en mode --opponent-sl le siege RL est tire au hasard : filtrer
    # sur le siege 0 donnait souvent 0 action (fausse alerte COLLAPSE).
    actions = [t['action'] for t in batch]
    if actions:
        counter = collections.Counter(actions)
        top3 = counter.most_common(3)
        top3_str = " ".join(f"{a}:{c}" for a, c in top3)
        counts = np.array(list(counter.values()), dtype=np.float32)
        probs = counts / counts.sum()
        action_entropy = float(-(probs * np.log(probs + 1e-8)).sum())
        max_freq = float(probs.max())
        action_div = 1.0 - max_freq
    else:
        top3_str = "-"
        action_entropy = 0.0
        action_div = 0.0

    advs = np.array([t['advantage'] for t in batch], dtype=np.float32)
    vals = np.array([t['value'] for t in batch], dtype=np.float32)
    rets = np.array([t['return'] for t in batch], dtype=np.float32)

    return {
        'actions_top3':   top3_str,
        'action_entropy': action_entropy,
        'action_div':     action_div,
        'adv_mean':       float(advs.mean()),
        'adv_std':        float(advs.std()),
        'adv_abs_mean':   float(np.abs(advs).mean()),
        'val_mean':       float(vals.mean()),
        'val_std':        float(vals.std()),
        'ret_mean':       float(rets.mean()),
        'n_valid':        len(batch),
    }


def check_drift_alerts(metrics, phase):
    if metrics is None:
        return []
    alerts = []
    if metrics['action_div'] < 0.2:
        alerts.append(f"COLLAPSE_ACTION (div={metrics['action_div']:.2f})")
    if metrics['action_entropy'] < 0.5:
        alerts.append(f"LOW_ACTION_H ({metrics['action_entropy']:.2f})")
    if metrics['adv_abs_mean'] > 10.0:
        alerts.append(f"BIG_ADV ({metrics['adv_abs_mean']:.1f})")
    if metrics['val_std'] < 0.1 and phase >= 2:
        alerts.append(f"FLAT_VALUE (std={metrics['val_std']:.3f})")
    gap = abs(metrics['val_mean'] - metrics['ret_mean'])
    if gap > 5.0:
        alerts.append(f"VAL_RET_GAP ({gap:.1f})")
    return alerts


# ---------------------------------------------------------------------- #
# Schedule 3 phases
# ---------------------------------------------------------------------- #
def get_phase_and_gamma(step, total_steps,
                         oracle_frac=0.20, transition_frac=0.50):
    p1_end = int(total_steps * oracle_frac)
    p2_end = int(total_steps * (oracle_frac + transition_frac))
    if step < p1_end:
        return 1, 1.0, 1.0
    elif step < p2_end:
        frac = (step - p1_end) / max(p2_end - p1_end, 1)
        return 2, max(0.0, 1.0 - frac), 1.0
    else:
        return 3, 0.0, 0.1


# ---------------------------------------------------------------------- #
# GAE
# ---------------------------------------------------------------------- #
class Episode(list):
    """Une partie (liste de transitions) + compteur d'utilisations."""
    uses = 0


def _critic_values(agent, transitions, reward_scale, chunk=512):
    """V(s) du critic (oracle complet) en unites de reward, sans gradient."""
    out = []
    for i in range(0, len(transitions), chunk):
        part = transitions[i:i + chunk]
        on = torch.as_tensor(np.stack([t['obs_normal'] for t in part]),
                             dtype=torch.float32, device=agent.device)
        oo = torch.as_tensor(
            np.stack([t.get('obs_oracle_full', t['obs_oracle'])
                      for t in part]),
            dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            v = agent.critic(torch.cat([on, oo], dim=1))
        out.append(v.cpu().numpy())
    return np.concatenate(out).astype(np.float32) * reward_scale


def advantages_from_values(rewards, values, min_ev=0.0):
    """Avantage = R - V(s) (baseline apprise) ; repli sur R - mean(R) si le
    critic n'explique pas la variance (EV < min_ev).

    Retourne (advs_bruts, ev, used_critic). Fonction pure (numpy) : testable.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    var_r = float(rewards.var())
    if values is None or var_r < 1e-8:
        return rewards - rewards.mean(), 0.0, False
    resid = rewards - np.asarray(values, dtype=np.float32)
    ev = 1.0 - float(resid.var()) / var_r
    if ev < min_ev:
        return rewards - rewards.mean(), ev, False
    return resid, ev, True


def compute_advantages(transitions, gamma=0.99, lam=0.95, agent=None,
                       adv_mode='batch_mean', reward_scale=10.0, min_ev=0.0):
    """Avantages des transitions.

    adv_mode='batch_mean' (ancien comportement, papier Suphx) :
        A = R - mean(R_batch)
    adv_mode='critic' :
        [V9] A = G_t - V(s_t). G_t = retour RESTANT de partie (somme
        escomptee des rewards des manches futures, cf.
        _finalize_episode_returns), pose par le worker sur chaque
        transition. V = critic asymetrique (voit l'oracle complet),
        entraine a predire G_t / reward_scale.

        Pourquoi le changement de cible : le delta GRP de la MANCHE
        (cible v8) est determine par l'issue de la manche (le futur)
        -> E[R|s] plat (probe 200k : EV=0.0000, val_std=0.0001) ->
        le critic degenerait en constante, aucune reduction de
        variance. G_t est previsible depuis l'etat (own/opp scores en
        buckets, manche, honba + oracle sont dans les 283 canaux) ->
        le critic reduit VRAIMENT la variance, surtout en fin de
        partie ou le placement se joue. A = G_t - V(s_t) reste sans
        biais : V ne depend que de s.

    Positionne compute_advantages.last_ev / .last_used_critic (logs).
    """
    # [V9] Cible du critic / avantage : retour restant G_t (repli :
    # reward de manche pour les episodes d'avant le patch)
    returns = np.array([t.get('return', t['reward']) for t in transitions],
                       dtype=np.float32)

    values = None
    if adv_mode == 'critic' and agent is not None:
        values = _critic_values(agent, transitions, reward_scale)
    advs, ev, used = advantages_from_values(returns, values, min_ev)
    compute_advantages.last_ev = ev
    compute_advantages.last_used_critic = used

    # Normalisation (comme avant)
    if len(advs) > 1 and advs.std() > 1e-6:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    for i, tr in enumerate(transitions):
        tr['advantage'] = advs[i]
        tr['return'] = returns[i]

    return advs


compute_advantages.last_ev = 0.0
compute_advantages.last_used_critic = False

# ---------------------------------------------------------------------- #
# PG + IS loss
# ---------------------------------------------------------------------- #
def pg_is_loss(agent, transitions, advantages, gamma_t,
                clip_eps=0.2, iw_threshold=None, phase=1,
                critic_mode=False, reward_scale=10.0,
                danger_coef=0.0, ablate=False):
    obs_n = torch.as_tensor(np.stack([t['obs_normal'] for t in transitions]),
                             dtype=torch.float32, device=agent.device)
    obs_o = torch.as_tensor(np.stack([t['obs_oracle'] for t in transitions]),
                            dtype=torch.float32, device=agent.device)
    masks = torch.as_tensor(np.stack([t['mask'] for t in transitions]),
                            dtype=torch.float32, device=agent.device)
    actions = torch.as_tensor([t['action'] for t in transitions],
                                dtype=torch.long, device=agent.device)
    old_logp = torch.as_tensor([t['logp'] for t in transitions],
                                 dtype=torch.float32, device=agent.device)
    returns = torch.as_tensor([t['return'] for t in transitions],
                                dtype=torch.float32, device=agent.device)
    advs = torch.as_tensor(advantages, dtype=torch.float32, device=agent.device)

    oof = None
    if critic_mode:
        oof = np.stack([t.get('obs_oracle_full', t['obs_oracle'])
                        for t in transitions])
    dng = None
    if danger_coef > 0 and all(t.get('danger') is not None
                               for t in transitions):
        dng = np.stack([t['danger'] for t in transitions]).astype(np.float32)
    new_logp, values, entropy = agent.evaluate_actions(
        obs_n, obs_o, masks, actions, gamma_t, obs_o_full=oof,
        danger=dng, ablate=ablate)
    ratio = torch.exp(new_logp - old_logp)
    # [DIAG] ecart moyen du ratio a 1 (lu par la boucle a l'epoch 0).
    # NB : non nul meme sans bug, car le buffer est un replay et les
    # workers jouent avec des poids publies tous les --publish-every steps.
    agent.last_ratio_dev = float((ratio.detach() - 1.0).abs().mean().item())

    if phase == 3 and iw_threshold is not None:
        keep = (ratio <= iw_threshold).float()
        n_kept = keep.sum().item()
        if n_kept < 2:
            return None, None, None
        keep = keep / keep.sum() * keep.numel()
    else:
        keep = torch.ones_like(ratio)

    if clip_eps is not None:
        s1 = ratio * advs
        s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
        policy_loss = -(torch.min(s1, s2) * keep).mean()
    else:
        policy_loss = -(ratio * advs * keep).mean()

    if critic_mode:
        # cible en unites normalisees (G_t / reward_scale) : le critic
        # predit le retour restant de partie, pas le delta de manche
        value_loss = ((values - returns / reward_scale) ** 2 * keep).mean()
    else:
        value_loss = ((values - returns) ** 2 * keep).mean()
    entropy_bonus = entropy.mean()
    return policy_loss, value_loss, entropy_bonus


# ---------------------------------------------------------------------- #
# Boucle principale
# ---------------------------------------------------------------------- #
def apply_phase1_guards(args):
    """[PHASE 1 GUARD] protege un run oracle-complet de l'erreur v10/v11.

    ref_policy (l'ancre KL) est une copie du SL de depart : ses canaux
    oracle sont ceux de --oracle-init (zero = totalement aveugle). En
    phase 1 (--oracle-frac 1.0) la policy DOIT s'ecarter du SL
    precisement la ou l'oracle revele un danger que le SL ne voit pas ->
    toute valeur kl_coef > 0 punit cet ecart et referme le pathway par
    les gradients (v10 : ||W_o|| 4.0 -> 0.18 avec kl=0.05 ; v11 lance
    avec 0.10, soit 2x plus vite encore). On force kl_coef=0, sauf
    opt-out explicite --allow-phase1-kl.
    """
    if (args.oracle_frac >= 1.0 and args.kl_coef > 0
            and not getattr(args, 'allow_phase1_kl', False)):
        print(f"[GUARD PHASE 1] --oracle-frac {args.oracle_frac:.2f} avec "
              f"--kl-coef {args.kl_coef} : l'ancre KL est le SL de depart, "
              f"aveugle a l'oracle -> effondrement ||W_o|| assure (cf. v10 "
              f"4.0 -> 0.18). kl_coef force a 0.0 (--allow-phase1-kl pour "
              f"conserver la valeur).", flush=True)
        args.kl_coef = 0.0


def train(args):
    apply_phase1_guards(args)   # [PHASE 1 GUARD] avant tout le reste
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[train_rl] GPU detected : {torch.cuda.get_device_name(0)}")
        print(f"[train_rl] VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    os.environ["RL_N_WORKERS"] = str(args.n_workers)

    torch.set_num_threads(2)

    print(f"[train_rl] device={device}, n_workers={args.n_workers}",
          flush=True)
    print(f"[train_rl] schedule 3 phases : "
          f"oracle={args.oracle_frac*100:.0f}%, "
          f"transition={args.transition_frac*100:.0f}%, "
          f"normal={(1-args.oracle_frac-args.transition_frac)*100:.0f}%",
          flush=True)
    print(f"[train_rl] alpha : init={args.alpha_init}, max={args.alpha_max}, "
          f"h_target={args.h_target}, beta={args.beta}",
          flush=True)

    # [H-TARGET AUTO] --h-target auto : la cible est l'entropie MESUREE de
    # la policy au demarrage (moyenne des --h-calib-updates premiers
    # updates, avant tout pas de gradient significatif), x --h-target-mult.
    # Evite une cible inatteignable qui pousse alpha au cap en permanence
    # (cf. logs : H 1.03 -> 1.65 avec h_target=1.6 et alpha colle a 0.15).
    h_auto = str(args.h_target).strip().lower() == 'auto'
    h_target = None if h_auto else float(args.h_target)
    h_calib = []

    # --- Parameter server (etat LEGER : poids, alpha, gamma) + Queue ---
    ps = ParameterServer()
    stats_proxy = ps.manager.dict()
    reset_stats_proxy(stats_proxy)

    # [NIVEAU 2 / FIX IPC] Les transitions passent par une mp.Queue :
    # mecanisme prevu pour le transfert volumineux inter-processus (pipes
    # dedies + backpressure via maxsize). L'ancien chemin -- liste proxy
    # Manager sollicitee par 4 workers a chaque partie -- tuait le serveur
    # Manager sous Windows au bout de ~75 min de charge IPC. Le Manager ne
    # porte plus desormais que l'etat leger (poids, stats).
    #
    # [FIX MEMOIRE] La Queue ne doit PAS etre dimensionnee comme le buffer :
    # chaque element est une PARTIE entiere (~400 coups x 268x34 float32 =
    # ~15 Mo) et le feeder thread de chaque worker garde une copie non
    # serialisee de tout ce qui attend d'entrer dans le pipe. Avec
    # maxsize=500 on pouvait empiler plusieurs Go cote workers EN PLUS du
    # buffer du trainer (MemoryError a ~2.46M steps). Le thread de drain
    # vide la Queue en continu : quelques elements suffisent.
    queue_size = args.queue_size or max(4, 2 * args.n_workers)
    transitions_queue = mp.Queue(maxsize=queue_size)

    # Buffer LOCAL au process principal : liste ordinaire protegee par un
    # lock, remplie par un thread daemon qui draine la Queue.
    rb_local = []
    rb_lock = threading.Lock()
    # Capacite du buffer (en parties), modifiable par la garde memoire.
    buf_cap = [args.buffer_size]

    import queue as _queue_mod

    def _drain_queue(q, rb_list, lock, cap):
        """Draine la Queue vers le buffer local (thread daemon).

        get(timeout=1.0) : le thread verifie periodiquement s'il doit
        s'arreter (Queue fermee en fin de run) sans polling actif.
        """
        while True:
            try:
                item = q.get(timeout=1.0)
                with lock:
                    rb_list.append(Episode(item))
                    while len(rb_list) > cap[0]:
                        rb_list.pop(0)
            except _queue_mod.Empty:
                continue
            except Exception:
                # Queue fermee (fin de run) ou pipe casse : arret propre.
                return

    drain_thread = threading.Thread(
        target=_drain_queue,
        args=(transitions_queue, rb_local, rb_lock, buf_cap),
        daemon=True,
    )
    drain_thread.start()

    # [FIX MEMOIRE] garde RAM (optionnelle, necessite psutil)
    def _mem_percent():
        return psutil.virtual_memory().percent if psutil is not None else None

    print(f"[train_rl] memoire : buffer={args.buffer_size} parties, "
          f"queue={queue_size}, garde RAM "
          f"{'>= %.0f%%' % args.mem_limit_pct if psutil else 'INACTIVE (pip install psutil)'}",
          flush=True)

    # --- Agent ---
    agent = OracleDiscardAgent(
        sl_discard_path=args.sl_discard,
        sl_frozen_dir=args.sl_frozen,
        device=device,
        policy_lr=args.lr,
        critic_lr=args.critic_lr,
        # [FIX ORACLE TAKEOFF] init non nulle des canaux oracle + suivi
        oracle_init=args.oracle_init,
        oracle_scale=args.oracle_init_scale,
    )

    # --- Resume (optionnel) ---
    start_step = 0
    start_alpha = args.alpha_resume if args.alpha_resume is not None \
                  else args.alpha_init
    if args.resume and os.path.isfile(args.resume):
        print(f"[train_rl] resume depuis {args.resume}", flush=True)
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            agent.policy.load_state_dict(ck['policy'])
            agent.critic.load_state_dict(ck['critic'])
            start_step = ck.get('step', 0) or 0
            # [FIX ALPHA-RESUME] priorite : --alpha-resume explicite >
            # alpha du checkpoint > --alpha-init.
            if args.alpha_resume is not None:
                start_alpha = args.alpha_resume
                print(f"[train_rl]   alpha : --alpha-resume "
                      f"({start_alpha:.4f}) prioritaire", flush=True)
            elif ck.get('alpha') is not None:
                start_alpha = float(ck['alpha'])
                print(f"[train_rl]   alpha : repris du checkpoint "
                      f"({start_alpha:.4f})", flush=True)
            else:
                print(f"[train_rl]   alpha : absent du checkpoint (ancien "
                      f"format) -> {start_alpha:.4f} ; utiliser "
                      f"--alpha-resume pour le fixer", flush=True)

            # Extraction depuis le nom si step=0 (anciens checkpoints)
            if start_step == 0:
                import re
                m = re.search(r'step(\d+)', args.resume)
                if m:
                    start_step = int(m.group(1))
                    print(f"[train_rl]   step extrait du nom : {start_step}",
                          flush=True)

            print(f"[train_rl]   -> step={start_step}, alpha={start_alpha:.4f}",
                  flush=True)
            # [PHASES] enchainer les phases proprement : un milestone de
            # fin de phase 1 porte un step eleve qui fausserait le
            # schedule gamma/lr de la phase suivante (gamma deja parti a
            # 0.x). --reset-step garde les poids, repart a step=0.
            if args.reset_step:
                print(f"[train_rl]   --reset-step : step {start_step} -> 0 "
                      f"(schedule de phase repart du debut)", flush=True)
                start_step = 0
            # [FIX ORACLE TAKEOFF] la reference ||W_o|| apres resume est
            # celle des poids RECHARgES (pas celle de l'init kaiming).
            agent.oracle_init_norm = agent.oracle_weight_norm()
            print(f"[train_rl]   ||W_oracle|| (resume) = "
                  f"{agent.oracle_init_norm:.4f}", flush=True)
        except Exception as e:
            print(f"[train_rl] ECHEC resume : {e}", flush=True)

    # [ANCRE KL phase 2/3] --kl-ref : charge la reference KL depuis un
    # checkpoint (la policy de phase 1, oracle-aware) au lieu du SL de
    # depart, aveugle a l'oracle. Inutilise en phase 1 (kl_coef=0) mais
    # decisif en phase 2/3 : sans lui, la penalite KL re-importe la
    # pression anti-oracle du SL qui a fait chuter ||W_o|| en v10.
    if args.kl_ref:
        ref_ck = torch.load(args.kl_ref, map_location=device,
                            weights_only=False)
        agent.load_kl_ref(ref_ck['policy'])

    # [FIX BATCHNORM] Policy en eval : stats BN du SL figees, identiques
    # entre rollout (batch 1) et update (batch ~1000).
    agent.set_eval_mode()
    print(f"[train_rl] policy.training={agent.policy.training} "
          f"(doit etre False)", flush=True)
    if args.resume:
        print("[train_rl] ATTENTION : un checkpoint RL produit AVANT ce fix "
              "contient des running stats BN polluees par le mode train. "
              "Repartir du SL (sans --resume) pour un test propre.",
              flush=True)

    # [SANITY t=0] banniere de controle : tout ce qui doit etre vrai au
    # demarrage d'un run de phase 1 (oracle complet) depuis le warmup.
    print(f"[SANITY t=0] oracle_frac={args.oracle_frac:.2f} "
          f"transition_frac={args.transition_frac:.2f} "
          f"kl_coef={args.kl_coef} alpha={start_alpha:.4f} "
          f"alpha_max={args.alpha_max} h_target={args.h_target}", flush=True)
    print(f"[SANITY t=0] ||W_o||={agent.oracle_init_norm:.4f} "
          f"(attendu : warmup ~13.57, SL+kaimingx0.25 ~4.0, zero ~0.0) "
          f"resume={args.resume} kl_ref={args.kl_ref}", flush=True)

    # --- Publish initial ---
    publish_to_state(
        ps.state, ps.lock,
        agent.policy.state_dict(),
        agent.critic.state_dict(),
        step=start_step, alpha=start_alpha, gamma_t=1.0,
    )

    # --- Workers : la Queue remplace la liste proxy ---
    workers = []
    for i in range(args.n_workers):
        p = mp.Process(
            target=worker_loop,
            args=(
                i,
                ps.state, ps.lock,
                transitions_queue,          # [NIVEAU 2] Queue au lieu de rb.data
                stats_proxy,
                args.sl_discard, args.sl_frozen,
                args.grp_checkpoint,
                args.games_per_cycle, device,
                args.opponent_sl, args.opponent_mode,
                args.danger_penalty,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)
    print(f"[train_rl] {args.n_workers} workers lances", flush=True)

    # --- Boucle ---
    step = start_step
    alpha = start_alpha
    recent_entropies = []
    # Si on resume dans une phase avancee, on demarre dans cette phase
    current_phase, _, _ = get_phase_and_gamma(
        step, args.total_steps, args.oracle_frac, args.transition_frac)
    base_lr = args.lr
    agent.compute_kl = args.kl_coef > 0
    print(f"[train_rl] ancre KL vers le SL : coef={args.kl_coef} "
          f"({'ON' if agent.compute_kl else 'OFF'}) ; checkpoints jalons "
          f"tous les {args.milestone_every} steps", flush=True)
    last_milestone = start_step // args.milestone_every \
        if args.milestone_every > 0 else 0
    # [FIX LR-RESUME] Les facteurs de LR n'etaient appliques qu'au CHANGEMENT
    # de phase : un resume directement en phase 2/3 gardait le LR plein.
    # L'optimizer n'est pas sauvegarde -> on applique le facteur ici.
    if current_phase != 1:
        _, _, _lrf = get_phase_and_gamma(
            step, args.total_steps, args.oracle_frac, args.transition_frac)
        _cf = {1: 1.0, 2: 0.5, 3: 0.1}[current_phase]
        for pg in agent.policy_optim.param_groups:
            pg['lr'] = base_lr * _lrf
        for pg in agent.critic_optim.param_groups:
            pg['lr'] = args.critic_lr * _cf
        print(f"[train_rl] resume en phase {current_phase} : "
              f"policy lr={base_lr * _lrf:.2e} (x{_lrf}), "
              f"critic lr={args.critic_lr * _cf:.2e}", flush=True)
    t_start = time.time()
    n_updates = 0

    # [FIX CRASH] Chemin de base pour les sauvegardes (normales et
    # d'urgence) -- calcule une fois, hors de la boucle.
    base_out = args.output[:-3] if args.output.endswith('.pt') else args.output

    last_mem_warn = 0.0
    try:
        critic_mode = (args.adv_mode == 'critic')
        if critic_mode and args.value_coef <= 0:
            print("[train_rl] --adv-mode critic : value-coef force a 1.0 "
                  "(le critic doit etre entraine)", flush=True)
            args.value_coef = 1.0
        warm_until = start_step + (args.critic_warmup_steps
                                   if critic_mode else 0)
        warm_announced = [False]
        if critic_mode:
            print(f"[train_rl] baseline critic : warmup critic seul jusqu'au "
                  f"step {warm_until} (policy = SL), reward_scale="
                  f"{args.reward_scale}, repli batch-mean si EV<"
                  f"{args.min_ev}", flush=True)
        while step < args.total_steps:
            # [FIX MEMOIRE] Garde RAM : si la memoire systeme depasse le
            # seuil, on reduit durablement la capacite du buffer (et on le
            # tronque) plutot que de mourir en MemoryError. Plancher : 2x
            # batch_episodes, pour ne jamais affamer l'entrainement.
            mem_pct = _mem_percent()
            guard, n_drop, new_cap = apply_mem_guard(
                rb_local, rb_lock, buf_cap, mem_pct, args.mem_limit_pct,
                floor=max(2 * args.batch_episodes, 1))
            if guard == 'shrunk':
                gc.collect()
                print(f"[mem] RAM {mem_pct:.0f}% >= "
                      f"{args.mem_limit_pct:.0f}% : capacite buffer -> "
                      f"{new_cap} parties ({n_drop} supprimees)", flush=True)
            elif guard == 'at_floor' and time.time() - last_mem_warn > 60:
                last_mem_warn = time.time()
                print(f"[mem] RAM {mem_pct:.0f}% : buffer deja au plancher "
                      f"({buf_cap[0]} parties) -- fermer d'autres programmes "
                      f"ou baisser --n-workers", flush=True)

            # [NIVEAU 2] Echantillonnage depuis le buffer local.
            # [NIVEAU 2] Echantillonnage depuis le buffer local.
            # [FIX FORMAT] rb_local contient des EPISODES (listes de
            # transitions, comme l'ancien buffer). L'ancien
            # ReplayBuffer.sample() aplatisait : [tr for ep in batch
            # for tr in ep]. Le nouveau chemin Queue doit faire pareil --
            # compute_advantages et pg_is_loss attendent une liste PLATE
            # de dicts de transitions.
            with rb_lock:
                buf_len = len(rb_local)
                if buf_len < args.batch_episodes:
                    time.sleep(1.0)
                    continue
                idx = np.random.choice(
                    buf_len, size=min(args.batch_episodes, buf_len),
                    replace=False)
                batch = [rb_local[i] for i in idx]
                # [REPLAY] limite le nb de reutilisations d'une partie
                # (sinon chaque transition est rejouee ~20x et la policy /
                # le critic ajustent le bruit). Les parties epuisees sont
                # retirees ; si le buffer devient trop petit, la boucle
                # attend de nouvelles parties.
                if args.max_episode_reuse > 0:
                    for ep in batch:
                        ep.uses += 1
                    rb_local[:] = [ep for ep in rb_local
                                   if ep.uses < args.max_episode_reuse]
            # Aplatissement episodes -> transitions (memes semantiques que
            # l'ancien ReplayBuffer.sample : le done=True de fin de partie
            # coupe correctement le bootstrap GAE entre episodes).
            batch = [tr for ep in batch for tr in ep]

            if not batch:
                continue
            if len(batch) > args.max_batch_transitions:
                sub = np.random.choice(len(batch), args.max_batch_transitions,
                                        replace=False)
                batch = [batch[i] for i in sub]

            phase, gamma_t, lr_factor = get_phase_and_gamma(
                step, args.total_steps, args.oracle_frac, args.transition_frac)

            # --- Changement de phase ---
            if phase != current_phase:
                print(f"[step {step:7d}] PHASE {current_phase} -> {phase} "
                      f"(gamma={gamma_t:.3f}, lr_factor={lr_factor})",
                      flush=True)
                current_phase = phase

                # Policy LR : base * lr_factor (1.0 en phase 2, 0.1 en phase 3)
                for pg in agent.policy_optim.param_groups:
                    pg['lr'] = base_lr * lr_factor

                # Critic LR : reduction independante de la policy
                critic_factor = {1: 1.0, 2: 0.5, 3: 0.1}[phase]
                for pg in agent.critic_optim.param_groups:
                    pg['lr'] = args.critic_lr * critic_factor

                # [STATS] reset pour mesurer par phase
                reset_stats_proxy(stats_proxy)
                print(f"[step {step}] stats reset pour phase {phase}",
                      flush=True)

            # --- Update ---
            warm = critic_mode and step < warm_until
            if critic_mode and not warm and not warm_announced[0]:
                warm_announced[0] = True
                print(f"[step {step}] fin du warmup critic : la policy "
                      f"s'entraine maintenant avec A = R - V(s)", flush=True)
            advantages = compute_advantages(
                batch, gamma=args.gamma, lam=args.lam, agent=agent,
                adv_mode=('critic' if critic_mode and not warm
                          else 'batch_mean'),
                reward_scale=args.reward_scale, min_ev=args.min_ev)

            # [FIX CRITIC BUDGET] Le compteur de steps compte des
            # TRANSITIONS consommees, pas des mises a jour : 2000 steps
            # = 2 updates x 3 epochs = 6 pas de gradient au total. V8
            # n'a donne que ~800 pas de gradient au critic en 228k
            # steps (dont ~300 pendant tout le warmup 50k) : budget
            # ~100x trop petit pour apprendre V(s) (val_std ~ 0,
            # FLAT_VALUE permanent). Inner loop : K pas supplementaires
            # par update, en minibatch aleatoire pour limiter l'overfit
            # des ~60-100 cibles distinctes du batch. Les avantages ont
            # deja ete calcules ci-dessus : cette boucle n'affecte pas
            # la policy, elle enrichit seulement le critic.
            if critic_mode and args.critic_inner_steps > 0:
                on_all = torch.as_tensor(
                    np.stack([t['obs_normal'] for t in batch]),
                    dtype=torch.float32, device=agent.device)
                oo_all = torch.as_tensor(
                    np.stack([t.get('obs_oracle_full', t['obs_oracle'])
                              for t in batch]),
                    dtype=torch.float32, device=agent.device)
                ret_all = torch.as_tensor(
                    [t['return'] for t in batch],
                    dtype=torch.float32, device=agent.device
                ) / args.reward_scale
                mb = min(256, len(batch))
                for _ in range(args.critic_inner_steps):
                    sel = torch.randint(0, len(batch), (mb,),
                                        device=agent.device)
                    agent.critic_optim.zero_grad()
                    v = agent.critic(
                        torch.cat([on_all[sel], oo_all[sel]], dim=1))
                    vloss = ((v - ret_all[sel]) ** 2).mean()
                    vloss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.critic.parameters(), 5.0)
                    agent.critic_optim.step()

            # Multi-epoch PPO : plusieurs passes de gradient par batch
            PPO_EPOCHS = args.ppo_epochs
            policy_loss_last = None
            value_loss_last = None
            entropy_last = None
            policy_gnorm_last = None
            critic_gnorm_last = None

            entropy_first = None
            ratio_dev_first = None
            kl_last = None
            pdanger_last = None
            for epoch_i in range(PPO_EPOCHS):
                agent.policy_optim.zero_grad()
                agent.critic_optim.zero_grad()
                iw_thresh = args.iw_threshold if phase == 3 else None
                policy_loss, value_loss, entropy = pg_is_loss(
                    agent, batch, advantages, gamma_t,
                    clip_eps=args.clip_eps, iw_threshold=iw_thresh, phase=phase,
                    critic_mode=critic_mode, reward_scale=args.reward_scale,
                    danger_coef=args.danger_loss_coef,
                    ablate=(epoch_i == 0 and args.ablate_every > 0
                            and n_updates % args.ablate_every == 0))
                if policy_loss is None:
                    break
                if epoch_i == 0:
                    entropy_first = entropy.item()
                    ratio_dev_first = getattr(agent, 'last_ratio_dev', None)
                if warm:
                    # warmup critic : seule la value loss est optimisee,
                    # la policy reste le SL.
                    loss = args.value_coef * value_loss
                else:
                    loss = (policy_loss + args.value_coef * value_loss
                            - alpha * entropy)
                kl_t = getattr(agent, 'last_kl', None)
                if (not warm) and args.kl_coef > 0 and kl_t is not None:
                    loss = loss + args.kl_coef * kl_t
                    kl_last = float(kl_t.detach().item())
                # [DANGER LOSS] penalite analytique E_pi[danger]
                pd_t = getattr(agent, 'last_p_danger', None)
                if (not warm) and args.danger_loss_coef > 0 \
                        and pd_t is not None:
                    loss = loss + args.danger_loss_coef * pd_t.mean()
                    pdanger_last = float(pd_t.detach().mean().item())
                loss.backward()
                policy_gnorm_last = torch.nn.utils.clip_grad_norm_(
                    agent.policy.parameters(), 5.0)
                critic_gnorm_last = torch.nn.utils.clip_grad_norm_(
                    agent.critic.parameters(), 5.0)
                if not warm:
                    agent.policy_optim.step()
                agent.critic_optim.step()
                policy_loss_last = policy_loss
                value_loss_last = value_loss
                entropy_last = entropy

            if policy_loss_last is None:
                continue

            # [H-TARGET AUTO] calibration sur l'entropie de depart
            if h_target is None and entropy_first is not None:
                h_calib.append(entropy_first)
                if len(h_calib) >= args.h_calib_updates:
                    h_target = float(np.mean(h_calib)) * args.h_target_mult
                    print(f"[train_rl] h_target auto = {h_target:.4f} "
                          f"(moyenne de {len(h_calib)} updates x "
                          f"{args.h_target_mult})", flush=True)

            policy_loss = policy_loss_last
            value_loss = value_loss_last
            entropy = entropy_last
            policy_gnorm = policy_gnorm_last if policy_gnorm_last is not None \
                else torch.tensor(0.0)
            critic_gnorm = critic_gnorm_last if critic_gnorm_last is not None \
                else torch.tensor(0.0)

            step += len(batch)
            n_updates += 1

            # --- Alpha (Eq. 3) avec CAP [FIX ALPHA] ---
            recent_entropies.append(entropy.item())
            if len(recent_entropies) > 50:
                recent_entropies.pop(0)
            if len(recent_entropies) >= 3 and h_target is not None:
                h_bar = np.mean(recent_entropies)
                # [FIX ALPHA] Cap : sans lui, alpha monte sans fin quand
                # H_target est inatteignable (policy discard competente
                # plafonne a H~1.7). Run 600k precedent : alpha=0.41 final,
                # bonus d'entropie ~7x la policy loss -> policy aplatie ->
                # 34 pourcent de 4es places contre 25 attendu.
                raw_update = alpha + args.beta * (h_target - h_bar)
                if args.alpha_max > 0:
                    alpha = max(0.0, min(raw_update, args.alpha_max))
                else:
                    alpha = max(0.0, raw_update)   # comportement papier

            # --- Publish ---
            if step % args.publish_every < len(batch):
                publish_to_state(
                    ps.state, ps.lock,
                    agent.policy.state_dict(),
                    agent.critic.state_dict(),
                    step=step, alpha=alpha, gamma_t=gamma_t,
                )

            # --- Log + Drift + GAME stats ---
            if step % args.log_every < len(batch):
                with rb_lock:
                    current_buf = len(rb_local)
                    last_ep = rb_local[-1] if rb_local else None
                # [MEMOIRE] taille REELLE du buffer (estimee sur la derniere
                # partie recue) et RAM systeme.
                buf_mb = 0.0
                if last_ep:
                    ep_bytes = sum(t['obs_normal'].nbytes
                                   + t['obs_oracle'].nbytes
                                   + t['mask'].nbytes for t in last_ep)
                    buf_mb = ep_bytes * current_buf / 1e6
                ram_pct = _mem_percent()
                mem_info = (f"({buf_mb:.0f}MB"
                            + (f",ram={ram_pct:.0f}%" if ram_pct is not None
                               else "") + ")")
                elapsed = time.time() - t_start
                metrics = compute_drift_metrics(batch)
                alerts = check_drift_alerts(metrics, phase)
                # [ORACLE MONITOR] norme des colonnes oracle de input_conv :
                # reste collee a la valeur d'init = la policy n'utilise pas
                # l'info cachee (cf. --oracle-init).
                wo_now = agent.oracle_weight_norm()
                wo_str = (f"{wo_now:.4f} "
                          f"(init {agent.oracle_init_norm:.4f}, "
                          f"{(wo_now / max(agent.oracle_init_norm, 1e-8) - 1.0) * 100:+.1f}%)")
                rdev_str = (f"{ratio_dev_first:.3f}"
                            if ratio_dev_first is not None else "n/a")
                kl_str = f"{kl_last:.4f}" if kl_last is not None else "off"
                pdng_str = (f"{pdanger_last:.4f}"
                            if pdanger_last is not None else "n/a")
                abl = getattr(agent, 'last_abl_kl', None)
                abl_str = f"{abl:.5f}" if abl is not None else "n/a"
                if critic_mode:
                    ev_str = (f"ev={compute_advantages.last_ev:+.3f}"
                              f"({'WARMUP' if warm else ('critic' if compute_advantages.last_used_critic else 'repli-mean')}) ")
                else:
                    ev_str = ""

                print(f"[step {step:7d} | phase {phase}] "
                      f"pl={policy_loss.item():+.4f} "
                      f"vl={value_loss.item():.4f} "
                      f"H={entropy.item():.4f} "
                      f"alpha={alpha:.4f} "
                      f"rdev={rdev_str} "
                      f"kl={kl_str} "
                      f"pdng={pdng_str} ablKL={abl_str} "
                      f"{ev_str}"
                      f"gamma={gamma_t:.3f} "
                      f"buf={current_buf}{mem_info} "
                      f"({elapsed:.0f}s)", flush=True)

                if metrics is not None:
                    print(
                        f"    | DRIFT: "
                        f"act_H={metrics['action_entropy']:.2f} "
                        f"act_div={metrics['action_div']:.2f} "
                        f"adv_abs={metrics['adv_abs_mean']:.2f} "
                        f"adv_std={metrics['adv_std']:.2f} "
                        f"val_std={metrics['val_std']:.2f} "
                        f"val_mean={metrics['val_mean']:+.2f} "
                        f"ret_mean={metrics['ret_mean']:+.2f} "
                        f"| top3=[{metrics['actions_top3']}] "
                        f"pgn={policy_gnorm.item():.2f} "
                        f"cgn={critic_gnorm.item():.2f} "
                        f"||W_o||={wo_str}",
                        flush=True,
                    )

                # [STATS] Ligne GAME
                game_line = format_game_stats(stats_proxy)
                if game_line is not None:
                    print(f"    | GAME: {game_line}", flush=True)

                for a in alerts:
                    print(f"    | !! {a}", flush=True)

                # [FAIL-FAST] Une tete SL figee en erreur (ou une requete
                # non-DISCARD inattendue) fausse toutes les parties : plus de
                # riichi/pon/chow corrects. S'entrainer dessus est du temps
                # perdu -> on s'arrete (sauf --allow-sl-errors).
                n_err = stats_proxy.get('n_sl_errors', 0)
                n_unh = stats_proxy.get('n_unhandled', 0)
                if (n_err or n_unh) and not args.allow_sl_errors:
                    raise RuntimeError(
                        f"tetes SL figees en erreur (n_sl_errors={n_err}, "
                        f"n_unhandled={n_unh}) : parties invalides. "
                        f"Voir le traceback [env] plus haut. "
                        f"--allow-sl-errors pour ignorer.")

            if step % args.save_every < len(batch):
                os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
                step_path = f"{base_out}_step{step:07d}.pt"
                latest_path = f"{base_out}_latest.pt"
                # [FIX RESUME] step passe au save -> le checkpoint contient
                # le step reel, plus besoin du regex sur le nom.
                agent.save(step_path, step=step, alpha=alpha)
                agent.save(latest_path, step=step, alpha=alpha)
                if args.keep_last_n > 0:
                    _purge_old_checkpoints(base_out, args.keep_last_n)
                # [JALONS] checkpoints JAMAIS purges, pour evaluer plus tard
                # (eval_checkpoints.py) et reprendre le meilleur : le dernier
                # n'est pas forcement le meilleur.
                if (args.milestone_every > 0
                        and step // args.milestone_every > last_milestone):
                    last_milestone = step // args.milestone_every
                    ms_path = f"{base_out}_milestone{step:09d}.pt"
                    agent.save(ms_path, step=step, alpha=alpha)
                    print(f"[step {step}] jalon {ms_path}", flush=True)
                print(f"[step {step}] save {step_path} + {latest_path}",
                      flush=True)

    except (BrokenPipeError, ConnectionError, ConnectionResetError,
            EOFError, IndexError, OSError, MemoryError) as e:
        # [FIX CRASH] Sauvegarde d'URGENCE avant de mourir. Le serveur
        # Manager sous Windows peut mourir sous la charge IPC : tous les
        # proxies deviennent invalides en cascade. Sans ce bloc, perte
        # totale du run si le crash arrive avant save_every.
        # [FIX MEMOIRE] MemoryError ajoute (il n'etait pas capture : aucun
        # checkpoint de crash n'avait ete ecrit). On libere d'abord le
        # buffer pour avoir la place d'ecrire.
        try:
            with rb_lock:
                rb_local.clear()
            gc.collect()
        except Exception:
            pass
        print(f"\n[train_rl] !!! CRASH a step={step} : "
              f"{type(e).__name__}: {e}", flush=True)
        print(f"[train_rl] Sauvegarde d'URGENCE des poids...", flush=True)
        try:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            emergency_path = f"{base_out}_crash_step{step:07d}.pt"
            agent.save(emergency_path, step=step, alpha=alpha)
            print(f"[train_rl] Poids sauvegardes : {emergency_path}",
                  flush=True)
            print(f"[train_rl] Pour reprendre :", flush=True)
            print(f"[train_rl]   --resume {emergency_path} "
                  f"(alpha={alpha:.4f} sauvegarde dans le checkpoint)",
                  flush=True)
        except Exception as e2:
            print(f"[train_rl] ECHEC sauvegarde d'urgence : {e2}", flush=True)
            print(f"[train_rl] Dernier checkpoint valide : cherchez "
                  f"{base_out}_step*.pt ou {base_out}_latest.pt", flush=True)
        raise

    # --- Fin normale ---
    agent.save(args.output, step=step, alpha=alpha)
    print(f"[train_rl] termine. Checkpoint : {args.output}", flush=True)

    # [NIVEAU 2] Fermeture propre de la Queue : close + join_thread
    # garantit que les transitions en transit sont consommees par le
    # drain thread avant l'extinction.
    try:
        transitions_queue.close()
        transitions_queue.join_thread()
    except Exception:
        pass

    for w in workers:
        w.terminate()
    for w in workers:
        w.join(timeout=2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--total-steps', type=int, default=1000000)
    p.add_argument('--n-workers', type=int, default=4)
    p.add_argument('--games-per-cycle', type=int, default=10)
    p.add_argument('--buffer-size', type=int, default=150,
                   help="capacite du buffer, en PARTIES (~15 Mo chacune en "
                        "float16 : 150 parties ~ 1.5-3 Go). Etait 500 "
                        "(7+ Go) -> MemoryError.")
    p.add_argument('--queue-size', type=int, default=None,
                   help="taille de la mp.Queue (defaut : max(4, 2*n_workers)). "
                        "Doit rester PETITE : chaque element est une partie.")
    p.add_argument('--mem-limit-pct', type=float, default=88.0,
                   help="seuil de RAM systeme (%%) au-dela duquel le buffer "
                        "est reduit (necessite psutil)")
    p.add_argument('--batch-episodes', type=int, default=20)
    p.add_argument('--max-batch-transitions', type=int, default=1000)

    p.add_argument('--lr', type=float, default=1e-4)
    # [FIX CRITIC BUDGET] 3e-3 rendait l'Adam du critic instable : il
    # trouvait un bon V(s) puis divergeait et le perdait (signature v8 :
    # EV +0.012 -> 0.000 en oscillation, val_std 0.13 -> 0.00).
    # Stable a 1e-3 (verifie sur cible synthetique).
    p.add_argument('--critic-lr', type=float, default=1e-3)
    # [FIX CRITIC BUDGET] pas de gradient SUPPLEMENTAIRES du critic par
    # update (minibatch aleatoire de 256). Sans inner loop, le compteur
    # de steps donne seulement 6 pas de gradient par 2000 steps (2
    # updates x 3 epochs) : ~800 pas en tout dans v8. 10 -> 13 pas par
    # update (26 par 2000 steps), ~3300 sur 250k steps. 0 desactive.
    p.add_argument('--critic-inner-steps', type=int, default=10)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--danger-penalty', type=float, default=0.0,
                   help="[DANGER] penalite sur la reward d'une decision qui "
                        "defausse une tuile qui ferait ron a un adversaire "
                        "tenpai (calcule depuis les mains cachees, sans "
                        "toucher au reseau). 0 = desactive. Conseille 0.5-1.0."
                        " Credit par-decision exact ; en phase 1 l'oracle "
                        "rend le danger lisible -> KPI dng%% dans les logs.")
    p.add_argument('--adv-mode', choices=['batch_mean', 'critic'],
                   default='batch_mean',
                   help="avantage : 'batch_mean' (A = R - moyenne du batch, "
                        "ancien) ou 'critic' (A = R - V(s), critic "
                        "asymetrique entraine a predire la reward de la "
                        "manche ; reduit fortement la variance)")
    p.add_argument('--critic-warmup-steps', type=int, default=300000,
                   help="[critic] steps pendant lesquels seul le critic "
                        "s'entraine (policy = SL) avant d'utiliser V(s)")
    p.add_argument('--reward-scale', type=float, default=10.0,
                   help="[critic] la cible du critic est R/reward_scale")
    p.add_argument('--min-ev', type=float, default=0.0,
                   help="[critic] si la variance expliquee du critic est "
                        "< min-ev sur un batch, repli sur batch-mean")
    p.add_argument('--max-episode-reuse', type=int, default=0,
                   help="nb max d'utilisations d'une partie du buffer "
                        "(0 = illimite, ancien comportement ; conseille 4)")
    p.add_argument('--h-target', type=str, default='2.0',
                   help="entropie cible (Eq. 3) : un nombre, ou 'auto' = "
                        "entropie mesuree de la policy au demarrage "
                        "(voir --h-calib-updates / --h-target-mult)")
    p.add_argument('--allow-sl-errors', action='store_true',
                   help="ne pas s'arreter si une tete SL figee echoue "
                        "(deconseille : parties invalides)")
    p.add_argument('--h-calib-updates', type=int, default=5,
                   help="nb d'updates moyennes pour --h-target auto")
    p.add_argument('--h-target-mult', type=float, default=1.0,
                   help="facteur applique a la cible auto (ex. 1.1)")
    p.add_argument('--alpha-init', type=float, default=0.01)
    p.add_argument('--beta', type=float, default=0.005)
    p.add_argument('--alpha-max', type=float, default=0.15,
                   help='[FIX ALPHA] cap sur le coefficient d entropie. '
                        'Sans lui, alpha monte sans fin quand H_target est '
                        'inatteignable (policy discard competente plafonne '
                        'a H~1.7) : le bonus d entropie finit par dominer '
                        'la policy loss. 0 = pas de cap (comportement '
                        'papier, deconseille a petite echelle).')

    p.add_argument('--danger-loss-coef', type=float, default=0.0,
                   help='[DANGER LOSS] poids de E_pi[danger] ajoute a la '
                        'loss (esperance exacte de defausser une tuile qui '
                        'fait ron, labels depuis les mains cachees). 0 = '
                        'desactive. Conseille 0.3-1.0 en phase 1.')
    p.add_argument('--ablate-every', type=int, default=10,
                   help='[DIAG] tous les N updates, mesure KL(pi || pi avec '
                        'oracle=0) -> "ablKL" dans les logs (0 = off).')
    p.add_argument('--oracle-frac', type=float, default=0.20)
    p.add_argument('--transition-frac', type=float, default=0.50)
    p.add_argument('--oracle-init', choices=['zero', 'kaiming'],
                   default='kaiming',
                   help='[FIX ORACLE TAKEOFF] init des canaux oracle absents '
                        'du checkpoint SL : zero (ancien, comportement '
                        'initial == SL mais l oracle n influence pas les '
                        'logits et doit croitre par pas PG bruites) ou '
                        'kaiming (contribution immediate, le PG la renforce '
                        'ou la corrige des le 1er update). Les canaux '
                        'rouges restent a zero dans les deux cas.')
    p.add_argument('--oracle-init-scale', type=float, default=0.25,
                   help='echelle de l init kaiming des canaux oracle '
                        '(0.25 : compromis mesure warm-start / vitesse de '
                        'prise en charge de l oracle)')
    p.add_argument('--grp-checkpoint', type=str,
                   default='checkpoints/grp_sl_east4.pt')

    p.add_argument('--iw-threshold', type=float, default=5.0)
    p.add_argument('--keep-last-n', type=int, default=0)
    p.add_argument('--kl-coef', type=float, default=0.05,
                   help='[ANCRE] poids de KL(pi || pi_SL) dans la loss. '
                        '0 = desactive. Plage utile 0.02-0.2 : trop bas -> '
                        'derive passive ; trop haut -> reste colle au SL. '
                        '[PHASE 1 GUARD] force a 0 si --oracle-frac >= 1.0 '
                        '(cf. apply_phase1_guards).')
    p.add_argument('--allow-phase1-kl', action='store_true',
                   help='[PHASE 1 GUARD] conserver --kl-coef tel quel meme '
                        'si --oracle-frac >= 1.0 (deconseille : ancre SL '
                        'aveugle a l oracle -> effondrement ||W_o||, cf. v10)')
    p.add_argument('--kl-ref', type=str, default=None,
                   help='[ANCRE] checkpoint dont les poids chargent la '
                        'reference KL (defaut : SL de depart, aveugle a '
                        'l oracle). Phase 2/3 : pointer le meilleur '
                        'checkpoint de phase 1 pour ancrer une policy '
                        'oracle-AWARE.')
    p.add_argument('--opponent-sl', action='store_true',
                   help='[ADVERSAIRES SL] 1 siege = policy RL (entrainee), '
                        '3 sieges = policy SL figee. Le siege RL est tire '
                        'au hasard a chaque partie (donc vent de depart '
                        'et ordre de jeu randomises). Seul le siege RL '
                        'produit des transitions. Defaut : self-play 4 '
                        'sieges RL.')
    p.add_argument('--opponent-mode', choices=['argmax', 'sample'],
                   default='argmax',
                   help='comportement des SL adversaires : argmax (comme '
                        "l'eval, defaut) ou sample (plus de diversite).")
    p.add_argument('--milestone-every', type=int, default=1000000,
                   help='un checkpoint *_milestoneN.pt jamais purge tous '
                        'les N steps (0 = desactive)')

    p.add_argument('--publish-every', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=50000)
    p.add_argument('--log-every', type=int, default=2000)

    p.add_argument('--resume', type=str, default=None,
                   help='chemin vers un checkpoint _latest.pt pour reprendre')
    p.add_argument('--ppo-epochs', type=int, default=3,
                   help='nombre d epochs PPO par batch (defaut 3)')

    p.add_argument('--alpha-resume', type=float, default=None,
                   help='alpha au resume ; prioritaire sur le checkpoint '
                        '(defaut: alpha du checkpoint, sinon --alpha-init). '
                        'OBLIGATOIRE pour ecraser l alpha du checkpoint '
                        'warmup (0.0022) : --alpha-resume 0.05.')
    p.add_argument('--reset-step', action='store_true',
                   help='[PHASES] reprend les POIDS du --resume mais repart '
                        'a step=0 : un milestone de fin de phase 1 porte un '
                        'step eleve qui fausserait le schedule gamma/lr de '
                        'la phase suivante.')

    args = p.parse_args()
    if args.danger_loss_coef > 0:
        # herite par les workers (spawn) : active les labels de danger
        os.environ['MJ_DANGER_LABELS'] = '1'
    train(args)


if __name__ == '__main__':
    mp.freeze_support()
    main()