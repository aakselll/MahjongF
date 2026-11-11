"""
Evaluation stable rank Tenhou (Suphx Appendix C), version parallele.

    stable_rank = (5*n1 + 2*n2) / n4 - 2

Utilise GameEnv (4 rounds East) et N workers pour parallelliser.

Usage :
py -3.14 -m mahjong_ai.training.eval_stable_rank `
--agent-checkpoint checkpoints/rl_tonpusen_v2/rl_tonpusen_v2_latest.pt `
--sl-discard checkpoints/sl_tonpusen_light/discard.pt `
--sl-frozen checkpoints/sl_tonpusen_light `
--weak-dir checkpoints/sl_tonpusen_light `
--n-games 200 --n-workers 5




"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from mahjong_ai.envs.suphx_features import N_CHANNELS
from mahjong_ai.envs.multi_agent import DISCARD


# ====================================================================== #
# Chargement des modeles
# ====================================================================== #
def load_sl_weak(weak_dir, device='cpu'):
    from mahjong_ai.agents.suphx_cnn import build_model
    heads = ['discard', 'riichi', 'chow', 'pong', 'kong']
    models = {}
    for h in heads:
        path = os.path.join(weak_dir, f'{h}.pt')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SL-weak manquant : {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        # [FIX] Architecture lue depuis la config du checkpoint
        # (les tetes light sont en 128 canaux / 6 blocs -- les defauts
        # de build_model sont 256/16 et provoquent un size mismatch).
        cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
        m = build_model(
            h,
            channels=cfg.get('channels', 256),
            n_blocks=cfg.get('n_blocks', 16),
            binary_hidden=cfg.get('binary_hidden', 256),
            in_channels=N_CHANNELS,   # [ROUGES] ancien ckpt adapte au chargement
        ).to(device)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        models[h] = m
    return models


class SLWeakAgent:
    def __init__(self, weak_dir, device='cpu', sample=False):
        self.models = load_sl_weak(weak_dir, device)
        self.device = device
        # [DIAG] True : les adversaires SL echantillonnent (comme a
        # l'entrainement --opponent-mode sample) au lieu de l'argmax.
        self.sample = sample
        self.frozen = {h: self.models[h] for h in
                        ['riichi', 'chow', 'pong', 'kong']}

    def act(self, obs_normal, obs_oracle=None, mask=None):
        """[FIX] Signature uniformisee avec RLAgent.act(obs, oracle, mask).
        Le SL-weak ignore l'oracle -- il joue sur l'obs normale seule."""
        on = torch.as_tensor(obs_normal, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.models['discard'](on)
            masked = logits + (mask_t - 1.0) * 1e9
            if self.sample:
                probs = torch.softmax(masked, dim=-1)
                return int(torch.multinomial(probs, 1).item())
        return int(masked.argmax(dim=-1).item())

class RLAgent:
    def __init__(self, checkpoint, sl_discard, sl_frozen, device='cpu',
                 sample=False, oracle_gamma=0.0):
        self.sample = sample
        # [DIAG ORACLE] 0.0 = deploiement (oracle coupe) ; 1.0 = oracle
        # complet visible. Comparer les deux rangs sur le meme checkpoint
        # dit si la policy utilise reellement l'info cachee.
        self.oracle_gamma = oracle_gamma
        from mahjong_ai.agents.oracle_agent import OracleDiscardAgent
        self.agent = OracleDiscardAgent(sl_discard, sl_frozen, device=device)
        ck = torch.load(checkpoint, map_location=device, weights_only=False)

        if 'policy' in ck:
            try:
                self.agent.policy.load_state_dict(ck['policy'])
            except RuntimeError:
                self.agent._load_partial_sl(ck['policy'])
            if 'critic' in ck:
                self.agent.critic.load_state_dict(ck['critic'])
        elif 'model_state_dict' in ck:
            self.agent._load_partial_sl(ck['model_state_dict'])
        else:
            raise ValueError(f"Format inconnu : {list(ck.keys())}")

        self.agent.policy.eval()
        self.device = device
        self.frozen = self.agent.frozen_sl

    def act(self, obs_normal, obs_oracle, mask):
        on = torch.as_tensor(obs_normal, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        oo = torch.as_tensor(obs_oracle, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = self.agent.forward_with_dropout(
                on, oo, gamma_t=self.oracle_gamma)
            masked = logits + (mask_t - 1.0) * 1e9
            if self.sample:
                # [EVAL] politique echantillonnee, comme a l'entrainement
                probs = torch.softmax(masked, dim=-1)
                return int(torch.multinomial(probs, 1).item())
        return int(masked.argmax(dim=-1).item())


# ====================================================================== #
# Worker : joue N parties et retourne les compteurs
# ====================================================================== #
def eval_worker(worker_id, n_games, agent_checkpoint, sl_discard,
                 sl_frozen, weak_dir, device='cpu', seed=0, sample=False,
                 worker_offset=0, oracle_gamma=0.0, weak_sample=False):
    """Chaque worker : charge ses propres modeles (thread-safe car process).

    Retourne (n1, n2, n3, n4) sur n_games parties.
    [DEBUG] Log a CHAQUE partie pour verifier la progression.
    """
    n_workers_total = int(os.environ.get("EVAL_N_WORKERS", "1"))
    threads = max(1, torch.get_num_threads() // max(n_workers_total, 1))
    torch.set_num_threads(threads)

    from mahjong_ai.training.game_env import GameEnv, PredictorHolder

    # Chargement (log pour chaque worker)
    t_load = time.time()
    torch.manual_seed(seed + worker_id)
    rl_agent = RLAgent(agent_checkpoint, sl_discard, sl_frozen, device,
                       sample=sample, oracle_gamma=oracle_gamma)
    weak = SLWeakAgent(weak_dir, device, sample=weak_sample)
    print(f"[worker {worker_id}] modeles charges en "
          f"{time.time()-t_load:.1f}s (threads={threads})", flush=True)

    predictor_holder = PredictorHolder()
    env = GameEnv(seed=seed + worker_id, predictor_holder=predictor_holder,
                  game_log_buffer=None, n_rounds=4)
    env.round_env.frozen_sl = weak.frozen

    print(f"[worker {worker_id}] debut des {n_games} parties", flush=True)

    n1 = n2 = n3 = n4 = 0
    t0 = time.time()
    for i in range(n_games):
        # [EVAL] rotation des sieges : le siege 0 est dealer a la 1re manche,
        # le fixer biaisait la mesure. Chaque siege est evalue autant.
        eval_seat = (worker_offset + i) % 4
        rank = _play_one_game(rl_agent, weak, env, eval_seat=eval_seat)
        if rank == 1: n1 += 1
        elif rank == 2: n2 += 1
        elif rank == 3: n3 += 1
        else: n4 += 1

        # [DEBUG] Log a chaque partie
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (n_games - i - 1) / max(rate, 1e-6)
        print(f"[worker {worker_id}] partie {i+1:3d}/{n_games} "
              f"rank={rank} | cum: 1st={n1} 2nd={n2} 3rd={n3} 4th={n4} "
              f"({rate:.2f} g/s, ETA {eta:.0f}s)", flush=True)

    return n1, n2, n3, n4


def _play_one_game(eval_agent, weak_agent, env, eval_seat=0):
    """Joue une partie 4-round East. Retourne le rang final de eval_seat."""
    req = env.reset_round()
    while True:
        seat = req['seat']
        action = None

        if req['action_type'] == DISCARD:
            if seat == eval_seat:
                action = eval_agent.act(req['obs'], req['oracle'], req['mask'])
            else:
                action = weak_agent.act(req['obs'], req['oracle'], req['mask'])

        result = env.step(action)

        if result['round_done']:
            if result.get('game_done', False):
                break
            req = env.reset_round()
        else:
            req = req if False else result['next_request']

    scores = env.game_scores
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    return order.index(eval_seat) + 1


def compute_stable_rank(n1, n2, n4):
    if n4 == 0:
        return float('inf')
    return (5 * n1 + 2 * n2) / n4 - 2


def mean_rank_ci(n1, n2, n3, n4):
    """Rang moyen +- demi-intervalle de confiance a 95 % (loi normale)."""
    import numpy as np
    ranks = np.repeat([1, 2, 3, 4], [n1, n2, n3, n4])
    m = float(ranks.mean())
    return m, 1.96 * float(ranks.std(ddof=1)) / (len(ranks) ** 0.5)


def stable_rank_ci(n1, n2, n3, n4, n_boot=20000, seed=0):
    """Bootstrap multinomial 95 % : le stable rank est un ratio tres bruite."""
    import numpy as np
    rng = np.random.default_rng(seed)
    tot = n1 + n2 + n3 + n4
    draws = rng.multinomial(tot, np.array([n1, n2, n3, n4]) / tot,
                            size=n_boot)
    vals = np.array([compute_stable_rank(d[0], d[1], d[3]) for d in draws])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ====================================================================== #
# Main
# ====================================================================== #
def evaluate(args):
    device = 'cpu'

    # Split des parties entre workers
    n_workers = max(1, args.n_workers)
    games_per_worker = args.n_games // n_workers
    remainder = args.n_games % n_workers
    splits = [games_per_worker + (1 if i < remainder else 0)
              for i in range(n_workers)]

    print(f"[eval] {args.n_games} parties sur {n_workers} workers "
          f"({splits} chacun)", flush=True)

    os.environ["EVAL_N_WORKERS"] = str(n_workers)
    #torch.set_num_threads(max(1, 12 // n_workers))

    # Lance les workers
    tasks = []
    for i in range(n_workers):
        if splits[i] == 0:
            continue
        tasks.append((i, splits[i], args.agent_checkpoint, args.sl_discard,
                       args.sl_frozen, args.weak_dir, device, args.seed,
                       args.sample, sum(splits[:i]), args.oracle_gamma,
                       args.weak_sample))

    print(f"[eval] lancement de {len(tasks)} workers...", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(eval_worker, tasks)

    elapsed = time.time() - t0

    # Agrege
    n1 = sum(r[0] for r in results)
    n2 = sum(r[1] for r in results)
    n3 = sum(r[2] for r in results)
    n4 = sum(r[3] for r in results)
    total = n1 + n2 + n3 + n4

    sr = compute_stable_rank(n1, n2, n4)
    mr, mr_ci = mean_rank_ci(n1, n2, n3, n4)
    sr_lo, sr_hi = stable_rank_ci(n1, n2, n3, n4)

    print(f"\n=== RESULTAT FINAL ===")
    print(f"  Games     : {total}")
    print(f"  1st/2nd/3rd/4th : {n1}/{n2}/{n3}/{n4}")
    print(f"  1st rate  : {n1/total:.4f}")
    print(f"  2nd rate  : {n2/total:.4f}")
    print(f"  3rd rate  : {n3/total:.4f}")
    print(f"  4th rate  : {n4/total:.4f}")
    print(f"  Rang moyen           : {mr:.3f} +- {mr_ci:.3f} (95 %)  "
          f"[2.500 = niveau egal]")
    print(f"  Stable rank (Tenhou) : {sr:.4f}  IC95 [{sr_lo:.2f}, {sr_hi:.2f}]")
    print(f"  Oracle (gamma)       : {args.oracle_gamma} "
          f"(0 = deploiement, 1 = oracle complet)")
    print(f"  Adversaires SL       : "
          f"{'echantillonnes' if args.weak_sample else 'argmax'}")
    print(f"  Mode joueur eval     : "
          f"{'echantillonne' if args.sample else 'argmax'}, sieges alternes")
    if total < 1000:
        print(f"  !! {total} parties : trop peu pour conclure "
              f"(viser >= 1000, IC ci-dessus).")
    print(f"  Duree     : {elapsed:.1f}s ({total/elapsed:.2f} games/s)")
    return {'n': (n1, n2, n3, n4), 'mean_rank': mr, 'mean_rank_ci': mr_ci,
            'stable_rank': sr, 'sr_ci': (sr_lo, sr_hi)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent-checkpoint', required=True)
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--weak-dir', required=True)
    p.add_argument('--n-games', type=int, default=500)
    p.add_argument('--n-workers', type=int, default=4,
                   help='nombre de workers paralleles (defaut 4)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample', action='store_true',
                   help='le RL echantillonne sa politique (comme a '
                        "l'entrainement) au lieu de l'argmax")
    p.add_argument('--oracle-gamma', type=float, default=0.0,
                   help='[DIAG] proportion d oracle visible pour le RL : '
                        '0 = deploiement (defaut), 1 = oracle complet')
    p.add_argument('--weak-sample', action='store_true',
                   help='[DIAG] les 3 adversaires SL echantillonnent '
                        "(au lieu de l'argmax)")
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    mp.freeze_support()
    main()