"""Évaluation head-to-head autonome : une policy contre 3 SL argmax.

Réplique exactement l'éval inline de rl/train.py (mode déploiement :
argmax partout, gamma=0, aucune transition collectée) mais :
  - n'importe quand, sur n'importe quel checkpoint (RL ou SL) ;
  - avec un n grand et un IC95 affiché ;
  - avec un design APPARIÉ : chaque chunk de C parties = C envs neufs
    (graine déterministe par slot), UNE partie par slot. Deux évals
    lancées avec la même --seed voient les mêmes donnes et les mêmes
    sièges agent -> la différence entre deux runs = pur effet policy.

Usage (racine du repo) :
  # baseline : la policy reste à l'init SL
  py -3.14 -m mahjong_ai.rl.eval_matchup ^
      --policy checkpoints/sl_tonpusen_la/discard.pt ^
      --sl-discard checkpoints/sl_tonpusen_la/discard.pt ^
      --sl-frozen checkpoints/sl_tonpusen_la ^
      --games 512 --seed 777 --compute-lookahead

  # checkpoint RL (clé 'policy' détectée automatiquement)
  py -3.14 -m mahjong_ai.rl.eval_matchup ^
      --policy checkpoints/rl_nooracle_v1/rl_discard_latest.pt ^
      --sl-discard checkpoints/sl_tonpusen_la/discard.pt ^
      --sl-frozen checkpoints/sl_tonpusen_la ^
      --games 512 --seed 777 --compute-lookahead

Conseil : gardez la MÊME --seed (et le même --games/--chunk-size) pour
toutes les évals comparées ; comparez les mean_rank entre eux.
"""
import argparse

import numpy as np
import torch

from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner


def is_rl_checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'policy' in ck:
        return True, ck
    return False, ck


def evaluate_chunk(agent, chunk, seed, device, compute_lookahead):
    """Un chunk = `chunk` envs neufs, une partie chacun (appariement exact)."""
    runner = VecRunner(agent, n_envs=chunk, seed=seed, grp_path=None,
                       compute_lookahead=compute_lookahead,
                       danger_labels=False, mode='opponent', device=device)
    _, stats = runner.collect_batch(
        chunk, gamma_t=0.0, deterministic=True, collect=False,
        gamma_override=0.0)
    return stats


def main():
    p = argparse.ArgumentParser(
        description='Éval head-to-head : policy vs 3 SL argmax (gamma=0)')
    p.add_argument('--policy', required=True,
                   help='checkpoint RL (clé policy) OU discard.pt SL '
                        '(= baseline : policy à l\'init SL)')
    p.add_argument('--sl-discard', required=True,
                   help='discard.pt SL (arch + poids init de la policy)')
    p.add_argument('--sl-frozen', required=True,
                   help='dossier des têtes SL figées (riichi/chow/pong/kong)')
    p.add_argument('--games', type=int, default=512)
    p.add_argument('--chunk-size', type=int, default=64,
                   help='parties jouées en parallèle ; garder IDENTIQUE '
                        'entre deux évals comparées (condition de '
                        'l\'appariement)')
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--compute-lookahead', action='store_true',
                   help='à activer si la policy a été entraînée/initialisée '
                        'avec lookahead (mêmes conditions que le training)')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)

    agent = PPOAgent(sl_discard_path=args.sl_discard,
                     sl_frozen_dir=args.sl_frozen, device=device)

    is_rl, ck = is_rl_checkpoint(args.policy)
    if is_rl:
        agent.load(args.policy, load_optimizers=False)
        print(f"[eval] policy RL : {args.policy} "
              f"(step={ck.get('step', '?')}, phase={ck.get('phase', '?')})",
              flush=True)
    else:
        print(f"[eval] BASELINE : {args.policy} = init SL "
              f"(val_acc={ck.get('val_acc', '?')})", flush=True)

    chunk = min(args.chunk_size, args.games)
    n_chunks = (args.games + chunk - 1) // chunk
    ranks, n_rounds, ends = [], [], {'tsumo': 0, 'ron': 0,
                                     'exhaustive': 0, 'abortive': 0}
    t0 = __import__('time').time()
    for c in range(n_chunks):
        n_here = min(chunk, args.games - c * chunk)
        stats = evaluate_chunk(agent, n_here, args.seed + c * chunk,
                               device, args.compute_lookahead)
        for s in stats:
            ranks.append(s['rank_of_seat0'])
            n_rounds.append(s['n_rounds'])
            for k in ends:
                ends[k] += s.get(f'n_{k}', 0)
        done = (c + 1) * chunk
        print(f"[eval] {min(done, args.games)}/{args.games} parties "
              f"({__import__('time').time()-t0:.0f}s)", flush=True)

    r = np.array(ranks, dtype=np.float64)
    n = len(r)
    mean = float(r.mean())
    se = float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    hist = [int((r == k).sum()) for k in (1, 2, 3, 4)]
    tot_ends = max(sum(ends.values()), 1)
    print()
    print(f"[eval] RESULTAT n={n}  mean_rank={mean:.3f} "
          f"(IC95 ±{1.96*se:.3f})  wr={hist[0]/n:.3f}")
    print(f"[eval] rangs [1..4] : {hist[0]} / {hist[1]} / {hist[2]} / "
          f"{hist[3]}   rounds_moy={np.mean(n_rounds):.2f}")
    print(f"[eval] fins : tsumo%={100*ends['tsumo']/tot_ends:.0f} "
          f"ron%={100*ends['ron']/tot_ends:.0f} "
          f"exh%={100*ends['exhaustive']/tot_ends:.0f}")
    print(f"[eval] (baseline SL attendue ~2.50 ; même --seed entre deux "
          f"évals = comparaison appariée)", flush=True)


if __name__ == '__main__':
    main()
