"""Évaluation head-to-head autonome : une policy contre 3 SL argmax.

Réplique exactement l'éval inline de rl/train.py (mode déploiement :
gamma=0, aucune transition collectée) mais :
  - n'importe quand, sur n'importe quel checkpoint (RL ou SL) ;
  - avec un n grand et un IC95 affiché ;
  - avec un design APPARIÉ : chaque chunk de C parties = C envs neufs
    (graine déterministe par slot), UNE partie par slot. Deux évals
    lancées avec la même --seed voient les mêmes donnes et les mêmes
    sièges agent -> la différence entre deux runs = pur effet policy.

[V2.4] Deux ajouts pour le plan de diagnostic post-rl_defense2 :
  --sampled   le siège agent ÉCHANTILLONNE (comme les stats GAME du
              trainer / comme le déploiement stochastique) au lieu de
              l'argmax. Les 3 adversaires SL restent argmax (identique
              au champ de collecte du trainer, --opponent-mode argmax).
              À ne pas oublier : jusqu'ici on ne connaissait que
              « SL argmax = 2.50 » ; le niveau du SL ÉCHANTILLONNÉ dans
              ce champ n'a jamais été mesuré — c'est le trou de
              référence que ce flag comble.
  --grp PATH  charge le GRP dans le runner (même champ de récompenses
              que le training ; les dynamiques de jeu sont identiques
              avec ou sans, mais on ne veut pas dépendre de cet
              argument théorique).

Usage (racine du repo) :
  # baseline argmax : la policy reste à l'init SL (référence 2.50)
  py -3.14 -m mahjong_ai.rl.eval_matchup ^
      --policy checkpoints/sl_tonpusen_la/discard.pt ^
      --sl-discard checkpoints/sl_tonpusen_la/discard.pt ^
      --sl-frozen checkpoints/sl_tonpusen_la ^
      --grp checkpoints/grp_tonpusen.pt ^
      --games 384 --chunk-size 32 --seed 777 --compute-lookahead

  # le même checkpoint en échantillonné
  ... --sampled

Conseil : gardez la MÊME --seed, le même --games et le même --chunk-size
pour toutes les évals comparées (condition de l'appariement).
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


def evaluate_chunk(agent, chunk, seed, device, compute_lookahead,
                   sampled, grp_path):
    """Un chunk = `chunk` envs neufs, une partie chacun (appariement exact)."""
    runner = VecRunner(agent, n_envs=chunk, seed=seed, grp_path=grp_path,
                       compute_lookahead=compute_lookahead,
                       danger_labels=False, mode='opponent', device=device)
    _, stats = runner.collect_batch(
        chunk, gamma_t=0.0, deterministic=not sampled, collect=False,
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
    p.add_argument('--sampled', action='store_true',
                   help='[V2.4] le siège agent échantillonne au lieu de '
                        'l\'argmax (comme les stats GAME du trainer). '
                        'Les 3 adversaires SL restent argmax.')
    p.add_argument('--grp', default=None,
                   help='[V2.4] checkpoint GRP à charger dans le runner '
                        '(même champ de récompenses que le training).')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    # échantillonnage reproductible d'un run à l'autre (l'argmax n'en
    # dépend pas, mais --sampled si)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    mode_str = 'échantillonné' if args.sampled else 'argmax'
    grp_str = f'GRP={args.grp}' if args.grp else 'pas de GRP'
    print(f"[eval] siège agent : {mode_str} | adversaires : argmax | "
          f"{grp_str} | "
          f"lookahead={'ON' if args.compute_lookahead else 'OFF'}",
          flush=True)

    chunk = min(args.chunk_size, args.games)
    n_chunks = (args.games + chunk - 1) // chunk
    ranks, n_rounds, ends = [], [], {'tsumo': 0, 'ron': 0,
                                     'exhaustive': 0, 'abortive': 0}
    t0 = __import__('time').time()
    for c in range(n_chunks):
        n_here = min(chunk, args.games - c * chunk)
        stats = evaluate_chunk(agent, n_here, args.seed + c * chunk,
                               device, args.compute_lookahead,
                               args.sampled, args.grp)
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
    print(f"[eval] siège agent {mode_str} ; baseline SL argmax attendue "
          f"~2.50 ; même --seed entre deux évals = comparaison appariée",
          flush=True)


if __name__ == '__main__':
    main()
