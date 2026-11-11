"""
eval_checkpoints.py -- Evalue plusieurs checkpoints RL (jalons) contre le SL
et designe le meilleur (rang moyen le plus bas).

Usage :
    py -3.14 -m mahjong_ai.training.eval_checkpoints ^
        --checkpoints checkpoints\\rl_tonpusen_v2\\*_milestone*.pt ^
        --sl-discard <discard SL> --sl-frozen <dossier tetes SL> ^
        --weak-dir <dossier SL-weak> --n-games 1000 --n-workers 4

Chaque checkpoint = n-games parties (20+ min pour 1000). Ordre croissant de
step : on voit OU la courbe se degrade. Reprendre/exporter le meilleur,
pas forcement le dernier.
"""
import argparse
import glob
import multiprocessing as mp
import re
from types import SimpleNamespace

from mahjong_ai.training.eval_stable_rank import evaluate


def _step_of(path):
    m = re.search(r'(?:step|milestone)(\d+)', path)
    return int(m.group(1)) if m else -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='chemins ou motifs glob')
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--weak-dir', required=True)
    p.add_argument('--n-games', type=int, default=1000)
    p.add_argument('--n-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample', action='store_true')
    a = p.parse_args()

    paths = []
    for pat in a.checkpoints:
        paths += glob.glob(pat) or [pat]
    paths = sorted(set(paths), key=_step_of)
    if not paths:
        raise SystemExit("aucun checkpoint trouve")

    rows = []
    for path in paths:
        print(f"\n##### {path}", flush=True)
        r = evaluate(SimpleNamespace(
            agent_checkpoint=path, sl_discard=a.sl_discard,
            sl_frozen=a.sl_frozen, weak_dir=a.weak_dir,
            n_games=a.n_games, n_workers=a.n_workers, seed=a.seed,
            sample=a.sample))
        rows.append((path, r))

    print("\n=== RECAPITULATIF (rang moyen, bas = mieux ; 2.500 = egal) ===")
    for path, r in rows:
        print(f"  step {_step_of(path):>10d}  rang={r['mean_rank']:.3f} "
              f"+-{r['mean_rank_ci']:.3f}  stable={r['stable_rank']:.2f}  "
              f"{path}")
    best = min(rows, key=lambda x: x[1]['mean_rank'])
    print(f"\nMeilleur : {best[0]} (rang {best[1]['mean_rank']:.3f})")


if __name__ == '__main__':
    mp.freeze_support()
    main()
