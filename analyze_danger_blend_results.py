"""[PLAN DANGER - ÉTAPE 3, post-sweep] Analyse appariée de results.json.

Le sweep (eval_danger_blend.py) a échoué sa porte pré-enregistrée. Ce script
exploite ce que la console ne montre pas : les stats PAR PARTIE appariées
(mêmes donnes dans tous les bras, clé 'c.e' = chunk.env). Il répond aux
questions pré-enregistrées avant re-discussion du plan :

  1. Où part le coût mr ? (Δrank moyen selon que le bras a évité / inchangé /
     ajouté des deals-in dans la même donne)
  2. McNemar des deals-in : combien de donnes protégées (baseline paye, bras
     non) vs nouvelles pertes (baseline saine, bras paye) -> faux swaps nets
  3. Rendement exact : deals-in évités pour 1000 swaps, et rendement MARGINAL
     lambda=4 -> 16
  4. Anomalie (1,16) : DeltaMr +0.000 avec 13.6% de swaps -- distribution
     appariée vs (2,16)/(4,16)
  5. Intégrité : les 'paired' stockés (dmr, rel_dng) recalculés from scratch

Usage (racine du repo, aucun GPU requis) :
    py -3.14 analyze_danger_blend_results.py
    py -3.14 analyze_danger_blend_results.py --in checkpoints\\danger_blend_sweep\\results.json --csv sweep_analyse.csv
"""
import argparse
import csv
import json
import os

import numpy as np

DEFAULT_IN = os.path.join('checkpoints', 'danger_blend_sweep', 'results.json')


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def per_game_arrays(arm):
    """-> (keys triées, rank, rd, dd) alignés sur les mêmes donnes."""
    pg = arm['per_game']
    keys = sorted(pg)                      # ordre identique pour tous les bras
    r = np.array([pg[k]['rank'] for k in keys], float)
    rd = np.array([pg[k]['rd'] for k in keys], float)
    dd = np.array([pg[k]['dd'] for k in keys], float)
    return keys, r, rd, dd


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float('nan')


def main():
    ap = argparse.ArgumentParser(
        description='Analyse appariée post-sweep du blend danger')
    ap.add_argument('--in', dest='inp', default=DEFAULT_IN)
    ap.add_argument('--csv', default=None,
                    help='export optionnel des stats par partie et par bras')
    args = ap.parse_args()

    res = load(args.inp)
    arms = res['arms']
    if 'baseline' not in arms:
        raise SystemExit("pas de bras 'baseline' dans results.json")
    bk, br, brd, bdd = per_game_arrays(arms['baseline'])
    n = len(bk)
    base_dng = bdd.sum() / max(brd.sum(), 1.0)
    print(f"[analyse] {n} donnes appariées | baseline : mr={br.mean():.3f} "
          f"wr={float((br == 1).mean()):.3f} dng={100 * base_dng:.2f}% "
          f"({int(bdd.sum())}/{int(brd.sum())})")
    print(f"[analyse] donnes baseline avec >=1 deal-in : "
          f"{int((bdd >= 1).sum())}/{n} | deals moyen/donne "
          f"{bdd.mean():.3f}\n")

    labels = [k for k in arms if k != 'baseline']
    rows_csv = []

    # ---------------- tableau récap + rendements ------------------------- #
    print("=" * 100)
    print(f"{'bras':12s} {'swaps%':>6s} {'dMr':>6s} {'red%':>6s} "
          f"{'evit.':>6s} {'/1000sw':>7s} {'prot.':>5s} {'pertes':>6s} "
          f"{'net':>5s}")
    print("-" * 100)
    summary = {}
    for lbl in labels:
        a = arms[lbl]
        _, r, rd, dd = per_game_arrays(a)
        drank = r - br
        ddd = dd - bdd
        sw = a['n_blend_swaps']
        dec = a['n_blend_decisions']
        avoided = float(bdd.sum() - dd.sum())          # deals évités (apparié)
        y1000 = 1000.0 * avoided / max(sw, 1)
        prot = int(((bdd >= 1) & (dd == 0)).sum())     # donnes protégées
        pertes = int(((bdd == 0) & (dd >= 1)).sum())   # nouvelles pertes
        red = ((base_dng - dd.sum() / max(rd.sum(), 1)) / base_dng
               if base_dng > 0 else float('nan'))
        summary[lbl] = dict(drank=drank, ddd=ddd, avoided=avoided,
                            y1000=y1000, prot=prot, pertes=pertes, red=red,
                            dpts=100 * (dd.sum() / max(rd.sum(), 1)
                                        - base_dng))
        red_s = f"{100 * red:5.1f}%" if red == red else '  n/a'
        print(f"{lbl:12s} {100 * sw / max(dec, 1):6.2f} "
              f"{drank.mean():+6.3f} {red_s} {avoided:+6.1f} "
              f"{y1000:7.1f} {prot:5d} {pertes:6d} "
              f"{prot - pertes:+5d}")
    print("=" * 100)
    print("evit. = deals-in évités vs baseline (donnes appariées) ; "
          "prot. = donnes où la baseline payait et le bras non ;\n"
          "pertes = donnes où la baseline était saine et le bras paye "
          "(faux swaps + dérive de trajectoire) ; net = prot. - pertes.\n")

    # ---------------- rendement marginal lambda=4 -> 16 ------------------ #
    print("-- rendement MARGINAL des swaps (lambda=4 -> 16, même tau) --")
    for tau in (1, 2, 4):
        a4, a16 = summary.get(f'tau={tau},lam=4'), summary.get(f'tau={tau},lam=16')
        if not (a4 and a16):
            continue
        d_sw = (arms[f'tau={tau},lam=16']['n_blend_swaps']
                - arms[f'tau={tau},lam=4']['n_blend_swaps'])
        d_ev = a16['avoided'] - a4['avoided']
        d_dmr = a16['drank'].mean() - a4['drank'].mean()
        print(f"  tau={tau} : +{d_sw} swaps -> {d_ev:+.0f} deal(s)-in évité(s) "
              f"en plus, DeltaMr {d_dmr:+.3f}")
    print()

    # ---------------- où part le coût mr ? ------------------------------- #
    print("-- DeltaRank moyen selon les deals-in de la donne (apparié) --")
    print(f"{'bras':12s} {'évités (ddd<0)':>16s} {'inchangés (=0)':>15s} "
          f"{'ajoutés (ddd>0)':>16s}")
    for lbl in labels:
        s = summary[lbl]
        m = lambda mask: (f"n={mask.sum():3d} dr={s['drank'][mask].mean():+.3f}"
                          if mask.any() else "n=  0")
        print(f"{lbl:12s} {m(s['ddd'] < 0):>16s} {m(s['ddd'] == 0):>15s} "
              f"{m(s['ddd'] > 0):>16s}")
    print("\nLecture : si le coût mr vient des donnes 'inchangés' (swaps sans "
          "effet sur les deals), c'est la valeur de main détruite par les\n"
          "faux swaps ; s'il vient des donnes 'évités', même éviter un deal "
          "ne rembourse pas le repli.\n")

    # ---------------- distribution appariée du rang ---------------------- #
    print("-- DeltaRank apparié par bras (négatif = mieux) --")
    print(f"{'bras':12s} {'moy':>6s} {'q05':>6s} {'q25':>6s} {'med':>6s} "
          f"{'q75':>6s} {'q95':>6s} {'%pire':>6s} {'%mieux':>7s}")
    for lbl in labels:
        d = summary[lbl]['drank']
        print(f"{lbl:12s} {d.mean():+6.3f} {q(d, 5):+6.1f} {q(d, 25):+6.1f} "
              f"{q(d, 50):+6.1f} {q(d, 75):+6.1f} {q(d, 95):+6.1f} "
              f"{100 * float((d > 0).mean()):5.1f}% "
              f"{100 * float((d < 0).mean()):6.1f}%")
    print()

    # ---------------- anomalie (1,16) ------------------------------------ #
    if 'tau=1,lam=16' in summary:
        print("-- anomalie (1,16) : DeltaMr +0.000 avec 13.6% de swaps --")
        for lbl in ('tau=1,lam=16', 'tau=2,lam=16', 'tau=4,lam=16'):
            if lbl not in summary:
                continue
            d = summary[lbl]['drank']
            print(f"  {lbl:12s} : {int((d < 0).sum())} donnes meilleures / "
                  f"{int((d == 0).sum())} égales / {int((d > 0).sum())} pires "
                  f"| pire donne {d.max():+.0f} rang(s), "
                  f"meilleure {d.min():+.0f}")
        print()

    # ---------------- intégrité vs 'paired' stockés ---------------------- #
    print("-- intégrité : recalcul vs 'paired' stocké --")
    for lbl in labels:
        pc = arms[lbl].get('paired')
        if not pc:
            continue
        _, r, rd, dd = per_game_arrays(arms[lbl])
        dmr = float((r - br).mean())
        dpts = 100 * (dd.sum() / max(rd.sum(), 1) - base_dng)
        ok = (abs(dmr - pc['dmr']) < 5e-4
              and abs(dpts - pc.get('d_dng_pts', dpts)) < 5e-3)
        red_s = (f"red {100 * ((base_dng - dd.sum() / max(rd.sum(), 1))
                              / base_dng):.2f}% "
                 f"(stocké {100 * pc['rel_dng']:.2f}%)"
                 if base_dng > 0 else 'red n/a (baseline sans deal)')
        print(f"  {lbl:12s} : dMr {dmr:+.4f} (stocké {pc['dmr']:+.4f}) | "
              f"{red_s} -> {'OK' if ok else 'ÉCART !'}")
    print()

    # ---------------- export csv optionnel ------------------------------- #
    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['bras', 'donne', 'rank_base', 'rank_bras',
                        'd_rank', 'dd_base', 'dd_bras', 'd_dd'])
            for lbl in labels:
                _, r, rd, dd = per_game_arrays(arms[lbl])
                for i, k in enumerate(bk):
                    w.writerow([lbl, k, int(br[i]), int(r[i]),
                                int(r[i] - br[i]), int(bdd[i]), int(dd[i]),
                                int(dd[i] - bdd[i])])
        print(f"[analyse] export : {args.csv}")


if __name__ == '__main__':
    main()
