"""[PLAN DANGER - ÉTAPE 3, post-sweep] Analyse appariée de results.json.

Fonctionne sur TOUS les formats de sweep (v1 384p, v2a/v2b 96p, v2c 288p
-- schéma identique ; les sections v1 sans objet sont ignorées). Le sweep
a échoué sa porte pré-enregistrée. Ce script exploite ce que la console
ne montre pas : les stats PAR PARTIE appariées (mêmes donnes dans tous
les bras, clé 'c.e' = chunk.env). Sections :

  1. Tableau récap apparié : swaps%, dMr, réduction dng%, deals évités,
     rendement /1000 swaps, donnes protégées vs nouvelles pertes
  2. [v2] Apparié par donne : McNemar EXACT (deals et victoires),
     delta_rd (exposition : les manches jouées changent-elles ?),
     corr(Δdd, Δrank) (les deals évités rapportent-ils du rang ?),
     top donnes |Δdd|
  3. [v2] Échelle λ par τ (générique) : rendement MARGINAL des swaps
     entre λ consécutifs -- c'est lui qui a montré la destruction
     marginale λ16 -> λ32 en v2b
  4. Où part le coût mr ? (Δrank selon deals évités / inchangés /
     ajoutés dans la même donne)
  5. Distribution appariée du rang
  6. Anomalie v1 (1,16) si présente
  7. Intégrité : recalcul vs 'paired' stockés

Usage (racine du repo, aucun GPU requis) :
    py -3.14 analyze_danger_blend_results.py --in checkpoints\\danger_blend_sweep_v2c\\results.json
    py -3.14 analyze_danger_blend_results.py --in ... --csv sweep_analyse.csv
"""
import argparse
import csv
import json
import math
import os
import re

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


def per_game_paid(arm):
    """(dpid, pts, ckan, ckpts) par donne si le sweep les a enregistrés
    (v2d/v3, patch fix_dealins_paid_metric), sinon None. dpid = deal-ins
    RÉELLEMENT payés (ron sur défausse dangereuse), pts = points payés
    (négatifs, unités reward) ; ckan/ckpts = chankans payés (ron adverse
    sur le shouminkan de l'agent)."""
    pg = arm['per_game']
    if not pg or 'dpid' not in next(iter(pg.values())):
        return None
    keys = sorted(pg)                      # même tri que per_game_arrays
    return (np.array([pg[k]['dpid'] for k in keys], float),
            np.array([pg[k].get('pts', 0.0) for k in keys], float),
            np.array([pg[k].get('ckan', 0) for k in keys], float),
            np.array([pg[k].get('ckpts', 0.0) for k in keys], float))


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float('nan')


def mcnemar_exact(b, c):
    """p exact bilatéral du test de McNemar (binomial sur b vs c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


ARM_RE = re.compile(r'^tau=([\d.]+),lam=([\d.]+)$')


def arm_params(lbl):
    m = ARM_RE.match(lbl)
    return (float(m.group(1)), float(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser(
        description='Analyse appariée post-sweep du blend danger (v1/v2)')
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

    # ---------------- 1. tableau récap + rendements ---------------------- #
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

    # ---------------- 1bis. deals PAYÉS & économie (v2d+) ------------- #
    paid_b = per_game_paid(arms['baseline'])
    if paid_b is not None:
        bpid, bpts, bckan, bckpts = paid_b
        has_ck = bool(bckan.sum() > 0 or any(
            (per_game_paid(arms[l]) or (None, None, np.zeros(1),
                                       np.zeros(1)))[2].sum() > 0
            for l in labels))
        print("-- deals PAYÉS (ron sur défausse dangereuse) & économie "
              "de points --")
        hdr = (f"{'bras':12s} {'payés b→a':>10s} {'prot.':>5s} "
               f"{'pertes':>6s} {'net':>5s} {'éco pts':>8s} "
               f"{'éco/dn':>7s} {'McN p':>7s}")
        if has_ck:
            hdr += f" {'chankan b→a':>12s}"
        print(hdr)
        for lbl in labels:
            pa = per_game_paid(arms[lbl])
            if pa is None:
                continue
            apid, apts, ackan, ackpts = pa
            prot_p = int(((bpid >= 1) & (apid == 0)).sum())
            pert_p = int(((bpid == 0) & (apid >= 1)).sum())
            eco = float(apts.sum() - bpts.sum())
            line = (f"{lbl:12s} {int(bpid.sum()):4d}→{int(apid.sum()):<5d} "
                    f"{prot_p:5d} {pert_p:6d} {prot_p - pert_p:+5d} "
                    f"{eco:+8.1f} {eco / n:+7.3f} "
                    f"{mcnemar_exact(prot_p, pert_p):7.3f}")
            if has_ck:
                line += (f" {int(bckan.sum()):5d}→{int(ackan.sum()):<6d}"
                         f" ({ackpts.sum() - bckpts.sum():+.0f} pts)")
            print(line)
        print("éco pts = points de deal-ins payés évités (unités reward "
              "= score/1000, positif = le bras paie moins) ; le dng% du "
              "tableau supra est lbl-based : il inclut les tuiles "
              "yakuless NON payables (le label contrefactuel ignore le "
              "yaku), les payés non.")
        if has_ck:
            print("chankan = rons adverses sur le shouminkan de l'agent "
                  "(exige riichi adverse) : le blend de DÉFAUSSE ne peut "
                  "pas les éviter -> garde-fou KONG dédié.\n")
        else:
            print()

    # ---------------- 2. apparié par donne (v2) -------------------------- #
    print("-- apparié par donne (v2) : tests exacts + exposition --")
    print(f"{'bras':12s} {'dRd/dn':>7s} {'cor(dd,rk)':>10s} "
          f"{'McN dd b/c':>11s} {'p_dd':>7s} {'McN wr b/c':>11s} "
          f"{'p_wr':>7s}")
    for lbl in labels:
        s = summary[lbl]
        a = arms[lbl]
        _, r, rd, dd = per_game_arrays(a)
        drank, ddd = s['drank'], s['ddd']
        drd = rd - brd
        b_dd = int(s['prot'])
        c_dd = int(s['pertes'])
        b_wr = int(((br == 1) & (r != 1)).sum())
        c_wr = int(((br != 1) & (r == 1)).sum())
        top = np.argsort(-np.abs(ddd))[:3]
        top_s = ', '.join(f"{bk[i]}({ddd[i]:+g})" for i in top
                          if ddd[i] != 0)
        print(f"{lbl:12s} {drd.mean():+7.2f} "
              f"{pearson(ddd, drank):+10.3f} "
              f"{b_dd:5d}/{c_dd:<5d} {mcnemar_exact(b_dd, c_dd):7.3f} "
              f"{b_wr:5d}/{c_wr:<5d} {mcnemar_exact(b_wr, c_wr):7.3f}")
        if top_s:
            print(f"{'':12s} top |Δdd| : {top_s}")
    print("dRd/dn = manches jouées en plus par donne (exposition : le dng% "
          "est par défausse, une trajectoire plus longue expose)\n"
          "cor(dd,rk) > 0 = les donnes où des deals sont évités (Δdd<0) "
          "gagnent du rang (Δrank<0) ; McN dd = protégées/nouvelles pertes "
          "(p exact) ;\n"
      )  # noqa: E501

    # ---------------- 3. échelle λ par τ (générique) --------------------- #
    print("-- rendement MARGINAL des swaps le long de l'échelle λ "
          "(même τ) --")
    any_ladder = False
    for lbl in labels:
        prm = arm_params(lbl)
        if not prm:
            continue
        tau, lam = prm
        succ = [l2 for l2 in labels
                if arm_params(l2)
                and arm_params(l2)[0] == tau
                and arm_params(l2)[1] > lam]
        if not succ:
            continue
        l2 = min(succ, key=lambda x: arm_params(x)[1])
        a1, a2 = arms[lbl], arms[l2]
        d_sw = a2['n_blend_swaps'] - a1['n_blend_swaps']
        d_ev = summary[l2]['avoided'] - summary[lbl]['avoided']
        d_dmr = (summary[l2]['drank'].mean()
                 - summary[lbl]['drank'].mean())
        any_ladder = True
        print(f"  tau={tau:g} : lam {lam:g} -> {arm_params(l2)[1]:g} : "
              f"+{d_sw} swaps -> {d_ev:+.1f} deal(s)-in évité(s) en plus, "
              f"DeltaMr marginal {d_dmr:+.3f}")
    if not any_ladder:
        print("  (bras λ isolé : pas d'échelle dans ce sweep)")
    print()

    # ---------------- 4. où part le coût mr ? ---------------------------- #
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

    # ---------------- 5. distribution appariée du rang ------------------- #
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

    # ---------------- 6. anomalie v1 (1,16) ------------------------------ #
    if 'tau=1,lam=16' in summary and 'tau=2,lam=16' in summary:
        print("-- anomalie v1 (1,16) : DeltaMr +0.000 avec 13.6% de swaps --")
        for lbl in ('tau=1,lam=16', 'tau=2,lam=16', 'tau=4,lam=16'):
            if lbl not in summary:
                continue
            d = summary[lbl]['drank']
            print(f"  {lbl:12s} : {int((d < 0).sum())} donnes meilleures / "
                  f"{int((d == 0).sum())} égales / {int((d > 0).sum())} pires "
                  f"| pire donne {d.max():+.0f} rang(s), "
                  f"meilleure {d.min():+.0f}")
        print()

    # ---------------- 7. intégrité vs 'paired' stockés ------------------- #
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
                        'd_rank', 'dd_base', 'dd_bras', 'd_dd',
                        'rd_base', 'rd_bras'])
            for lbl in labels:
                _, r, rd, dd = per_game_arrays(arms[lbl])
                for i, k in enumerate(bk):
                    w.writerow([lbl, k, int(br[i]), int(r[i]),
                                int(r[i] - br[i]), int(bdd[i]), int(dd[i]),
                                int(dd[i] - bdd[i]), int(brd[i]), int(rd[i])])
        print(f"[analyse] export : {args.csv}")


if __name__ == '__main__':
    main()
