# -*- coding: utf-8 -*-
"""[PLAN DANGER - POST-DIAG v3] Vraie lecture du diag_dealins.npz, sans GPU.

Le diag v1 (96 donnes) comptait comme « deal-ins » TOUT pay<0 sur une
issue en ron. Autopsie (analyze_danger_diag v2, 2026-11-17) : les 11
« fantômes » (dealt_in sans label) avaient des points médians -4.90
(q25 -6.80, q75 -2.80, total -55.7) -- des mains RÉELLES, pas des bâtons
riichi (-1.0). Cause trouvée dans multi_agent.py (bloc chankan,
~l.1093) : le ron sur le SHOUMINKAN de l'agent (chankan) sort avec
outcome='ron' -> le diag v1 l'attribuait à la dernière défausse (saine).
Le chankan exige riichi adverse : ce sont de vraies pertes, imputables
à la décision KONG, pas aux défausses.

Ce script lit le npz et sépare, selon la convention :
  * v3 (patché is_chankan) : deal-ins payés (défausses) / chankans
    payés / bâtons riichi restants / lbl yakuless non payables ;
  * v1/v2 : verdict « chankans probables » pour les fantômes à gros
    points (<= -2.0), bâtons pour les ~-1.0.

Sections :
  S1 taux réels (lbl / payés défausses / chankans / bâtons) + totaux
  S2 autopsie des écarts (fantômes, yakuless) + verdict chankan
  S3 décomposition (A)/(B)/(C) sur payés + points + croisement riichi
  S4 ciblage head-blind (rang de la fatale, 3 familles)
  S5 contre-factuel (τ, λ) : économie réelle (défaisses payées évitées)
  S6 tête par régime (AUC/ECE riichi vs silencieux) + part yakuless

Usage (racine du repo, numpy seul, ~5 s) :
    py -3.14 analyze_danger_diag.py --npz checkpoints\\danger_diag\\diag_dealins.npz
    py -3.14 analyze_danger_diag.py --npz ... --p-blind 0.05 ^
        --taus 1,2,4 --lams 8,16,32,64 --json checkpoints\\danger_diag\\diag_v3.json
"""
import argparse
import json

import numpy as np


def _q(v, ps):
    v = np.asarray(v, float)
    return ([float(np.percentile(v, p)) for p in ps] if len(v)
            else [float('nan')] * len(ps))


def _fmt_q(v, ps, fmt='{:.4f}'):
    return ' '.join(f"q{p:g}=" + fmt.format(x) for p, x in zip(ps, _q(v, ps)))


def auc_rank(y, s):
    """AUC (Mann-Whitney) avec rangs moyens pour les ex-aequo."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s), float)
    i = 0
    r = 1.0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = r + (j - i) / 2.0
        r += j - i + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece_bins(p, y, n_bins=15):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if len(p) == 0:
        return float('nan')
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def counterfactual(dat, tau, lam):
    """Miroir de diag_dealins.counterfactual_blend (trajectoire gelée)."""
    L, P, M, D, A = dat['L'], dat['P'], dat['M'], dat['D'], dat['A']
    idx = np.arange(dat['N'])
    near = M & ((dat['top1'][:, None] - L) <= tau)
    score = np.where(near, L - lam * P, -np.inf)
    a_cf = score.argmax(axis=1)
    swap = a_cf != A
    return a_cf, swap


def main():
    ap = argparse.ArgumentParser(
        description='[POST-DIAG v3] deal-ins payés vs chankans vs bâtons '
                    'riichi, ciblage head-blind et économie réelle du '
                    'blend, depuis diag_dealins.npz (conventions v1/v2/v3)'
    )
    ap.add_argument('--npz', required=True)
    ap.add_argument('--p-blind', type=float, default=0.05)
    ap.add_argument('--taus', default='1,2,4')
    ap.add_argument('--lams', default='8,16,32,64')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    z = np.load(args.npz)
    taus = [float(t) for t in str(args.taus).split(',') if t.strip()]
    lams = [float(l) for l in str(args.lams).split(',') if l.strip()]
    p_blind = args.p_blind

    N = len(z['action'])
    dat = {
        'N': N,
        'L': z['logits'].astype(np.float32),
        'P': z['p'].astype(np.float32),
        'M': z['mask'].astype(bool),
        'D': z['danger'].astype(bool),
        'A': z['action'].astype(np.int64),
        'riichi': z['riichi'].astype(bool),
        'dealt_in': z['dealt_in'].astype(bool),
        'points': z['points'].astype(np.float64),
    }
    idx = np.arange(N)
    dat['top1'] = dat['L'][idx, dat['A']]
    dat['p_ch'] = dat['P'][idx, dat['A']]
    dat['lbl_dd'] = dat['D'][idx, dat['A']]

    v3 = 'chankan_paid' in z.files
    v2 = 'stick_lost' in z.files
    conv = 'v3' if v3 else ('v2' if v2 else 'v1')
    if v3:
        dat['chankan_paid'] = z['chankan_paid'].astype(bool)
        dat['chankan_pts'] = z['chankan_pts'].astype(np.float64)
    else:
        dat['chankan_paid'] = np.zeros(N, bool)
        dat['chankan_pts'] = np.zeros(N)
    if v2:
        dat['stick_lost'] = z['stick_lost'].astype(bool)
        dat['stick_pts'] = z['stick_pts'].astype(np.float64)
    else:
        dat['stick_lost'] = np.zeros(N, bool)
        dat['stick_pts'] = np.zeros(N)

    lbl, dd, P2 = dat['lbl_dd'], dat['dealt_in'], dat['points']
    paid = dd & lbl
    ghost = dd & ~lbl
    yakuless = lbl & ~dd
    n_lbl, n_paid = int(lbl.sum()), int(paid.sum())
    n_ckan = int(dat['chankan_paid'].sum())
    ckan_pts = float(dat['chankan_pts'][dat['chankan_paid']].sum())
    riichi = dat['riichi']

    rep = {'npz': args.npz, 'convention': conv, 'n_decisions': N,
           'p_blind': p_blind}

    # ---------------- S1 : taux réels ---------------------------------- #
    pts_paid = P2[paid]
    print('=' * 76)
    print(f"[S1] {N} défausses | label lbl (dng% des sweeps) : {n_lbl} "
          f"({100 * n_lbl / max(N, 1):.2f}%) | deal-ins PAYÉS (défausses) : "
          f"{n_paid} ({100 * n_paid / max(N, 1):.2f}%)"
          + (f" | fantômes (npz {conv}) : {int(ghost.sum())}"
             if ghost.any() else ""), flush=True)
    if v3:
        print(f"     CHANKANS payés (ron adverse sur MON shouminkan) : "
              f"{n_ckan}, {ckan_pts:.1f} pts -- pertes réelles, décision "
              f"KONG, PAS des défausses (garde-fou kong = patch suivant)",
              flush=True)
    if paid.any():
        print(f"     points défausses payées (unités reward=score/1000) : "
              f"total {pts_paid.sum():.1f} | ĉ moyen {pts_paid.mean():.3f} | "
              f"médian {np.median(pts_paid):.3f} | "
              + _fmt_q(pts_paid, (10, 90), '{:.2f}'), flush=True)
    tot_ron = pts_paid.sum() + ckan_pts
    n_ron = n_paid + n_ckan
    if n_ron:
        print(f"     TOTAL pertes par ron (défausses + chankans) : "
              f"{n_ron} paiements, {tot_ron:.1f} pts | taux réel "
              f"{100 * n_ron / max(N, 1):.2f}% | ĉ global "
              f"{tot_ron / n_ron:.3f}", flush=True)
    rep['rates'] = {'n_lbl': n_lbl, 'n_paid': n_paid,
                    'paid_pct': 100 * n_paid / max(N, 1),
                    'lbl_pct': 100 * n_lbl / max(N, 1),
                    'n_ghost': int(ghost.sum()),
                    'n_yakuless': int(yakuless.sum()),
                    'n_chankan': n_ckan, 'chankan_points': ckan_pts,
                    'c_hat': (float(pts_paid.mean()) if n_paid else None),
                    'points_paid_total': (float(pts_paid.sum())
                                          if n_paid else 0.0),
                    'total_ron_points': float(tot_ron),
                    'c_hat_global': (float(tot_ron / n_ron)
                                     if n_ron else None)}

    # ---------------- S2 : autopsie des écarts ------------------------- #
    print()
    print(f"[S2] autopsie des écarts "
          f"({int((lbl != dd).sum())} au total, convention {conv}) :",
          flush=True)
    if ghost.any():
        gp = P2[ghost]
        med = float(np.median(gp)) if len(gp) else float('nan')
        if v3:
            verdict = ('RÉSIDUELS inexpliqués : à ré-examiner (la clé '
                       'is_chankan aurait dû les couvrir)')
        elif med <= -2.0:
            verdict = ('CHANKANS probables (ron adverse sur MON shouminkan, '
                       'exige riichi adverse ; outcome "ron" cf. '
                       'multi_agent.py bloc chankan : le diag v1 les '
                       'comptait sur la dernière défausse). Confirmation : '
                       'patch v3 + re-run diag (is_chankan)')
        else:
            verdict = 'batons riichi (~-1.0) : confirmés'
        print(f"     dealt_in SANS label : {int(ghost.sum())} | points "
              + _fmt_q(gp, (25, 50, 75), '{:.2f}')
              + f" -> {verdict}", flush=True)
    if yakuless.any():
        yk = dat['p_ch'][yakuless]
        hi = float((yk >= p_blind).mean())
        print(f"     label SANS ron (yakuless, non payables) : "
              f"{int(yakuless.sum())} | p de la tête sur ces tuiles : "
              + _fmt_q(yk, (25, 50, 75, 90))
              + f" | {100 * hi:.0f}% avaient p>={p_blind} (le blend y "
              f"brûle des swaps pour rien)", flush=True)
    rep['ghosts'] = {'n': int(ghost.sum()),
                     'points_median': (float(np.median(P2[ghost]))
                                       if ghost.any() else None)}
    rep['yakuless'] = {'n': int(yakuless.sum()),
                       'pct_p_ge_blind': (float((dat['p_ch'][yakuless]
                                                >= p_blind).mean())
                                          if yakuless.any() else None)}

    # ---------------- S3 : décomposition (A)/(B)/(C) sur PAYÉS --------- #
    blind = paid & (dat['p_ch'] < p_blind)
    visible = paid & (dat['p_ch'] >= p_blind)
    tot_abs = abs(P2[blind].sum()) + abs(P2[visible].sum())
    print()
    print(f"[S3] décomposition des {n_paid} deal-ins PAYÉS "
          f"(p_blind={p_blind}) :", flush=True)
    if n_paid:
        print(f"     (A) head-blind : {int(blind.sum()):3d} "
              f"({100 * blind.sum() / n_paid:4.1f}%) | "
              f"{100 * abs(P2[blind].sum()) / max(tot_abs, 1e-9):4.1f}% "
              f"des points | ĉ {P2[blind].mean():.2f}", flush=True)
        print(f"     visible       : {int(visible.sum()):3d} "
              f"({100 * visible.sum() / n_paid:4.1f}%) | "
              f"{100 * abs(P2[visible].sum()) / max(tot_abs, 1e-9):4.1f}% "
              f"des points | ĉ {P2[visible].mean():.2f}", flush=True)
        for nm, m in (('riichi', riichi), ('silencieux', ~riichi)):
            b, v = int((blind & m).sum()), int((visible & m).sum())
            print(f"       {nm:10s} : blind {b:3d} | visible {v:3d} | "
                  f"ĉ {P2[paid & m].mean():.2f}"
                  if (paid & m).any() else
                  f"       {nm:10s} : aucun deal-in payé", flush=True)
        print("     Lecture : blind concentré SILENCIEUX -> tête ; visible "
              "concentré RIICHI + hand-locked -> policy (fold/planif).",
              flush=True)
    rep['decomposition'] = {
        'n_paid': n_paid, 'A_head_blind': int(blind.sum()),
        'visible': int(visible.sum()),
        'points_A': float(P2[blind].sum()),
        'points_visible': float(P2[visible].sum()),
        'c_hat_blind': (float(P2[blind].mean()) if blind.any() else None),
        'c_hat_visible': (float(P2[visible].mean())
                          if visible.any() else None),
        'blind_riichi': int((blind & riichi).sum()),
        'blind_silencieux': int((blind & ~riichi).sum())}

    for tau in taus:
        safe_near = (dat['M'] & ~dat['D']
                     & ((dat['top1'][:, None] - dat['L']) <= tau))
        has_safe = safe_near.any(axis=1)
        C = int((visible & has_safe).sum())
        B = int((visible & ~has_safe).sum())
        gap = np.where(dat['M'] & ~dat['D'],
                       dat['top1'][:, None] - dat['L'], np.inf).min(axis=1)
        med_gap = (float(np.median(gap[visible & ~has_safe]))
                   if (visible & ~has_safe).any() else None)
        print(f"     tau={tau:g} : (C) blendable-manqué {C:3d} | "
              f"(B) hand-locked {B:3d}"
              + (f" (gap méd {med_gap:.1f} logits)" if med_gap is not None
                 else ""), flush=True)
        rep[f'tau={tau:g}'] = {'C': C, 'B': B, 'med_gap': med_gap}

    # ---------------- S4 : ciblage head-blind -------------------------- #
    print()
    print("[S4] ciblage des head-blind : que voyait la tête ? "
          "(tuiles légales)", flush=True)
    if blind.any() or visible.any():
        S = np.where(dat['M'], dat['P'], -1.0)
        order = np.argsort(-S, axis=1, kind='stable')
        pos = np.empty((N, 34), np.int64)
        pos[np.arange(N)[:, None], order] = np.arange(34)[None, :]
        rank_fat = pos[idx, dat['A']] + 1        # 1 = la plus dangereuse
        other = dat['M'].copy()
        other[idx, dat['A']] = False
        p_max = np.where(other, dat['P'], -1.0).max(axis=1)
        p_fat = dat['p_ch']
        for nm, m in (('blind', blind), ('visible', visible)):
            if not m.any():
                continue
            r = rank_fat[m]
            print(f"     {nm:8s} ({int(m.sum()):2d}) : rang de la tuile "
                  f"fatale méd {np.median(r):.0f} | top-1 "
                  f"{100 * (r == 1).mean():3.0f}% | top-3 "
                  f"{100 * (r <= 3).mean():3.0f}% | top-5 "
                  f"{100 * (r <= 5).mean():3.0f}% | >10 "
                  f"{100 * (r > 10).mean():3.0f}%", flush=True)
            rep[f'rank_fatal_{nm}'] = {
                'n': int(m.sum()), 'median': float(np.median(r)),
                'pct_top1': float((r == 1).mean()),
                'pct_top3': float((r <= 3).mean()),
                'pct_top5': float((r <= 5).mean()),
                'pct_gt10': float((r > 10).mean())}
        fam = {}
        if blind.any():
            m = blind
            f1 = m & (p_max < p_blind)
            f2 = m & (p_max >= np.maximum(2.0 * p_fat, 0.10))
            f3 = m & ~f1 & ~f2
            fam = {'aveugle_total_pmax<blind': int(f1.sum()),
                   'localisation_ratee_pmax>=max(2p,0.10)': int(f2.sum()),
                   'vu_mais_sous_estime': int(f3.sum())}
            print(f"     familles blind : aveugle total (p_max<{p_blind}) "
                  f"{f1.sum()} | localisation ratée (p_max>=2×p ou 0.10) "
                  f"{f2.sum()} | vu mais sous-estimé {f3.sum()}", flush=True)
            if f2.any():
                print("       localisation ratée : p_max concurrent "
                      + _fmt_q(p_max[f2], (25, 50, 75))
                      + f" vs p fatale {_fmt_q(p_fat[f2], (25, 50, 75))}",
                      flush=True)
        rep['familles_blind'] = fam
        print("     Lecture : aveugle total -> features manquantes "
              "(dump/entrées) ; localisation ratée -> la tête sent le "
              "danger mais pas la bonne tuile (lecture riichi/genbutsu) ;"
              "\n                vu mais sous-estimé -> échelle de "
              "calibration par régime, tête discriminante.", flush=True)

    # ---------------- S5 : contre-factuel économique ------------------- #
    print()
    print("[S5] contre-factuel blend (τ, λ), trajectoire gelée — "
          "ÉCONOMIE RÉELLE (défausses payées évitées) :", flush=True)
    print(f"     {'tau':>4s} {'lam':>5s} {'swaps%':>7s} {'évités':>7s} "
          f"{'dont payés':>11s} {'pts économisés':>15s} {'créés lbl':>9s}",
          flush=True)
    cf_rep = {}
    for tau in taus:
        for lam in lams:
            a_cf, swap = counterfactual(dat, tau, lam)
            save_lbl = swap & lbl & ~dat['D'][idx, a_cf]
            save_paid = swap & paid & ~dat['D'][idx, a_cf]
            create_lbl = swap & ~lbl & dat['D'][idx, a_cf]
            eco = float(-P2[save_paid].sum())
            print(f"     {tau:4g} {lam:5g} {100 * swap.mean():7.2f} "
                  f"{int(save_lbl.sum()):7d} {int(save_paid.sum()):11d} "
                  f"{eco:15.1f} {int(create_lbl.sum()):9d}", flush=True)
            cf_rep[f'tau={tau:g},lam={lam:g}'] = {
                'swaps': int(swap.sum()), 'swap_pct': float(swap.mean()),
                'evites_lbl': int(save_lbl.sum()),
                'evites_payes': int(save_paid.sum()),
                'points_economises': eco,
                'crees_lbl': int(create_lbl.sum())}
    rep['counterfactual'] = cf_rep
    if n_lbl:
        pay_ratio = n_paid / n_lbl
        print(f"     (les 'évités' lbl incluent ~{100 * (1 - pay_ratio):.0f}% "
              f"de tuiles yakuless non payables -> l'économie réelle "
              f"'dont payés' est la métrique de décision)", flush=True)
    if n_ckan:
        print(f"     NB : les {n_ckan} chankans ({ckan_pts:.1f} pts) ne "
              f"sont PAS atteignables par le blend de défausse -- "
              f"garde-fou KONG dédié (refuser shouminkan si "
              f"p_danger(tuile du kan) >= seuil).", flush=True)

    # ---------------- S6 : tête par régime ----------------------------- #
    print()
    print("[S6] tête danger par régime (label lbl, toutes tuiles "
          "légales) :", flush=True)
    M, D, Pm = dat['M'], dat['D'], dat['P']
    for nm, m in (('tous', np.ones(N, bool)), ('riichi', riichi),
                  ('silencieux', ~riichi)):
        mm = M & m[:, None]
        if mm.any() and (D & mm).any():
            print(f"     {nm:10s} : AUC {auc_rank(D[mm], Pm[mm]):.4f} | "
                  f"ECE {ece_bins(Pm[mm], D[mm].astype(float)):.4f} | "
                  f"{int(m.sum())} états", flush=True)
            rep[f'head_{nm}'] = {
                'auc': float(auc_rank(D[mm], Pm[mm])),
                'ece': float(ece_bins(Pm[mm], D[mm].astype(float)))}
    for nm, m in (('riichi', riichi), ('silencieux', ~riichi)):
        if (lbl & m).any():
            yk = float((yakuless & m).sum()) / float((lbl & m).sum())
            print(f"     part yakuless des lbl en {nm} : "
                  f"{100 * yk:.0f}% "
                  + ("(riichi = yaku garanti : lbl quasi payables)"
                     if nm == 'riichi' else
                     "(le dng% silencieux est un proxy bruité)"),
                  flush=True)
            rep[f'yakuless_share_{nm}'] = yk

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print()
        print(f"[out] {args.json}", flush=True)


if __name__ == '__main__':
    main()
