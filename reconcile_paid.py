# -*- coding: utf-8 -*-
"""[POST-V2D] Réconciliation des compteurs « deal-ins payés » diag vs sweep.

Constat à élucider (MÊMES 96 donnes, rejouage bit-exact, 6297 défausses
et 80 lbl identiques des deux côtés) :

  * diag_dealins.npz (v3) : 61 deal-ins PAYÉS, -254.8 pts -- convention :
    la dernière défausse de la manche est marquee dealt_in=points SI le
    ron adverse tombe dessus ET qu'elle est lbl (marquage v2/v3) ;
  * sweep v2d, bras baseline (results.json) : PAYÉS 55, -328 pts --
    compteur n_dealins_paid du runner, remonte par partie ('dpid').

6 événements et ~73 pts d'écart. Avant que « payés / économie » ne
devienne la métrique de la porte v2e, il faut UNE définition et savoir
laquelle des deux comptabilités reflète la vérité. Ce script joint les
deux sources PAR DONNE (chunk, env) et liste les donnes en désaccord
avec le détail des lignes npz (lbl / dealt_in / points / bâtons /
chankan par manche), plus les rounds du diag_summary.json si fourni.

Hypothèses que la sortie doit trancher :
  H1  le runner exclut des paiements que le diag marque (ou l'inverse) ;
  H2  double ron sur la même défausse (1 manche, 2 paiements) ;
  H3  conventions de points différentes (honba / bâtons inclus ou non
      dans points_paid du runner : -328 vs -254.8 - 55.7 = -310.5, il
      manque encore ~17.5) ;
  H4  les 11 « non imputables » (stick_lost, -55.7 pts, ~-5/manche) :
      ni des bâtons (-1.0) ni des chankans (0 comptés) -> probablement
      la part agent de TSUMO adverses (mangan/haneman), à vérifier dans
      les rounds du summary.

Usage (racine du repo, numpy seul, ~5 s) :
    py -3.14 reconcile_paid.py --npz checkpoints\\danger_diag_v3\\diag_dealins.npz ^
        --results checkpoints\\danger_blend_sweep_v2d\\results.json ^
        [--summary checkpoints\\danger_diag_v3\\diag_summary.json]
"""
import argparse
import json

import numpy as np

TILE34 = ([f'{n}m' for n in range(1, 10)]
          + [f'{n}p' for n in range(1, 10)]
          + [f'{n}s' for n in range(1, 10)]
          + ['E', 'S', 'W', 'N', 'haku', 'hatsu', 'chun'])


def t34(i):
    return TILE34[int(i)]


def load_npz(path):
    z = np.load(path)
    need = ('danger', 'action', 'dealt_in', 'points', 'chunk', 'env',
            'round')
    miss = [k for k in need if k not in z.files]
    if miss:
        raise SystemExit(f"[rec] npz sans clés {miss} -- convention v3 "
                         f"attendue (re-run diag_dealins v3)")
    N = len(z['action'])
    idx = np.arange(N)
    lbl_dd = z['danger'][idx, z['action'].astype(np.int64)] > 0
    rows = {
        'N': N,
        'chunk': z['chunk'].astype(np.int64),
        'env': z['env'].astype(np.int64),
        'round': z['round'].astype(np.int64),
        'action': z['action'].astype(np.int64),
        'lbl_dd': lbl_dd,
        'dealt_in': z['dealt_in'].astype(bool),
        'points': z['points'].astype(np.float64),
        'paid': z['dealt_in'].astype(bool) & lbl_dd,
        'stick_lost': (z['stick_lost'].astype(bool)
                       if 'stick_lost' in z.files
                       else np.zeros(N, bool)),
        'stick_pts': (z['stick_pts'].astype(np.float64)
                      if 'stick_pts' in z.files
                      else np.zeros(N)),
        'chankan_paid': (z['chankan_paid'].astype(bool)
                         if 'chankan_paid' in z.files
                         else np.zeros(N, bool)),
        'chankan_pts': (z['chankan_pts'].astype(np.float64)
                        if 'chankan_pts' in z.files
                        else np.zeros(N)),
        'p_ch': (z['p'][idx, z['action'].astype(np.int64)].astype(np.float64)
                 if 'p' in z.files else np.full(N, np.nan)),
    }
    return rows


def per_game_npz(rows):
    """Agrège les lignes npz par donne (chunk, env)."""
    out = {}
    for i in range(rows['N']):
        k = (int(rows['chunk'][i]), int(rows['env'][i]))
        g = out.setdefault(k, {'n_rows': 0, 'n_lbl': 0, 'n_paid': 0,
                               'pts_paid': 0.0, 'n_stick': 0,
                               'stick_pts': 0.0, 'n_ckan': 0,
                               'ckan_pts': 0.0, 'rows': []})
        g['n_rows'] += 1
        g['n_lbl'] += int(rows['lbl_dd'][i])
        g['n_paid'] += int(rows['paid'][i])
        g['pts_paid'] += rows['points'][i] if rows['paid'][i] else 0.0
        g['n_stick'] += int(rows['stick_lost'][i])
        g['stick_pts'] += (rows['stick_pts'][i]
                           if rows['stick_lost'][i] else 0.0)
        g['n_ckan'] += int(rows['chankan_paid'][i])
        g['ckan_pts'] += (rows['chankan_pts'][i]
                          if rows['chankan_paid'][i] else 0.0)
        g['rows'].append(i)
    return out


def main():
    ap = argparse.ArgumentParser(
        description='[POST-V2D] réconciliation payés diag vs sweep '
                    '(jointure par donne chunk.env)')
    ap.add_argument('--npz', required=True)
    ap.add_argument('--results', required=True,
                    help='results.json du sweep v2d (bras baseline)')
    ap.add_argument('--summary', default=None,
                    help='diag_summary.json (optionnel : rounds détaillés '
                         'des donnes en désaccord)')
    ap.add_argument('--arm', default='baseline')
    args = ap.parse_args()

    rows = load_npz(args.npz)
    gnpz = per_game_npz(rows)

    with open(args.results, encoding='utf-8') as f:
        res = json.load(f)
    if args.arm not in res.get('arms', {}):
        raise SystemExit(f"[rec] bras '{args.arm}' absent de "
                         f"{args.results} (bras : "
                         f"{sorted(res.get('arms', {}))})")
    cfg = res.get('config', {})
    pg = res['arms'][args.arm].get('per_game', {})
    print('=' * 76)
    print(f"[rec] npz {args.npz}")
    print(f"[rec] sweep {args.results} | bras {args.arm} | games "
          f"{cfg.get('games')} | taus {cfg.get('taus')} | lams "
          f"{cfg.get('lams')} | seed {cfg.get('seed')}", flush=True)

    # --- canary global : mêmes donnes, mêmes défausses, mêmes lbl ------ #
    rd_sw = sum(int(v['rd']) for v in pg.values())
    dd_sw = sum(int(v['dd']) for v in pg.values())
    n_lbl_npz = int(sum(g['n_lbl'] for g in gnpz.values()))
    print(f"[rec] canary global : défausses npz {rows['N']} vs sweep "
          f"{rd_sw} | lbl npz {n_lbl_npz} vs sweep {dd_sw}"
          + ("  (OK)" if rows['N'] == rd_sw and n_lbl_npz == dd_sw
             else "  (ÉCART : ne pas interpréter la suite !)"),
          flush=True)

    # --- jointure par donne -------------------------------------------- #
    keys_npz = set(gnpz)
    keys_sw = {(int(k.split('.')[0]), int(k.split('.')[1]))
               for k in pg}
    tot = {'dpid': 0, 'paid_npz': 0, 'pts_sw': 0.0, 'pts_npz': 0.0,
           'ckan_sw': 0, 'ckan_npz': 0}
    bad = []
    for k in sorted(keys_npz | keys_sw):
        s = pg.get(f'{k[0]}.{k[1]}')
        g = gnpz.get(k)
        if s is None or g is None:
            bad.append((k, 'DONNE ABSENTE d\'un côté', None))
            continue
        tot['dpid'] += int(s.get('dpid', 0))
        tot['paid_npz'] += g['n_paid']
        tot['pts_sw'] += float(s.get('pts', 0.0))
        tot['pts_npz'] += g['pts_paid']
        tot['ckan_sw'] += int(s.get('ckan', 0))
        tot['ckan_npz'] += g['n_ckan']
        mis = []
        if int(s.get('dpid', 0)) != g['n_paid']:
            mis.append(f"payés sweep {s.get('dpid')} vs npz {g['n_paid']}")
        if abs(float(s.get('pts', 0.0)) - g['pts_paid']) > 0.05:
            mis.append(f"pts sweep {s.get('pts', 0.0):.1f} vs npz "
                       f"{g['pts_paid']:.1f}")
        if int(s.get('ckan', 0)) != g['n_ckan']:
            mis.append(f"chankan {s.get('ckan')} vs {g['n_ckan']}")
        if mis:
            bad.append((k, ' ; '.join(mis), g))

    print()
    print(f"[rec] TOTAUX bras {args.arm} : payés sweep {tot['dpid']} vs "
          f"npz {tot['paid_npz']} | pts sweep {tot['pts_sw']:.1f} vs npz "
          f"{tot['pts_npz']:.1f} | chankans {tot['ckan_sw']} vs "
          f"{tot['ckan_npz']}", flush=True)
    n_diag_plus = sum(1 for k, m, g in bad
                      if g is not None and 'payés' in m
                      and g['n_paid'] > int(pg[f'{k[0]}.{k[1]}'].get('dpid', 0)))
    n_sw_plus = sum(1 for k, m, g in bad
                    if g is not None and 'payés' in m
                    and g['n_paid'] < int(pg[f'{k[0]}.{k[1]}'].get('dpid', 0)))
    print(f"[rec] donnes en désaccord : {len(bad)} "
          f"(npz compte + : {n_diag_plus} donnes | sweep compte + : "
          f"{n_sw_plus} donnes)", flush=True)

    # --- détail des donnes en désaccord -------------------------------- #
    rounds_by_game = {}
    if args.summary:
        with open(args.summary, encoding='utf-8') as f:
            summ = json.load(f)
        for r in summ.get('rounds', []):
            rounds_by_game.setdefault(
                (int(r.get('chunk', -1)), int(r.get('env_idx', -1))),
                []).append(r)

    for k, msg, g in bad:
        print()
        print(f"---- donne chunk {k[0]} env {k[1]} : {msg}")
        if g is None:
            continue
        s = pg[f'{k[0]}.{k[1]}']
        print(f"     sweep : rd {s['rd']} | dd {s['dd']} | dpid "
              f"{s.get('dpid')} | pts {s.get('pts', 0.0):.1f} | ckan "
              f"{s.get('ckan')}")
        print(f"     npz   : rows {g['n_rows']} | lbl {g['n_lbl']} | "
              f"payés {g['n_paid']} | pts {g['pts_paid']:.1f} | sticks "
              f"{g['n_stick']} ({g['stick_pts']:.1f} pts) | chankan "
              f"{g['n_ckan']} ({g['ckan_pts']:.1f} pts)")
        for i in g['rows']:
            interesting = (rows['paid'][i] or rows['lbl_dd'][i]
                           or rows['stick_lost'][i]
                           or rows['chankan_paid'][i])
            if not interesting:
                continue
            flags = []
            if rows['paid'][i]:
                flags.append(f"PAYÉ {rows['points'][i]:.1f}")
            if rows['lbl_dd'][i] and not rows['paid'][i]:
                flags.append('lbl (non payé)')
            if rows['stick_lost'][i]:
                flags.append(f"stick {rows['stick_pts'][i]:.1f}")
            if rows['chankan_paid'][i]:
                flags.append(f"chankan {rows['chankan_pts'][i]:.1f}")
            print(f"       r{rows['round'][i]:02d} tuile "
                  f"{t34(rows['action'][i]):5s} p {rows['p_ch'][i]:.4f} "
                  f"-> {' | '.join(flags)}", flush=True)
        for r in rounds_by_game.get(k, []):
            print(f"       [summary round] {json.dumps(r, ensure_ascii=False)}",
                  flush=True)

    print()
    if not bad:
        print("[rec] AUCUN désaccord par donne : les deux compteurs "
              "coïncident partout (l'écart 55/61 vient alors d'un total "
              "agrégé différent -- vérifier les donnes couvertes).")
    else:
        print("[rec] Lecture : comparer 'PAYÉ x' (sweep) vs 'payés y' "
              "(npz) donne par donne ci-dessus ; si l'écart se "
              "concentre sur des manches stick ~-5 sans lbl, cf. H4 "
              "(tsumo adverses payés par l'agent). Coller cette sortie "
              "pour arbitrer la convention de la porte v2e.")


if __name__ == '__main__':
    main()
