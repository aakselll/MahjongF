"""[PLAN DANGER - post-sweep] Audit du checkpoint danger_head.pt.

Le sweep a loggé « calib embarquée ? » au chargement : le checkpoint chargé
n'a PAS la clé 'calib', donc il préexiste probablement à la version du
trainer qui embarque la correction −log(pos_weight) dans le biais. Si c'est
confirmé, le p blendé dans le sweep était le score BRUT (BCE pos_weightée,
moyenne ~0.4-0.5) et non une proba : l'axe λ du sweep n'était pas en unités
de probabilité.

Partie 1 (toujours, ~1 s, aucun import du repo) :
  * clés du checkpoint, calib, pos_weight, prior, métriques val stockées
    (AUC, ECE, AUC riichi/silencieux, métriques de déploiement swap_ok /
    faux_swap par delta) ;
  * sigma(biais conv2) vs prior (= logit attendu si corrigé) et vs optimum
    pos_weighté (= ~0.5 si NON corrigé) -> verdict indicatif.

Partie 2 (optionnelle, --data, depuis la racine du repo) :
  * rejoue la tête sur le fold val des shards : quantiles de p brut vs
    p calibré, ECE des deux, AUC de contrôle, et la pénalité EFFECTIVE
    que le sweep a appliquée (λ·p) pour lambda in {1, 4, 16}.

Usage :
    py -3.14 audit_danger_head.py
    py -3.14 audit_danger_head.py --data checkpoints\\danger_data
    py -3.14 audit_danger_head.py --fix-out checkpoints\\danger_head\\danger_head_cal.pt
        (écrit une COPIE corrigée du checkpoint si l'audit conclut
         « non corrigé » ; l'original n'est jamais modifié)
"""
import argparse
import math
import os

import numpy as np
import torch


def fmt(x, nd=4):
    return '?' if x is None else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser(description='Audit du checkpoint danger head')
    ap.add_argument('--head',
                    default=os.path.join('checkpoints', 'danger_head',
                                         'danger_head.pt'))
    ap.add_argument('--sl-discard',
                    default=os.path.join('checkpoints', 'sl_tonpusen_la',
                                         'discard.pt'),
                    help='pour --data : reconstruire le tronc SL figé')
    ap.add_argument('--data', default=None,
                    help='dossier des shards de l\'étape 1 (partie 2)')
    ap.add_argument('--pos-weight', type=float, default=None,
                    help='pos_weight si absent du checkpoint (console étape '
                         '2 : « pos_weight=XX.X »)')
    ap.add_argument('--max-states', type=int, default=20000,
                    help='plafond d\'états val pour la partie 2')
    ap.add_argument('--fix-out', default=None,
                    help='écrit un checkpoint corrigé (copie) si non corrigé')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    ck = torch.load(args.head, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or 'head' not in ck:
        raise SystemExit(f"{args.head} n'est pas un checkpoint danger_head "
                         f"(clé 'head' absente)")

    # ------------------- partie 1 : inventaire du checkpoint ------------- #
    print('=' * 78)
    print(f"[audit] checkpoint : {args.head}")
    print(f"[audit] clés       : {sorted(ck.keys())}")
    calib = ck.get('calib')
    pos_w = ck.get('pos_weight', args.pos_weight)
    prior = ck.get('prior')
    print(f"[audit] calib     : {calib if calib else 'ABSENT'}")
    print(f"[audit] pos_weight: {fmt(pos_w, 1)} | prior: "
          f"{fmt(100 * prior, 3) + '%' if prior is not None else '?'}"
          f" | epoch: {ck.get('epoch', '?')}")

    bias = float(ck['head']['conv2.bias'].flatten()[0])
    sig = 1.0 / (1.0 + math.exp(-bias))
    print(f"[audit] sigma(biais conv2) = {sig:.4f} (logit {bias:+.3f})")
    if prior is not None:
        p_corr = prior                       # attendu si corrigé
        if pos_w is not None:
            p_opt = pos_w * prior / (pos_w * prior + 1.0 - prior)
        else:
            p_opt = None
        print(f"[audit] attendu si CORRIGÉ    : sigma(biais) ~ {p_corr:.4f} "
              f"(logit {math.log(p_corr / (1 - p_corr)):+.2f})")
        if p_opt is not None:
            print(f"[audit] attendu si NON CORRIGÉ: sigma(biais) ~ {p_opt:.4f} "
                  f"(logit {math.log(p_opt / (1 - p_opt)):+.2f})")
        if calib:
            verdict = f"CORRIGÉ (calib embarquée : {calib.get('method')})"
        elif p_opt is not None:
            verdict = ('NON CORRIGÉ (biais ~ optimum pos_weighté)'
                       if abs(sig - p_opt) < abs(sig - p_corr)
                       else 'CORRIGÉ (biais ~ prior)')
        else:
            verdict = ('NON CORRIGÉ (biais ~0.5, typique du score brut)'
                       if sig > 0.2 else 'probablement CORRIGÉ')
    else:
        verdict = ('probablement NON CORRIGÉ (sigma(biais) > 0.2)'
                   if sig > 0.2 else 'incertain (prior absent du ck)')
    print(f"[audit] VERDICT indicatif : {verdict}")
    print('=' * 78)

    met = ck.get('metrics', {})
    if met:
        print(f"[audit] métriques val stockées : AUC {fmt(met.get('auc'))} "
              f"(plancher unseen {fmt(met.get('auc_floor_eff'))}) | "
              f"ECE {fmt(met.get('ece'))}")
        print(f"[audit]   AUC riichi {fmt(met.get('auc_riichi'))} / "
              f"silencieux {fmt(met.get('auc_silent'))} "
              f"({100 * met.get('pct_riichi_states', 0):.0f}% des états val "
              f"avec riichi adverse)")
        d = met.get('deployment')
        if d:
            print(f"[audit] déploiement : SL argmax joue une tuile dangereuse "
                  f"sur {fmt(d.get('sl_argmax_dng_pct'), 2)}% des "
                  f"{d.get('n_states', '?')} états val")
            for k in sorted(d):
                if not k.startswith('delta='):
                    continue
                dm = d[k]
                hit, n_hit = dm['swap_ok']
                fs, n_fs = dm['faux_swap']
                print(f"[audit]   {k:9s} : {dm['n_alt_mean']:.1f} alt/état | "
                      f"swap_ok {hit}/{n_hit} "
                      f"({100 * hit / max(n_hit, 1):.1f}%) | faux_swap "
                      f"{fs}/{n_fs} ({100 * fs / max(n_fs, 1):.1f}%) | "
                      f"AUC_near {fmt(dm['auc_near'])}")
        print('=' * 78)

    # ------------------- copie corrigée (optionnel) ---------------------- #
    if args.fix_out:
        if calib:
            print("[audit] --fix-out ignoré : le checkpoint est déjà corrigé")
        elif pos_w is None:
            raise SystemExit("[audit] --fix-out impossible : pos_weight "
                             "inconnu (le passer via --pos-weight, cf. "
                             "console étape 2)")
        elif 'NON CORRIGÉ' not in verdict:
            print(f"[audit] --fix-out ignoré : verdict = {verdict}")
        else:
            sd = {k: v.clone() for k, v in ck['head'].items()}
            sd['conv2.bias'] = sd['conv2.bias'] - math.log(pos_w)
            ck2 = dict(ck)
            ck2['head'] = sd
            ck2['calib'] = {'method': 'prior_correction_pos_weight_a_posteriori',
                            'logit_shift': -float(math.log(pos_w)),
                            'source_checkpoint': os.path.abspath(args.head)}
            torch.save(ck2, args.fix_out)
            new_bias = float(sd['conv2.bias'].flatten()[0])
            print(f"[audit] copie corrigée écrite : {args.fix_out}")
            print(f"[audit]   sigma(biais) {sig:.4f} -> "
                  f"{1 / (1 + math.exp(-new_bias)):.4f} "
                  f"(shift -log({pos_w:.1f}) = {-math.log(pos_w):.3f})")

    # ------------------- partie 2 : rejouer le fold val ------------------ #
    if not args.data:
        print("[audit] (partie 2 : ajouter --data checkpoints\\danger_data "
              "pour rejouer le fold val)")
        return
    try:
        from mahjong_ai.rl.train_danger_head import (
            DangerNet, load_shard, scan_shards, N_FOLDS, ece_bins, auc_rank)
    except ImportError as e:
        raise SystemExit(f"[audit] partie 2 : lancer depuis la racine du "
                         f"repo ({e})")
    if pos_w is None:
        raise SystemExit("[audit] partie 2 : pos_weight inconnu -- le "
                         "passer via --pos-weight")
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ck.get('config', {})
    model = DangerNet(args.sl_discard, hidden=cfg.get('hidden', 128),
                      device=device)
    model.danger.load_state_dict(ck['head'])
    model.eval()
    fold = ck.get('val_fold', 0)

    files = scan_shards(args.data)
    p_all, y_all = [], []
    n_seen = 0
    for f in files:
        z = load_shard(f)
        m = (z['game'] % N_FOLDS) == fold
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        if len(idx) + n_seen > args.max_states:
            rng = np.random.RandomState(0)
            idx = rng.choice(idx, args.max_states - n_seen, replace=False)
        for s in range(0, len(idx), 512):
            j = idx[s:s + 512]
            with torch.no_grad():
                logits = model.danger_logits(z['obs_normal'][j])
            p = torch.sigmoid(logits).cpu().numpy().astype('float64')
            mb = z['mask'][j] > 0
            yb = z['danger'][j] > 0
            p_all.append(p[mb])
            y_all.append(yb[mb])
        n_seen += len(idx)
        if n_seen >= args.max_states:
            break
    p = np.concatenate(p_all)
    y = np.concatenate(y_all).astype('float64')
    p_cal = p / (p + pos_w * (1.0 - p))
    qs = (0.10, 0.50, 0.90, 0.99)
    print(f"[audit] fold val {fold} : {n_seen} états, "
          f"{len(p)} tuiles légales | taux danger réel "
          f"{100 * y.mean():.3f}%")
    print(f"[audit] quantiles p BRUT     (celui du sweep) : "
          + ' '.join(f"q{int(100 * q)}={np.quantile(p, q):.3f}" for q in qs))
    print(f"[audit] quantiles p CALIBRÉ (attendu)         : "
          + ' '.join(f"q{int(100 * q)}={np.quantile(p_cal, q):.3f}"
                     for q in qs))
    print(f"[audit] ECE brut {ece_bins(p, y):.4f} vs calibré "
          f"{ece_bins(p_cal, y):.4f} | AUC {auc_rank(y, p):.4f} "
          f"(stocké {fmt(met.get('auc'))})")
    print()
    print("[audit] pénalité EFFECTIVE appliquée par le sweep (λ·p, en "
          "logits) :")
    print(f"{'λ':>4s} {'pénalité médiane BRUT':>21s} {'q90 BRUT':>10s} "
          f"{'médiane calibré':>17s}")
    for lam in (1.0, 4.0, 16.0):
        print(f"{lam:4.0f} {lam * np.quantile(p, 0.5):19.3f} "
              f"{lam * np.quantile(p, 0.9):10.3f} "
              f"{lam * np.quantile(p_cal, 0.5):17.3f}")
    print()
    print("[audit] lecture : si la colonne BRUT est 5-20x la colonne "
          "calibré, le sweep a\n        sur-pénalisé d'un facteur massif ; "
          "le λ pré-enregistré « en unités de\n        proba » n'a jamais "
          "été testé. Ré-sweeper sur la copie corrigée avec\n        "
          "λ in {0.5, 1, 2, 4} x τ in {1, 2, 4} (--games 96 en "
          "préversion).")


if __name__ == '__main__':
    main()
