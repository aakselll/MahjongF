"""[PLAN DANGER - post-sweep] Audit du checkpoint danger_head.pt.

CONTEXTE (audit du 2026-11-16, après le sweep v1 échoué) : le checkpoint
n'a PAS la clé 'calib' -> il préexiste à la version du trainer qui embarque
la correction −log(pos_weight) dans le biais. Et le test « biais ~ prior »
de la partie 1 s'est révélé NON DISCRIMINANT (faux positif « CORRIGÉ ») :

  * biais conv2 observé = −3.47 ≈ logit(prior) = −3.52 : c'est la VALEUR
    D'INIT du trainer (biais ← logit(prior)), pas une preuve de correction.
    En ~3 époques d'AdamW (lr 3e-4, ~1000-1400 pas), le biais ne peut
    bouger que de ±0.3-0.4 logits : il n'absorbe PAS le +log(pos_weight).
  * L'inflation des odds (×34.8 ≈ pos_weight 33.8) vit donc dans les POIDS
    du chemin conv (+2.2 logits en médiane) : la sortie du checkpoint est
    le score BCE pos_weighté, PAS une probabilité.
  * Preuve (partie 2) : p_cal = p/(p + w(1−p)) — identité algébrique
    exacte d'inversion du pos_weight, ZÉRO paramètre ajusté — calibre le
    fold val (jamais vu à l'entraînement de la tête) à ECE 0.0048.
    Conséquence sweep v1 : pénalités ×4 (q99) à ×27 (médiane) trop fortes,
    λ_brut {1,4,16} ≈ λ_calibré ~{27,106,426} en médiane / ~{9,35,141}
    au q90. Le λ pré-enregistré « en unités de proba » n'a jamais été
    testé.

Partie 1 (toujours, ~1 s, aucun import du repo) : inventaire du checkpoint
+ verdict NUANCÉ (le biais seul ne tranche pas ; le couple (clé calib,
ECE) tranche).

Partie 2 (--data, depuis la racine du repo) : rejoue la tête sur le fold
val — quantiles/ECE brut vs calibré (global + riichi/silencieux), AUC de
contrôle, pénalité effective λ·p pour l'ancienne grille BRUTE et la
nouvelle échelle CALIBRÉE.

--fix-out : écrit une COPIE corrigée (biais −log(pos_weight) + clé
'calib') SI les preuves montrent une tête non calibrée (ECE fold mesuré,
ou ECE stocké ≥ 0.10, ou --force), puis REJOUE la tête depuis le FICHIER
écrit pour vérifier bout-en-bout. L'original n'est jamais modifié.

Usage :
    py -3.14 audit_danger_head.py
    py -3.14 audit_danger_head.py --data checkpoints\\danger_data
    py -3.14 audit_danger_head.py --data checkpoints\\danger_data --fix-out checkpoints\\danger_head\\danger_head_cal.pt
"""
import argparse
import math
import os

import numpy as np
import torch

CAL_LAMS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)   # échelle v2
OLD_LAMS = (1.0, 4.0, 16.0)                                     # grille v1
QS = (0.10, 0.50, 0.90, 0.99)


def fmt(x, nd=4):
    return '?' if x is None else f'{x:.{nd}f}'


def qline(p):
    return ' '.join(f"q{int(100 * q)}={np.quantile(p, q):.3f}" for q in QS)


def main():
    ap = argparse.ArgumentParser(
        description='Audit du checkpoint danger head (verdict calibration '
                    '+ copie corrigée vérifiée)')
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
                    help='écrit un checkpoint corrigé (copie) si non '
                         'calibré, puis le vérifie en rejouant le fold')
    ap.add_argument('--force', action='store_true',
                    help='forcer --fix-out sans preuve partie 2 '
                         '(déconseillé)')
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
        logit_prior = math.log(prior / (1.0 - prior))
        print(f"[audit] attendu si CORRIGÉ après SATURATION du biais : "
              f"sigma ~ {prior:.4f} (logit {logit_prior:+.2f})")
        if pos_w is not None:
            p_opt = pos_w * prior / (pos_w * prior + 1.0 - prior)
            print(f"[audit] attendu si NON CORRIGÉ (biais saturé)      : "
                  f"sigma ~ {p_opt:.4f} (logit "
                  f"{math.log(p_opt / (1 - p_opt)):+.2f})")
        print(f"[audit] attendu si NON CORRIGÉ (biais = INIT court)   : "
              f"sigma ~ {prior:.4f}  <- MÊME VALEUR : le test biais seul "
              f"est NON DISCRIMINANT")
        if calib:
            verdict = f"CORRIGÉ (calib embarquée : {calib.get('method')})"
        elif abs(bias - logit_prior) < 1.0:
            verdict = ("NON CORRIGÉ (probable) : biais = valeur d'INIT "
                       "logit(prior), clé 'calib' absente ; en <= 6 époques "
                       "d'AdamW le biais ne peut pas absorber +log(w), "
                       "l'inflation vit dans les poids -> trancher par "
                       "--data (ECE brut vs calibré)")
        elif pos_w is not None and abs(sig - p_opt) < abs(sig - prior):
            verdict = "NON CORRIGÉ (biais ~ optimum pos_weighté)"
        else:
            verdict = "INCERTAIN (biais atypique) -- lancer la partie 2"
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
        if (not calib and met.get('ece') is not None
                and met['ece'] >= 0.10):
            print(f"[audit] indice : ECE stocké {met['ece']:.4f} — une tête "
                  f"calibrée serait <= ~0.02 ; ~0.28 = score brut "
                  f"pos_weighté")
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

    # ------------------- partie 2 : rejouer le fold val ------------------ #
    # (avant --fix-out : c'est la mesure qui autorise ou non la correction)
    p2 = None
    if not args.data:
        print("[audit] (partie 2 : ajouter --data checkpoints\\danger_data "
              "pour rejouer le fold val)")
    else:
        try:
            from mahjong_ai.rl.train_danger_head import (
                DangerNet, load_shard, scan_shards, N_FOLDS, ece_bins,
                auc_rank, I_OPPRIICHI, W_OPPRIICHI)
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

        def replay(head_sd):
            """Rejoue la tête sur le fold val -> p, y, riichi (par tuile
            légale) + n_states. Sous-échantillonnage déterministe
            (RandomState(0) par shard) : deux passes = mêmes états."""
            p_all, y_all, r_all = [], [], []
            n_seen = 0
            for f in files:
                z = load_shard(f)
                idx = np.flatnonzero((z['game'] % N_FOLDS) == fold)
                if len(idx) == 0:
                    continue
                if len(idx) + n_seen > args.max_states:
                    rng = np.random.RandomState(0)
                    idx = rng.choice(idx, args.max_states - n_seen,
                                     replace=False)
                obs, mask, danger = (z['obs_normal'], z['mask'],
                                     z['danger'])
                r_states = (obs[:, I_OPPRIICHI:I_OPPRIICHI + W_OPPRIICHI, :]
                            .max(axis=(1, 2)) > 0)
                for s in range(0, len(idx), 512):
                    j = idx[s:s + 512]
                    with torch.no_grad():
                        logits = model.danger_logits(obs[j])
                    pj = torch.sigmoid(logits).cpu().numpy().astype('float64')
                    mb = mask[j] > 0
                    p_all.append(pj[mb])
                    y_all.append(danger[j][mb])
                    r_all.append(np.repeat(r_states[j], mb.sum(axis=1)))
                n_seen += len(idx)
                if n_seen >= args.max_states:
                    break
            return (np.concatenate(p_all),
                    np.concatenate(y_all).astype('float64'),
                    np.concatenate(r_all), n_seen)

        p, y, r, n_states = replay(ck['head'])
        p_cal = p / (p + pos_w * (1.0 - p))     # identité exacte : z - log(w)
        ece_raw, ece_cal = ece_bins(p, y), ece_bins(p_cal, y)
        print(f"[audit] fold val {fold} : {n_states} états, "
              f"{len(p)} tuiles légales | taux danger réel "
              f"{100 * y.mean():.3f}%")
        for name, m in (('riichi', r), ('silencieux', ~r)):
            if m.any():
                print(f"[audit]   {name:9s} : {int(m.sum()):6d} tuiles | "
                      f"danger {100 * y[m].mean():.2f}% | ECE brut "
                      f"{ece_bins(p[m], y[m]):.4f} / calibré "
                      f"{ece_bins(p_cal[m], y[m]):.4f}")
        print(f"[audit] quantiles p BRUT   (celui du sweep v1)  : {qline(p)}")
        print(f"[audit] quantiles CALIBRÉ (identité −log(w))    : "
              f"{qline(p_cal)}")
        print(f"[audit] ECE brut {ece_raw:.4f} vs calibré {ece_cal:.4f} | "
              f"AUC {auc_rank(y, p):.4f} (stocké {fmt(met.get('auc'))})")
        q50b, q90b, q99b = (float(np.quantile(p, q)) for q in (0.5, 0.9, 0.99))
        q50c, q90c, q99c = (float(np.quantile(p_cal, q))
                            for q in (0.5, 0.9, 0.99))
        print(f"[audit] inflation du p du sweep v1 : ×{q50b / q50c:.0f} "
              f"(médiane), ×{q90b / q90c:.0f} (q90), ×{q99b / q99c:.0f} "
              f"(q99) — odds ×{pos_w:.1f} = pos_weight")
        print(f"[audit] équivalence grille v1 : λ_brut 1/4/16 ≈ λ_calibré "
              f"{q50b / q50c:.0f}/{4 * q50b / q50c:.0f}/"
              f"{16 * q50b / q50c:.0f} (médiane), "
              f"{q90b / q90c:.0f}/{4 * q90b / q90c:.0f}/"
              f"{16 * q90b / q90c:.0f} (q90)")
        print()
        print("[audit] pénalité EFFECTIVE appliquée au score (λ·p, logits) :")
        print(f"{'λ':>5s} {'BRUT méd':>9s} {'BRUT q90':>9s} "
              f"{'CAL méd':>8s} {'CAL q90':>8s}")
        for lam in CAL_LAMS:
            if lam in OLD_LAMS:
                b = f"{lam * q50b:9.3f} {lam * q90b:9.3f}"
            else:
                b = "         —         —"
            print(f"{lam:5g} {b} {lam * q50c:8.3f} {lam * q90c:8.3f}")
        p2 = dict(p=p, y=y, p_cal=p_cal, ece_raw=ece_raw, ece_cal=ece_cal,
                  n_states=n_states)
        print()
        print(f"[audit] lecture : le sweep v1 a blendé le score BCE "
              f"pos_weighté (odds ×{pos_w:.1f}), pas une proba — pénalités "
              f"×4 à ×27 trop fortes ; la grille pré-enregistrée n'a jamais "
              f"été testée. La pénalité calibrée est aussi plus PROGRESSIVE "
              f"(q90/médiane ×{q90c / q50c:.0f} vs ×{q90b / q50b:.0f} en "
              f"brut) : elle frappe les tuiles réellement dangereuses plus "
              f"dur pour une même charge sur la tuile médiane.")
        print('=' * 78)

    # ------------------- copie corrigée (optionnel) ---------------------- #
    if not args.fix_out:
        print("[audit] (copie corrigée : ajouter --fix-out "
              "checkpoints\\danger_head\\danger_head_cal.pt)")
        return
    if calib:
        print("[audit] --fix-out ignoré : le checkpoint est déjà corrigé "
              f"({calib.get('method')})")
        return
    if pos_w is None:
        raise SystemExit("[audit] --fix-out impossible : pos_weight inconnu "
                         "(le passer via --pos-weight, cf. console étape 2)")
    proof = None
    if p2 is not None and p2['ece_raw'] > 2.0 * max(p2['ece_cal'], 1e-6):
        proof = (f"ECE fold val mesuré : brut {p2['ece_raw']:.4f} >> "
                 f"calibré {p2['ece_cal']:.4f}")
    if proof is None and p2 is None and met.get('ece') is not None \
            and met['ece'] >= 0.10:
        proof = f"ECE stocké {met['ece']:.4f} (>= 0.10 = score brut), sans partie 2"
    if proof is None and args.force:
        proof = "--force (sans preuve mesurée)"
    if proof is None:
        print("[audit] --fix-out REFUSÉ : preuves insuffisantes. Lancer avec "
              "--data pour mesurer l'ECE brut vs calibré (ou --force).")
        return
    print(f"[audit] --fix-out : preuve retenue -> {proof}")

    sd = {k: v.clone() for k, v in ck['head'].items()}
    sd['conv2.bias'] = sd['conv2.bias'] - math.log(pos_w)
    ck2 = dict(ck)
    ck2['head'] = sd
    ck2['calib'] = {
        'method': 'prior_correction_pos_weight_a_posteriori',
        'logit_shift': -float(math.log(pos_w)),
        'source_checkpoint': os.path.abspath(args.head),
        'pos_weight': float(pos_w),
    }
    if p2 is not None:
        ck2['calib'].update({'ece_fold_val_raw': float(p2['ece_raw']),
                             'ece_fold_val_cal': float(p2['ece_cal']),
                             'n_val_states': int(p2['n_states'])})
        m2 = dict(met)
        if 'ece' in m2:
            m2['ece_raw_pre_calib'] = m2['ece']      # provenance
        m2['ece'] = float(p2['ece_cal'])             # ce que la tête sort maintenant
        ck2['metrics'] = m2
    torch.save(ck2, args.fix_out)
    new_bias = float(sd['conv2.bias'].flatten()[0])
    print(f"[audit] copie corrigée écrite : {args.fix_out}")
    print(f"[audit]   biais {bias:+.3f} -> {new_bias:+.3f} "
          f"(shift −log({pos_w:.1f}) = {math.log(pos_w):.3f}) | "
          f"sigma {sig:.4f} -> "
          f"{1 / (1 + math.exp(-new_bias)):.5f}")

    # vérification bout-en-bout : recharger le FICHIER et rejouer le fold
    if p2 is not None:
        ck3 = torch.load(args.fix_out, map_location='cpu',
                         weights_only=False)
        if 'calib' not in ck3:
            raise SystemExit("[audit] vérification : clé 'calib' absente du "
                             "fichier écrit !")
        model.danger.load_state_dict(ck3['head'])
        model.eval()
        p_fix, y_fix, _, _ = replay(ck3['head'])
        dmax = float(np.max(np.abs(p_fix - p2['p_cal'])))
        ece_fix = ece_bins(p_fix, y_fix)
        print(f"[audit] vérification (tête rejouée DEPUIS le fichier) : "
              f"ECE {ece_fix:.4f} (attendu {p2['ece_cal']:.4f}) | "
              f"max|Δp| vs identité analytique {dmax:.1e}")
        print(f"[audit]   quantiles mesurés sur la copie : {qline(p_fix)}")
        if not (dmax < 1e-3 and abs(ece_fix - p2['ece_cal']) < 5e-3):
            raise SystemExit("[audit] vérification ÉCHOUÉE -- ne pas "
                             "déployer cette copie")
    print()
    print("[audit] sweep v2 sur la copie corrigée (seed 777 = mêmes donnes "
          "appariées que v1) :")
    print("  v2a pré-enregistré : py -3.14 -m mahjong_ai.rl.eval_danger_blend "
          f"--danger-head {args.fix_out} ^")
    print("        --lams 0.5,1,2,4 --taus 1,2,4 --games 96 "
          "--out checkpoints\\danger_blend_sweep_v2a")
    print("  v2b ponts exploratoires (τ=1 seul : τ est inert à petit λ) :")
    print("        py -3.14 -m mahjong_ai.rl.eval_danger_blend "
          f"--danger-head {args.fix_out} ^")
    print("        --lams 8,16,32,64 --taus 1 --games 96 "
          "--out checkpoints\\danger_blend_sweep_v2b")
    print("  (λ_cal 32 ≈ ancien λ_brut 1 en médiane ; λ_cal 64 frappe le q90 "
          "plus fort que l'ancien λ_brut 4 tout en épargnant la tuile "
          "médiane)")


if __name__ == '__main__':
    main()
