"""[PLAN DANGER - POST-BLEND V2] Diagnostic des deal-ins de la baseline.

Questions PRE-ENREGISTRÉES avant lancement (les réponses décident du
go/no-go RL et du coefficient du reward) : le dng% résiduel de la
baseline (1.27% à 96 donnes, 1.37% à 288, sweeps v2) se décompose-t-il
en

  (A) HEAD-BLIND : la tuile fatale avait une p_cal BASSE (< --p-blind)
      -> ni le blend ni un RL alimenté par CETTE tête ne peuvent voir
      ces dangers ; le levier est la tête (features, dump), pas la
      policy ;
  (B) HAND-LOCKED : p_cal haute mais AUCUNE alternative légale sûre à
      prix abordable (pas de quasi-égalité sûre dans le rayon τ du
      top-1 SL) -> la sécurité coûterait la main MAINTENANT ; c'est le
      travail d'une policy qui planifie (RL : folder/tenir plus tôt) ;
  (C) BLENDABLE-MANQUÉ : p_cal haute ET quasi-égalité sûre existante ->
      le blend aurait pu l'éviter ; le contre-factuel (τ, λ) offline
      dit pourquoi il ne l'a pas fait (sélectivité, trajectoire gelée).

Plus le coût moyen c_hat d'un deal-in en unités reward (score de manche
/1000), pour caler la pénalité κ·p_cal·c_hat du reward RL (le reward
facture déjà le deal-in quand il arrive ; κ=1 = shaping actuariel,
κ>1 = internalisation du risque de rang).

Méthode : rejouer la baseline EXACTE des sweeps v2 (PPOAgent argmax,
blend OFF, lookahead ON, adversaires SL argmax, siège aléatoire par
RandomState(seed+7777)) sur les MÊMES donnes (seed 777, chunks de 32 ->
clés (chunk, env) identiques au per_game de results.json), avec
l'instrumentation [DIAG DEAL-INS] de rollout.py. À CHAQUE défausse de
l'agent : logits SL (34), p_cal tête danger (34), masque, label danger
CONTREFACTUEL (34, calculé des mains cachées par l'env), riichi adverse.
À la fin de chaque manche : si l'agent a payé un ron, sa dernière
défausse de la manche est marquée dealt_in avec les points payés.

Contrôles d'intégrité embarqués :
  * dealt_in ⟹ danger[action] > 0 [v2/v3 : le marquage EXIGE que la
    dernière défausse de la manche soit lbl-danger ET que le ron ne
    soit pas un chankan (clé is_chankan de l'env) ; diag v1 : 11/72
    "deal-ins" étaient des CHANKANS -- rons adverses sur le
    shouminkan de l'agent, comptés à tort sur la dernière défausse
    (points -2.8/-4.9/-6.8 = mains réelles, pas des batons -1.0) ->
    comptés chankan_paid à part] ; réciproque FAUSSE :
    danger[action] sans ron = adversaire tenpai non-furiten mais
    YAKULESS (le label contrefactuel ignore le yaku par construction,
    cf. _danger_labels) -> tuile lbl non payable ;
  * --check-sweep : n_rd / n_dd / rang par donne rejoués vs le bras
    baseline d'un results.json de sweep v2 -> le rejouage doit être
    EXACT (même seed, mêmes graines d'envs, mêmes actions argmax).

Usage (racine du repo ; ~1.3x un bras baseline : ~12 min pour 96 donnes,
~36 min pour 288 sur RTX 2060) :
    py -3.14 -m mahjong_ai.rl.diag_dealins ^
      --danger-head checkpoints/danger_head/danger_head_cal.pt ^
      --games 96 --out checkpoints/danger_diag ^
      --check-sweep checkpoints/danger_blend_sweep_v2b/results.json

Sorties : <out>/diag_dealins.npz (toutes les décisions) +
<out>/diag_summary.json + rapport console. Les p sont CELLES de la tête
calibrée (le checkpoint doit embarquer la clé 'calib' ; pas
d'échappatoire ici : tout le diagnostic lit des probabilités).
Post-analyse d'un npz EXISTANT (ciblage tête v2, sans GPU, 5 s) :
    py -3.14 analyze_danger_diag.py --npz <out>\\diag_dealins.npz
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner
from mahjong_ai.rl.train_danger_head import (
    DangerHead, auc_rank, ece_bins,
)
from mahjong_ai.envs.suphx_features import CHANNEL_GROUPS


def _channel_span(name):
    """(index, largeur) d'un groupe de canaux dans obs_normal."""
    idx = 0
    for gname, n in CHANNEL_GROUPS:
        if gname == name:
            return idx, n
        idx += n
    raise KeyError(name)


I_OPPRIICHI, W_OPPRIICHI = _channel_span('opp_riichi')


class DiagDealinsAgent(PPOAgent):
    """PPOAgent + tête danger CALIBRÉE en lecteur passif.

    Les actions viennent INTÉGRALEMENT de super().act_rl (chemin
    déterministe inchangé -> trajectoires bit-exactes avec le bras
    baseline des sweeps). L'instrumentation ajoute deux forwards par
    batch rl (policy sur oracle muet pour les logits de quasi-égalité,
    tronc+tête pour p_cal) et rend une ligne par requête via pop_diag ;
    elle ne consomme AUCUN RNG (mode eval, aucun sampling).
    """

    def __init__(self, danger_head_path, **kwargs):
        super().__init__(**kwargs)
        ck = torch.load(danger_head_path, map_location=self.device,
                        weights_only=False)
        if not isinstance(ck, dict) or 'head' not in ck:
            raise SystemExit(
                f"[diag] {danger_head_path} n'est pas un checkpoint "
                f"danger_head (clé 'head' absente)")
        cfg = ck.get('config', {})
        for k, mine in (('channels', self.arch['channels']),
                        ('n_blocks', self.arch['n_blocks'])):
            if cfg.get(k) != mine:
                raise SystemExit(
                    f"[diag] config danger_head {k}={cfg.get(k)} != SL "
                    f"({mine}) -- couple checkpoint SL / tête danger "
                    f"incohérent")
        calib = ck.get('calib')
        if calib is None:
            raise SystemExit(
                f"[diag] {danger_head_path} n'embarque pas de clé 'calib' "
                f" : tout ce diagnostic lit des PROBABILITÉS (seuils "
                f"--p-blind, contre-factuel λ·p, c_hat). Produire la copie "
                f"calibrée :\n    py -3.14 audit_danger_head.py --data "
                f"checkpoints\\danger_data --fix-out "
                f"checkpoints\\danger_head\\danger_head_cal.pt")
        self.danger_head = DangerHead(self.arch['channels'],
                                      cfg.get('hidden', 128)).to(self.device)
        self.danger_head.load_state_dict(ck['head'])
        self.danger_head.eval()
        for q in self.danger_head.parameters():
            q.requires_grad = False
        self.diag_armed = False
        self._diag_rows = None
        met = ck.get('metrics', {})
        print(f"[diag] tête danger chargée (AUC val {met.get('auc', '?')}, "
              f"ECE {met.get('ece', '?')}, calib "
              f"{calib.get('method', '?')})", flush=True)

    def pop_diag(self):
        rows, self._diag_rows = self._diag_rows, None
        return rows

    @torch.no_grad()
    def act_rl(self, obs_n, obs_o, masks, gamma_t, deterministic=False):
        out = super().act_rl(obs_n, obs_o, masks, gamma_t,
                             deterministic=deterministic)
        if not (self.diag_armed and deterministic):
            return out
        on = torch.as_tensor(obs_n, dtype=torch.float32,
                             device=self.device)
        oo = torch.as_tensor(obs_o, dtype=torch.float32,
                             device=self.device)
        # logits du chemin déterministe : oracle muet (gamma_override=0
        # à l'éval -> x = [on, 0], exactement l'entrée de super().act_rl)
        x = torch.cat([on, torch.zeros_like(oo)], dim=1)
        logits = self.policy(x)                            # (B, 34)
        feats = self.sl_discard.trunk(on.unsqueeze(-1))    # (B, C, 34, 1)
        p = torch.sigmoid(self.danger_head(feats))         # (B, 34) calibrée
        riichi = (on[:, I_OPPRIICHI:I_OPPRIICHI + W_OPPRIICHI, :]
                  .amax(dim=(1, 2)) > 0)                   # (B,)
        L = logits.float().cpu().numpy()
        P = p.float().cpu().numpy()
        R = riichi.cpu().numpy()
        self._diag_rows = [
            # float32 (pas float16) : la résolution du f16 (~5e-3 sur des
            # logits ~5) inverse des argmax de quasi-égalités et casse
            # l'auto-consistance du contre-factuel λ=0 (0 swap attendu)
            {'logits': L[i].copy(),
             'p': P[i].copy(),
             'riichi': bool(R[i])}
            for i in range(L.shape[0])]
        return out


# ------------------------------------------------------------------ #
# Rejouage baseline (miroir exact de eval_danger_blend.run_arm)
# ------------------------------------------------------------------ #
def run_diag(agent, games, chunk, seed, device):
    """Joue `games` donnes en chunks appariés (clés (chunk, env_idx)
    identiques aux sweeps) et collecte l'instrumentation diag."""
    agent.diag_armed = True
    per_game, diag_rows, diag_rounds = {}, [], []
    t0 = time.time()
    n_chunks = (games + chunk - 1) // chunk
    for c in range(n_chunks):
        n_here = min(chunk, games - c * chunk)
        runner = VecRunner(agent, n_envs=n_here, seed=seed + c * chunk,
                           grp_path=None, compute_lookahead=True,
                           danger_labels=True, mode='opponent',
                           device=device)
        _, stats = runner.collect_batch(
            n_here, gamma_t=0.0, deterministic=True, collect=False,
            gamma_override=0.0)
        for s in stats:
            per_game[(c, int(s['env_idx']))] = s
        for g in runner.diag_games:
            for row in g['rows']:
                row['chunk'] = c
                row['env_idx'] = g['env_idx']
                row['game'] = g['game']
            diag_rows.extend(g['rows'])
            for r in g['rounds']:
                r['chunk'] = c
                r['env_idx'] = g['env_idx']
            diag_rounds.extend(g['rounds'])
        print(f"[diag] {min((c + 1) * chunk, games)}/{games} donnes "
              f"({time.time() - t0:.0f}s, {len(diag_rows)} défausses "
              f"agent)", flush=True)
    if len(per_game) != games:
        raise SystemExit(f"[diag] {len(per_game)} stats != {games} "
                         f"parties (plomberie env_idx ?)")
    return per_game, diag_rows, diag_rounds


# ------------------------------------------------------------------ #
# Assemblage + analyse (numpy pur)
# ------------------------------------------------------------------ #
def assemble(rows):
    N = len(rows)
    L = np.stack([r['logits'] for r in rows]).astype(np.float32)
    P = np.stack([r['p'] for r in rows]).astype(np.float32)
    M = np.stack([r['mask'] for r in rows]).astype(bool)
    D = np.stack([r['danger'] for r in rows]).astype(bool)
    idx = np.arange(N)
    A = np.array([r['action'] for r in rows], dtype=np.int64)
    return {
        'N': N, 'L': L, 'P': P, 'M': M, 'D': D, 'A': A,
        'riichi': np.array([r['riichi'] for r in rows], dtype=bool),
        'dealt_in': np.array([r.get('dealt_in', False) for r in rows],
                             dtype=bool),
        'points': np.array([r.get('points_paid', 0.0) for r in rows],
                           dtype=np.float64),
        # [DIAG v2] paiements NON imputables à une défausse (baton
        # riichi perdu, honba) : comptés à part, PAS des deal-ins
        'stick_lost': np.array([r.get('stick_lost', False) for r in rows],
                               dtype=bool),
        'stick_pts': np.array([r.get('stick_points', 0.0) for r in rows],
                              dtype=np.float64),
        # [DIAG v3] chankans payés (ron adverse sur MON shouminkan,
        # clé is_chankan de l'env) : pertes réelles, imputables à la
        # décision KONG, pas à une défausse
        'chankan_paid': np.array([r.get('chankan_paid', False) for r in rows],
                                 dtype=bool),
        'chankan_pts': np.array([r.get('chankan_points', 0.0) for r in rows],
                                dtype=np.float64),
        'chunk': np.array([r['chunk'] for r in rows], dtype=np.int64),
        'env': np.array([r['env_idx'] for r in rows], dtype=np.int64),
        'round': np.array([r['round'] for r in rows], dtype=np.int64),
        'top1': L[idx, A],          # baseline = argmax -> top1 = L[A]
        'p_ch': P[idx, A],
        'lbl_dd': D[idx, A],
    }


def counterfactual_blend(dat, tau, lam):
    """Contre-factuel du blend score = L − λ·p sur les quasi-égalités
    (même formule que eval_danger_blend, trajectoire gelée). Retourne
    (a_cf, swap, save, create)."""
    L, P, M, D, A = dat['L'], dat['P'], dat['M'], dat['D'], dat['A']
    idx = np.arange(dat['N'])
    near = M & ((dat['top1'][:, None] - L) <= tau)
    score = np.where(near, L - lam * P, -np.inf)
    a_cf = score.argmax(axis=1)
    swap = a_cf != A
    save = swap & dat['lbl_dd'] & ~D[idx, a_cf]      # deal-in réel évité
    create = swap & ~dat['lbl_dd'] & D[idx, a_cf]    # deal-in créé
    return a_cf, swap, save, create


def analyze(dat, taus, lams, p_blind, out_json=None):
    N, A, D, M = dat['N'], dat['A'], dat['D'], dat['M']
    idx = np.arange(N)
    dd = dat['dealt_in']
    n_dd = int(dd.sum())
    rep = {'n_decisions': N, 'n_dealins': n_dd}

    # --- intégrité [v2] --------------------------------------------- #
    # Deux familles d'écarts de sémantique OPPOSÉE :
    #  * ghost  = dealt_in SANS label : impossible avec le marquage v2
    #    (les batons riichi -> stick_lost) ; 0 attendu ;
    #  * yakuless = label SANS deal-in : l'adversaire tenpai non-furiten
    #    attendait la tuile mais sa main n'a pas de yaku -> pas de ron.
    #    Ces lbl ne sont PAS payables : le dng% des sweeps (lbl-based)
    #    surestime le risque réellement payé.
    paid = dd & dat['lbl_dd']
    ghost = dd & ~dat['lbl_dd']
    yakuless = dat['lbl_dd'] & ~dd
    n_paid = int(paid.sum())
    ckan = dat.get('chankan_paid', np.zeros(N, bool))
    n_ckan = int(ckan.sum())
    rep['n_dealins_paid'] = n_paid
    rep['n_chankan_paid'] = n_ckan
    rep['integrity'] = {
        'dealt_in_sans_label': int(ghost.sum()),
        'label_sans_deal_in_yakuless': int(yakuless.sum()),
        'n_sticks': int(dat['stick_lost'].sum()),
        'stick_points': float(
            dat['stick_pts'][dat['stick_lost']].sum()),
        'n_chankan': n_ckan,
        'chankan_points': float(
            dat['chankan_pts'][ckan].sum()) if 'chankan_pts' in dat
        else 0.0,
    }
    rep['integrity_mismatch'] = int((dat['lbl_dd'] != dd).sum())

    # --- tête danger EN LIVE sur la trajectoire d'éval ---------------- #
    live = {'auc': float(auc_rank(D[M], dat['P'][M])),
            'ece': float(ece_bins(dat['P'][M], D[M].astype(float)))}
    r_mask = dat['riichi']
    for nm, m in (('riichi', r_mask), ('silencieux', ~r_mask)):
        if m.any():
            live[f'auc_{nm}'] = float(
                auc_rank(D[M & m[:, None]], dat['P'][M & m[:, None]]))
    live['pct_riichi_states'] = float(r_mask.mean())
    rep['live_head'] = live

    # --- p_cal de la tuile jouée : globale vs deal-ins ---------------- #
    q = lambda v, ps: ([float(np.percentile(v, p_)) for p_ in ps]
                       if len(v) else [float('nan')] * len(ps))
    rep['p_played'] = {'all': q(dat['p_ch'], (10, 50, 90, 99))}
    if n_paid:
        rep['p_played']['dealins'] = q(dat['p_ch'][paid],
                                       (10, 25, 50, 75, 90))

    # --- c_hat : coût des deal-ins PAYÉS (unités reward=score/1000) -- #
    if n_paid:
        pts = dat['points'][paid]
        rep['c_hat'] = {'n': int(n_paid), 'mean': float(pts.mean()),
                        'median': float(np.median(pts)),
                        'q10': float(np.percentile(pts, 10)),
                        'q90': float(np.percentile(pts, 90)),
                        'total': float(pts.sum())}

    # --- décomposition (A)/(B)/(C) des deal-ins PAYÉS ----------------- #
    if n_paid:
        blind = paid & (dat['p_ch'] < p_blind)
        visible = paid & (dat['p_ch'] >= p_blind)
        w_pts = dat['points']          # 0 hors deal-ins -> pondération OK
        dec = {'p_blind': p_blind, 'n_paid': n_paid,
               'A_head_blind': int(blind.sum()),
               'visible': int(visible.sum())}
        for tau in taus:
            safe_near = (M & ~D
                         & ((dat['top1'][:, None] - dat['L']) <= tau))
            has_safe = safe_near.any(axis=1)
            # alternative sûre légale SANS contrainte de prix :
            has_safe_any = (M & ~D).any(axis=1)
            gap_safe = np.where(M & ~D,
                                dat['top1'][:, None] - dat['L'], np.inf)
            min_gap = gap_safe.min(axis=1)
            dec[f'tau={tau:g}'] = {
                'C_blendable_manque': int((visible & has_safe).sum()),
                'B_hand_locked': int((visible & ~has_safe).sum()),
                'B_avec_sure_lointaine': int(
                    (visible & ~has_safe & has_safe_any).sum()),
                'B_sans_aucune_sure': int(
                    (visible & ~has_safe & ~has_safe_any).sum()),
                'med_gap_vers_sure_visible': (
                    float(np.median(min_gap[visible & ~has_safe]))
                    if (visible & ~has_safe).any() else None),
            }
        dec['points_A'] = float(w_pts[blind].sum())
        dec['points_visible'] = float(w_pts[visible].sum())
        rep['decomposition'] = dec

    # --- contre-factuel blend (τ, λ), trajectoire gelée --------------- #
    cf = {}
    for tau in taus:
        for lam in lams:
            a_cf, swap, save, create = counterfactual_blend(dat, tau, lam)
            save_paid = swap & paid & ~D[idx, a_cf]
            cf[f'tau={tau:g},lam={lam:g}'] = {
                'n_swaps': int(swap.sum()),
                'swap_pct': float(swap.mean()),
                'dealins_evites': int(save.sum()),
                # [v2] 'payés évités' + économie réelle : les lbl évités
                # incluent une part yakuless non payable ; seuls les
                # deal-ins PAYÉS évités économisent des points
                'dealins_payes_evites': int(save_paid.sum()),
                'points_economises': float(
                    -dat['points'][save_paid].sum()),
                'dealins_crees': int(create.sum()),
                'net': int(save.sum() - create.sum()),
                'dealins_residuels': int((paid & ~save).sum()),
            }
    rep['counterfactual'] = cf
    if out_json:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
    return rep


def print_report(rep, taus, lams):
    N, n_dd = rep['n_decisions'], rep['n_dealins']
    n_paid = rep.get('n_dealins_paid', n_dd)
    integ = rep.get('integrity', {})
    print()
    print(f"[diag] {N} défausses agent | {n_paid} deal-ins PAYÉS "
          f"({100 * n_paid / max(N, 1):.2f}%)", flush=True)
    n_gh = int(integ.get('dealt_in_sans_label', 0))
    n_stick = max(n_gh, int(integ.get('n_sticks', 0)))
    n_ck = int(integ.get('n_chankan', 0))
    if n_ck:
        print(f"[diag] CHANKANS payés (ron adverse sur MON shouminkan) : "
              f"{n_ck} manches, {integ.get('chankan_points', 0.0):.1f} pts "
              f"-- pertes réelles imputables à la décision KONG, pas aux "
              f"défausses (le diag v1 les comptait comme deal-ins de la "
              f"dernière défausse : 11/72)", flush=True)
    if n_stick:
        print(f"[diag] paiements NON imputables restants (baton riichi "
              f"perdu / honba) : {n_stick} manches, "
              f"{integ.get('stick_points', 0.0):.1f} pts", flush=True)
    print(f"[diag] intégrité : dealt_in sans label : {n_gh}"
          + (" (ATTENDU 0 avec le marquage v2 !)" if n_gh
             else " (ok)")
          + " | label sans ron (yakuless, le label ignore le yaku) : "
          f"{int(integ.get('label_sans_deal_in_yakuless', 0))}",
          flush=True)
    lv = rep['live_head']
    print(f"[diag] tête EN LIVE sur la trajectoire : AUC {lv['auc']:.4f} "
          f"| ECE {lv['ece']:.4f} | riichi {lv.get('auc_riichi', float('nan')):.4f}"
          f" / silencieux {lv.get('auc_silencieux', float('nan')):.4f} "
          f"({100 * lv['pct_riichi_states']:.0f}% états riichi)", flush=True)
    pa = rep['p_played']['all']
    print(f"[diag] p_cal de la tuile jouée (toutes) : q10={pa[0]:.4f} "
          f"q50={pa[1]:.4f} q90={pa[2]:.4f} q99={pa[3]:.4f}", flush=True)
    if 'dealins' in rep['p_played']:
        pd_ = rep['p_played']['dealins']
        print(f"[diag] p_cal de la tuile FATALE (deal-ins PAYÉS) : "
              f"q10={pd_[0]:.4f} q25={pd_[1]:.4f} q50={pd_[2]:.4f} "
              f"q75={pd_[3]:.4f} q90={pd_[4]:.4f}", flush=True)
    if 'c_hat' in rep:
        c = rep['c_hat']
        print(f"[diag] ĉ (coût deal-in, unités reward=score/1000) : "
              f"moyen {c['mean']:.3f} | médian {c['median']:.3f} | "
              f"q10 {c['q10']:.3f} q90 {c['q90']:.3f} | total "
              f"{c['total']:.1f}", flush=True)
    if 'decomposition' in rep:
        d = rep['decomposition']
        n_paid_d = d.get('n_paid', n_dd)
        # [FIX v2] les points payés sont NÉGATIFS (pertes, unités
        # reward) : l'ancien max(tot_pts, 1e-9) retournait TOUJOURS
        # l'epsilon -> pourcentages -1.6e13 (diag v1, 96 donnes)
        tot_pts = abs(d['points_A']) + abs(d['points_visible'])
        print()
        print(f"-- décomposition des {n_paid_d} deal-ins PAYÉS "
              f"(p_blind={d['p_blind']}) --", flush=True)
        print(f"  (A) head-blind  : {d['A_head_blind']:4d} "
              f"({100 * d['A_head_blind'] / max(n_paid_d, 1):4.1f}%) | "
              f"{100 * abs(d['points_A']) / max(tot_pts, 1e-9):4.1f}% "
              f"des points payés", flush=True)
        print(f"  visible (p≥{d['p_blind']}) : {d['visible']:4d} "
              f"({100 * d['visible'] / max(n_paid_d, 1):4.1f}%) | "
              f"{100 * abs(d['points_visible']) / max(tot_pts, 1e-9):4.1f}% "
              f"des points", flush=True)
        for tau in taus:
            k = f'tau={tau:g}'
            t = d[k]
            print(f"  tau={tau:g} : (C) blendable-manqué "
                  f"{t['C_blendable_manque']:4d} | (B) hand-locked "
                  f"{t['B_hand_locked']:4d} dont {t['B_sans_aucune_sure']}"
                  f" sans AUCUNE tuile sûre légale, "
                  f"{t['B_avec_sure_lointaine']} avec sûre lointaine "
                  f"(gap méd {t['med_gap_vers_sure_visible']:.1f} logits)"
                  if t['med_gap_vers_sure_visible'] is not None else
                  f"  tau={tau:g} : (C) {t['C_blendable_manque']:4d} | "
                  f"(B) {t['B_hand_locked']:4d}", flush=True)
    print()
    print("-- contre-factuel blend (τ, λ) sur trajectoire gelée "
          "(1er ordre : les swaps ne re-dynamisent PAS la partie) --",
          flush=True)
    print(f"  {'tau':>4s} {'lam':>5s} {'swaps%':>7s} {'évités':>7s} "
          f"{'payés':>6s} {'pts éco':>8s} {'créés':>7s} {'net':>5s} "
          f"{'dd résid':>8s}", flush=True)
    for tau in taus:
        for lam in lams:
            c = rep['counterfactual'][f'tau={tau:g},lam={lam:g}']
            print(f"  {tau:4g} {lam:5g} {100 * c['swap_pct']:7.2f} "
                  f"{c['dealins_evites']:7d} "
                  f"{c.get('dealins_payes_evites', 0):6d} "
                  f"{c.get('points_economises', 0.0):8.1f} "
                  f"{c['dealins_crees']:7d} "
                  f"{c['net']:+5d} {c['dealins_residuels']:8d}",
                  flush=True)
    print()
    print("Lecture : (A) dominant -> améliorer la tête AVANT tout RL ; "
          "(B) dominant -> RL bien posé (planification/fold), blend "
          "myopique plafonné ;\n(C) dominant -> le blend était la bonne "
          "idée mais mal réglée (voir le contre-factuel pour le (τ, λ) "
          "sélectif), et le RL doit apprendre la sélectivité.",
          flush=True)
    print("v2 : 'évités' compte les lbl (dont une part yakuless NON "
          "payable) ; 'payés' = deal-ins réellement payés évités, "
          "'pts éco' = économie réelle en unités reward (score/1000).",
          flush=True)


def check_sweep(per_game, results_path):
    """Canari : le rejouage doit reproduire EXACTEMENT le bras baseline
    du sweep (mêmes graines -> mêmes donnes -> mêmes stats par donne)."""
    with open(results_path, encoding='utf-8') as f:
        res = json.load(f)
    base = res['arms']['baseline']['per_game']
    n_ok = n_bad = 0
    for (c, e), s in per_game.items():
        k = f'{c}.{e}'
        if k not in base:
            continue
        b = base[k]
        if (int(b['rd']) == int(s['n_rl_discards'])
                and int(b['dd']) == int(s['n_danger_discards'])
                and int(b['rank']) == int(s['rank_of_seat0'])):
            n_ok += 1
        else:
            n_bad += 1
            print(f"[diag] ÉCART donne {k} : rejoué "
                  f"rd={s['n_rl_discards']} dd={s['n_danger_discards']} "
                  f"rank={s['rank_of_seat0']} vs sweep "
                  f"rd={b['rd']} dd={b['dd']} rank={b['rank']}",
                  flush=True)
    print(f"[diag] canari --check-sweep : {n_ok} donnes EXACTES, "
          f"{n_bad} écarts"
          + (" -- REJOUAGE BIT-EXACT, l'analyse porte sur les mêmes "
             "donnes que le sweep" if n_bad == 0 else
             " ! Vérifier seed/chunk/repo avant de lire l'analyse"),
          flush=True)
    return n_bad == 0


def main():
    p = argparse.ArgumentParser(
        description='[PLAN DANGER - POST-BLEND] diagnostic des deal-ins '
                    'de la baseline : head-blind / hand-locked / '
                    'blendable-manqué + coût c_hat (go/no-go RL)')
    p.add_argument('--danger-head',
                   default='checkpoints/danger_head/danger_head_cal.pt',
                   help='checkpoint danger CALIBRÉ (clé calib obligatoire)'
                        ' : danger_head_cal.pt de audit_danger_head.py '
                        '--fix-out')
    p.add_argument('--sl-discard',
                   default='checkpoints/sl_tonpusen_la/discard.pt')
    p.add_argument('--sl-frozen',
                   default='checkpoints/sl_tonpusen_la')
    p.add_argument('--games', type=int, default=96,
                   help='donnes à rejouer (défaut 96 = donnes v2a/v2b ; '
                        '288 = donnes v2c)')
    p.add_argument('--chunk-size', type=int, default=32,
                   help='identique aux sweeps (appariement des donnes)')
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--taus', default='1,2,4',
                   help='τ du contre-factuel et de la décomposition')
    p.add_argument('--lams', default='8,16,32,64',
                   help='λ du contre-factuel (probas calibrées)')
    p.add_argument('--p-blind', type=float, default=0.05,
                   help='seuil head-blind : p_cal de la tuile fatale '
                        'en dessous duquel la tête ne voyait pas le '
                        'danger')
    p.add_argument('--check-sweep', default=None,
                   help='results.json d\'un sweep v2 : le rejouage doit '
                        'reproduire le bras baseline donne par donne')
    p.add_argument('--out', default='checkpoints/danger_diag')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    taus = [float(t) for t in str(args.taus).split(',') if t.strip()]
    lams = [float(l) for l in str(args.lams).split(',') if l.strip()]
    if not taus or not lams:
        raise SystemExit("[diag] --taus/--lams vides")
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = DiagDealinsAgent(
        args.danger_head,
        sl_discard_path=args.sl_discard, sl_frozen_dir=args.sl_frozen,
        device=device)
    chunk = min(args.chunk_size, args.games)
    print(f"[diag] rejouage baseline : {args.games} donnes (chunks de "
          f"{chunk}, seed {args.seed}, lookahead ON) | questions "
          f"pré-enregistrées : (A) head-blind p<{args.p_blind} / "
          f"(B) hand-locked / (C) blendable-manqué + c_hat", flush=True)

    per_game, rows, rounds = run_diag(agent, args.games, chunk,
                                      args.seed, device)
    if not rows:
        raise SystemExit("[diag] aucune défausse instrumentée -- "
                         "danger_labels inactif ?")
    if args.check_sweep:
        check_sweep(per_game, args.check_sweep)

    dat = assemble(rows)
    # intégrité vs stats du runner (le runner compte les mêmes événements)
    n_rd_stats = sum(int(s['n_rl_discards']) for s in per_game.values())
    n_dd_stats = sum(int(s['n_danger_discards'])
                     for s in per_game.values())
    if dat['N'] != n_rd_stats or int(dat['lbl_dd'].sum()) != n_dd_stats:
        raise SystemExit(
            f"[diag] désalignement instrumentation : lignes {dat['N']} "
            f"vs n_rd {n_rd_stats}, lbl_dd "
            f"{int(dat['lbl_dd'].sum())} vs n_dd {n_dd_stats}")

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, 'diag_dealins.npz')
    np.savez_compressed(
        npz_path,
        logits=dat['L'], p=dat['P'],
        mask=dat['M'], danger=dat['D'].astype(np.uint8),
        action=dat['A'].astype(np.int16),
        riichi=dat['riichi'], dealt_in=dat['dealt_in'],
        points=dat['points'].astype(np.float32),
        stick_lost=dat['stick_lost'],
        stick_pts=dat['stick_pts'].astype(np.float32),
        chankan_paid=dat['chankan_paid'],
        chankan_pts=dat['chankan_pts'].astype(np.float32),
        chunk=dat['chunk'], env=dat['env'], round=dat['round'])
    summary_path = os.path.join(args.out, 'diag_summary.json')
    rep = analyze(dat, taus, lams, args.p_blind, out_json=summary_path)
    rep['rounds'] = rounds
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print_report(rep, taus, lams)
    print()
    print(f"[diag] sorties : {npz_path} | {summary_path}", flush=True)


if __name__ == '__main__':
    main()
