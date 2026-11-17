"""[PLAN DANGER - ÉTAPE 3] Sweep du blend danger sur les quasi-égalités.

Protocole pré-enregistré (avant l'entraînement de l'étape 2) :
  * blend : score = logit_policy − λ·p_danger, appliqué UNIQUEMENT aux
    tuiles légales dans un rayon τ du top-1 des logits (quasi-égalités).
    Hors quasi-égalité l'argmax SL est inchangé : le dommage éventuel
    au niveau de jeu est borné par construction.
  * p_danger = σ(tête danger(tronc SL figé(obs 279 canaux))) — la tête
    doit être CALIBRÉE (clé 'calib' embarquée dans le checkpoint).
    POST-MORTEM sweep v1 : le checkpoint entraîné n'embarquait PAS le
    décalage −log(pos_weight) (ECE fold val 0.280 brut vs 0.005
    calibré) — le v1 a blendé le score BCE pos_weighté, pénalités ×4
    (q99) à ×27 (médiane) trop fortes ; la grille λ pré-enregistrée « en
    unités de proba » n'a jamais été testée. Le sweep REFUSE donc une
    tête sans clé 'calib' (échappatoire --allow-raw-p, déconseillé) :
    passer la copie calibrée danger_head_cal.pt produite par
    audit_danger_head.py --fix-out.
  * sweep (τ, λ) sur l'éval APPARIÉE : 384 parties, seed 777, chunks
    de 32, adversaires 3 × SL argmax, siège agent argmax, lookahead ON
    — les conditions EXACTES de la référence mr = 2.484 ± 0.113.
  * PORTE de déploiement : dng% réduit d'au moins 20% vs baseline ET
    Δmr ≤ +0.05. Échec = échec pré-enregistré, on re-discute (pas de
    rattrapage improvisé).

Le bras 0 (baseline, blend désactivé) réplique EXACTEMENT
eval_matchup.py : même construction d'agent, même runner, même
planning de graines par chunk. C'est l'éval baseline instrumentée qui
restait à lancer — mr doit reproduire 2.484 au bit près (contrôle de
non-régression des edits rollout.py : instrumentation pure) et son
dng% est la référence de la porte.

Fidélité de l'appariement inter-bras : chaque chunk construit des envs
NEUFS aux graines explicites (seed + c*chunk + i) avec UNE partie par
env -> les donnes (murs + sièges agent) sont IDENTIQUES dans tous les
bras ; les stats portent env_idx -> comparaisons appariées par partie
(rangs, dng) + bootstrap apparié.

Usage (racine du repo ; SWEEP V2 = tête calibrée, λ en unités de
proba ; ~36 min/bras à 384p sur RTX 2060, ~9 min à 96p) :
    # 0) une fois : copie calibrée (audit partie 2 + fix-out vérifié)
    py -3.14 audit_danger_head.py --data checkpoints/danger_data ^
      --fix-out checkpoints/danger_head/danger_head_cal.pt

    # v2a - grille pré-enregistrée, ENFIN en probas (13 bras x 96p)
    py -3.14 -m mahjong_ai.rl.eval_danger_blend ^
      --danger-head checkpoints/danger_head/danger_head_cal.pt ^
      --sl-discard checkpoints/sl_tonpusen_la/discard.pt ^
      --sl-frozen checkpoints/sl_tonpusen_la ^
      --grp checkpoints/grp_tonpusen.pt ^
      --lams 0.5,1,2,4 --taus 1,2,4 --games 96 --seed 777 ^
      --out checkpoints/danger_blend_sweep_v2a

    # v2b - ponts vers les forces v1 (λ_cal 32 ≈ ancien λ_brut 1 en
    #        médiane ; τ=1 seul : τ est inerte à petit λ)
    py -3.14 -m mahjong_ai.rl.eval_danger_blend ^
      --danger-head checkpoints/danger_head/danger_head_cal.pt ^
      --lams 8,16,32,64 --taus 1 --games 96 --seed 777 ^
      --out checkpoints/danger_blend_sweep_v2b

    # analyse appariée (aucun GPU requis), puis confirmation du/des
    # gagnants à --games 384 pour la porte pré-enregistrée
    py -3.14 analyze_danger_blend_results.py ^
      --in checkpoints/danger_blend_sweep_v2a/results.json

Le GRP n'altère pas les dynamiques (récompenses seules, inutilisées à
l'éval) mais on le passe identique à la run de référence, par prudence.
Si la porte passe, l'intégration définitive du blend dans l'agent de
déploiement suivra (patch séparé, (τ, λ) gelés).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner
from mahjong_ai.rl.train_danger_head import DangerHead

REF_MR = 2.484        # référence pré-enregistrée (384p, seed 777, argmax,
                      # lookahead ON) : le bras baseline doit la reproduire


class DangerBlendAgent(PPOAgent):
    """PPOAgent + tête danger : argmax SL reranké sur les quasi-égalités.

    score = logit − λ·p sur {tuiles légales à <= τ du top-1} ; argmax du
    score (a* est toujours dans le set : le blend ne peut choisir QUE
    une quasi-égalité légale du SL). p_danger vient du tronc SL figé de
    l'agent (sl_discard.trunk), EXACTEMENT la fonction feature de
    l'entraînement de la tête — indépendant de la policy chargée.

    p doit être une PROBA CALIBRÉE (clé 'calib' du checkpoint danger) :
    le constructeur REFUSE une tête brute (score BCE pos_weighté, le
    bug du sweep v1) sauf allow_raw_p=True.

    Blend actif uniquement en mode déterministe (déploiement argmax) :
    en échantillonné ou blend_active=False, comportement PPOAgent pur.
    """

    def __init__(self, danger_head_path, tau=2.0, lam=4.0,
                 allow_raw_p=False, **kwargs):
        super().__init__(**kwargs)
        ck = torch.load(danger_head_path, map_location=self.device,
                        weights_only=False)
        if not isinstance(ck, dict) or 'head' not in ck:
            raise SystemExit(
                f"[blend] {danger_head_path} n'est pas un checkpoint "
                f"danger_head (clé 'head' absente) -- lancer d'abord "
                f"mahjong_ai.rl.train_danger_head")
        cfg = ck.get('config', {})
        for k, mine in (('channels', self.arch['channels']),
                        ('n_blocks', self.arch['n_blocks'])):
            if cfg.get(k) != mine:
                raise SystemExit(
                    f"[blend] config danger_head {k}={cfg.get(k)} != SL "
                    f"({mine}) -- couple checkpoint SL / tête danger "
                    f"incohérent")
        # [v2] garde calibration : sans clé 'calib', σ(sortie) est le score
        # BCE pos_weighté (odds ×pos_weight), PAS une probabilité — le
        # sweep v1 a blendé ce score (pénalités ×4 à ×27, porte échouée).
        calib = ck.get('calib')
        if calib is None and not allow_raw_p:
            raise SystemExit(
                f"[blend] {danger_head_path} n'embarque pas de clé 'calib' "
                f"(pos_weight={ck.get('pos_weight', '?')}) : σ(sortie de "
                f"tête) est le score BCE pos_weighté, PAS une probabilité "
                f"— c'est exactement le bug du sweep v1 (ECE 0.280 brut vs "
                f"0.005 calibré, pénalités ×4 à ×27 trop fortes). Produire "
                f"la copie calibrée, puis la passer à --danger-head :\n"
                f"    py -3.14 audit_danger_head.py --data "
                f"checkpoints\\danger_data --fix-out "
                f"checkpoints\\danger_head\\danger_head_cal.pt\n"
                f"(forcer ce checkpoint brut malgré tout : --allow-raw-p, "
                f"déconseillé — répète le bug v1)")
        self.danger_head = DangerHead(self.arch['channels'],
                                      cfg.get('hidden', 128)).to(self.device)
        self.danger_head.load_state_dict(ck['head'])
        self.danger_head.eval()
        for p in self.danger_head.parameters():
            p.requires_grad = False
        self.tau = float(tau)
        self.lam = float(lam)
        self.blend_active = False          # armé bras par bras par le sweep
        self.n_blend_decisions = 0
        self.n_blend_swaps = 0
        met = ck.get('metrics', {})
        print(f"[blend] tête danger chargée (AUC val "
              f"{met.get('auc', '?')}, ECE {met.get('ece', '?')}, "
              f"calib embarquée "
              f"{(calib or {}).get('method', 'ABSENTE')})", flush=True)
        if calib is None:
            print("[blend] ATTENTION : p BRUT non calibré (--allow-raw-p) "
                  "— λ·p s'applique en unités du score pos_weighté, pas "
                  "en probabilités (conditions du sweep v1)", flush=True)

    def reset_counters(self):
        self.n_blend_decisions = 0
        self.n_blend_swaps = 0

    @torch.no_grad()
    def act_rl(self, obs_n, obs_o, masks, gamma_t, deterministic=False):
        if (not deterministic or not self.blend_active
                or self.tau <= 0.0 or self.lam <= 0.0):
            return super().act_rl(obs_n, obs_o, masks, gamma_t,
                                  deterministic=deterministic)
        on = torch.as_tensor(obs_n, dtype=torch.float32,
                             device=self.device)
        oo = torch.as_tensor(obs_o, dtype=torch.float32,
                             device=self.device)
        mask_t = torch.as_tensor(masks, dtype=torch.float32,
                                 device=self.device)
        if gamma_t >= 1.0:
            oo_masked = oo
        elif gamma_t <= 0.0:
            oo_masked = torch.zeros_like(oo)   # éval : oracle muet
        else:
            oo_masked = oo * torch.bernoulli(torch.full_like(oo, gamma_t))
        x = torch.cat([on, oo_masked], dim=1)
        logits = self.policy(x)
        masked = logits + (mask_t - 1.0) * 1e9
        a_star = masked.argmax(dim=-1)
        # quasi-égalités : tuiles légales à <= tau du top-1
        top1 = masked.max(dim=-1, keepdim=True).values
        near = (masked > top1 - self.tau) & (mask_t > 0)
        # p_danger calibrée sur le tronc SL figé 279 canaux
        feats = self.sl_discard.trunk(on.unsqueeze(-1))
        p = torch.sigmoid(self.danger_head(feats))
        score = (masked - self.lam * p).masked_fill(~near, -1e9)
        actions = score.argmax(dim=-1)
        # interface inchangée (logp/value inutilisés en éval collect=False)
        logps = torch.log_softmax(masked, dim=-1)\
            .gather(1, actions.unsqueeze(1)).squeeze(1)
        values = self.critic(torch.cat([on, oo], dim=1))
        self.n_blend_decisions += int(mask_t.shape[0])
        self.n_blend_swaps += int((actions != a_star).sum().item())
        return (actions.cpu().numpy().astype(np.int64),
                logps.cpu().numpy().astype(np.float32),
                values.cpu().numpy().astype(np.float32),
                oo_masked.cpu().numpy().astype(np.float16))


# ------------------------------------------------------------------ #
# Bras d'éval (réplique exacte d'eval_matchup.evaluate_chunk)
# ------------------------------------------------------------------ #
def run_arm(label, agent, games, chunk, seed, device, grp_path, quiet=False):
    """Joue `games` parties en chunks appariés (une par env neuf).

    Retourne {(chunk_idx, env_idx): stats} — clé d'appariement stable
    entre bras (mêmes graines -> mêmes donnes)."""
    per_game = {}
    t0 = time.time()
    n_chunks = (games + chunk - 1) // chunk
    for c in range(n_chunks):
        n_here = min(chunk, games - c * chunk)
        runner = VecRunner(agent, n_envs=n_here, seed=seed + c * chunk,
                           grp_path=grp_path, compute_lookahead=True,
                           danger_labels=True, mode='opponent',
                           device=device)
        _, stats = runner.collect_batch(
            n_here, gamma_t=0.0, deterministic=True, collect=False,
            gamma_override=0.0)
        for s in stats:
            per_game[(c, int(s['env_idx']))] = s
        if not quiet:
            done = min((c + 1) * chunk, games)
            print(f"[sweep] {label} : {done}/{games} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if len(per_game) != games:
        raise SystemExit(f"[sweep] {label} : {len(per_game)} stats != "
                         f"{games} parties (plomberie env_idx ?)")
    return per_game


def arm_summary(per_game):
    ranks = np.array([s['rank_of_seat0'] for s in per_game.values()],
                     dtype=np.float64)
    n_rd = sum(int(s['n_rl_discards']) for s in per_game.values())
    n_dd = sum(int(s['n_danger_discards']) for s in per_game.values())
    # [v2d] deal-ins RÉELLEMENT payés (ron sur défausse dangereuse de
    # l'agent) + points payés : le dng% lbl-based inclut une part
    # yakuless NON payable (le label contrefactuel ignore le yaku) ;
    # c'est la métrique économique de la porte
    n_dpid = sum(int(s.get('n_dealins_paid', 0))
                 for s in per_game.values())
    pts_paid = sum(float(s.get('points_paid', 0.0))
                   for s in per_game.values())
    # [v3] chankans payés (ron adverse sur le shouminkan de l'agent) :
    # surveillance -- le blend ne touche pas les kans, mais le futur
    # garde-fou kong (--danger-head sur KONG_YES) aura cette référence
    n_ckan = sum(int(s.get('n_chankan_paid', 0)) for s in per_game.values())
    pts_ckan = sum(float(s.get('points_chankan', 0.0))
                   for s in per_game.values())
    se = (float(ranks.std(ddof=1) / np.sqrt(len(ranks)))
          if len(ranks) > 1 else 0.0)
    return {'mr': float(ranks.mean()), 'mr_ic95': 1.96 * se,
            'wr': float((ranks == 1).mean()), 'n_rd': n_rd, 'n_dd': n_dd,
            'dng_pct': 100.0 * n_dd / max(n_rd, 1),
            'n_dpid': n_dpid, 'paid_pct': 100.0 * n_dpid / max(n_rd, 1),
            'pts_paid': float(pts_paid),
            'n_ckan': n_ckan, 'pts_ckan': float(pts_ckan)}


def paired_compare(base, arm, n_boot=2000, seed=777):
    """Différences APPARIÉES par donne (clé (chunk, env_idx)) + bootstrap
    apparié (IC95) sur Δmr et la réduction relative de dng%."""
    if set(base) != set(arm):
        raise SystemExit(f"[sweep] clés d'appariement divergentes "
                         f"({len(base)} vs {len(arm)})")
    keys = sorted(base)
    rb = np.array([base[k]['rank_of_seat0'] for k in keys], float)
    ra = np.array([arm[k]['rank_of_seat0'] for k in keys], float)
    db = np.array([base[k]['n_danger_discards'] for k in keys], float)
    da = np.array([arm[k]['n_danger_discards'] for k in keys], float)
    nb = np.array([base[k]['n_rl_discards'] for k in keys], float)
    na = np.array([arm[k]['n_rl_discards'] for k in keys], float)
    # [v2d] deal-ins payés + points payés (appariés)
    pb = np.array([base[k].get('n_dealins_paid', 0) for k in keys], float)
    pa = np.array([arm[k].get('n_dealins_paid', 0) for k in keys], float)
    xb = np.array([base[k].get('points_paid', 0.0) for k in keys], float)
    xa = np.array([arm[k].get('points_paid', 0.0) for k in keys], float)
    dng_b = float(db.sum() / max(nb.sum(), 1.0))
    dng_a = float(da.sum() / max(na.sum(), 1.0))
    rel = (dng_b - dng_a) / max(dng_b, 1e-9)
    rng = np.random.RandomState(seed)
    n = len(keys)
    idx = rng.randint(0, n, size=(n_boot, n))
    dmr_b = (ra[idx].mean(axis=1) - rb[idx].mean(axis=1))
    dng_b_b = (da[idx].sum(axis=1)
               / np.maximum(na[idx].sum(axis=1), 1.0))
    dng_b_base = (db[idx].sum(axis=1)
                  / np.maximum(nb[idx].sum(axis=1), 1.0))
    # Δ absolu (points de dng%) : toujours bien défini
    dpts_b = 100.0 * (dng_b_b - dng_b_base)
    # réduction relative : rééchantillons dégénérés (aucun dng baseline)
    # exclus ; s'ils dominent (petit n), l'IC relatif est marqué NA
    valid = dng_b_base > 0.0
    if valid.sum() >= max(int(0.5 * n_boot), 2):
        rel_b = (dng_b_base[valid] - dng_b_b[valid]) / dng_b_base[valid]
        rel_ic = (float(np.percentile(rel_b, 2.5)),
                  float(np.percentile(rel_b, 97.5)))
    else:
        rel_ic = (float('nan'), float('nan'))
    q = lambda v: (float(np.percentile(v, 2.5)),
                   float(np.percentile(v, 97.5)))
    # économie de points payés (positif = le bras paie moins) ;
    # points NÉGATIFS (pertes) -> économie = pertes_base - pertes_arm
    # = (-xa) - (-xb) = xa - xb
    saved_b = (xa[idx].sum(axis=1) - xb[idx].sum(axis=1))
    return {'n_pairs': n,
            'dmr': float((ra - rb).mean()), 'dmr_ic95': q(dmr_b),
            'dng_base': dng_b, 'dng_arm': dng_a,
            'd_dng_pts': 100.0 * (dng_a - dng_b),
            'd_dng_pts_ic95': q(dpts_b),
            'rel_dng': float(rel), 'rel_dng_ic95': rel_ic,
            'paid_base': int(pb.sum()), 'paid_arm': int(pa.sum()),
            'd_paid': int(pa.sum() - pb.sum()),
            'pts_base': float(xb.sum()), 'pts_arm': float(xa.sum()),
            'pts_saved': float(xa.sum() - xb.sum()),
            'pts_saved_ic95': q(saved_b)}


def main():
    p = argparse.ArgumentParser(
        description='[PLAN DANGER - ÉTAPE 3] sweep apparié du blend '
                    'logit − λ·p_danger sur les quasi-égalités (porte '
                    'pré-enregistrée : dng% −20% ET Δmr <= +0.05)')
    p.add_argument('--danger-head',
                   default='checkpoints/danger_head/danger_head.pt',
                   help='checkpoint danger_head CALIBRÉ (clé calib) : la '
                        'copie danger_head_cal.pt de '
                        'audit_danger_head.py --fix-out')
    p.add_argument('--allow-raw-p', action='store_true',
                   help='(déconseillé) autoriser une tête danger sans '
                        'clé calib : λ·p s\'applique alors au score BCE '
                        'pos_weighté — le bug du sweep v1')
    p.add_argument('--sl-discard',
                   default='checkpoints/sl_tonpusen_la/discard.pt')
    p.add_argument('--sl-frozen',
                   default='checkpoints/sl_tonpusen_la',
                   help='dossier des têtes SL figées (adversaires + appels)')
    p.add_argument('--grp', default=None,
                   help='checkpoint GRP (identifier à la run de référence ;'
                        ' dynamics-neutral)')
    p.add_argument('--games', type=int, default=384,
                   help='parties PAR BRAS (défaut 384 = protocole '
                        'pré-enregistré ; 96 = préversion rapide)')
    p.add_argument('--chunk-size', type=int, default=32,
                   help='parties en parallèle par chunk (garder IDENTIQUE '
                        'à la référence)')
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--taus', default='1,2,4',
                   help='liste des τ (rayon de quasi-égalité en logits), '
                        'séparés par des virgules')
    p.add_argument('--lams', default='1,4,16',
                   help='liste des λ (poids de p_danger CALIBRÉE, unités '
                        'de proba), séparés par des virgules ; v2a '
                        'recommandé : 0.5,1,2,4 ; ponts v1 : 8,16,32,64')
    p.add_argument('--gate-dng', type=float, default=0.20,
                   help='porte : réduction relative minimale de dng%%')
    p.add_argument('--gate-mr', type=float, default=0.05,
                   help='porte : Δmr maximal (positif = dégradation)')
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--out', default='checkpoints/danger_blend_sweep',
                   help='dossier de sortie (results.json)')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    taus = [float(t) for t in str(args.taus).split(',') if t.strip()]
    lams = [float(l) for l in str(args.lams).split(',') if l.strip()]
    if not taus or not lams:
        raise SystemExit("[sweep] --taus/--lams vides")
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    # même initialisation RNG qu'eval_matchup avant construction agent
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = DangerBlendAgent(
        args.danger_head, tau=taus[0], lam=lams[0],
        allow_raw_p=args.allow_raw_p,
        sl_discard_path=args.sl_discard, sl_frozen_dir=args.sl_frozen,
        device=device)

    arms = [('baseline', None, None)]
    arms += [(f'tau={t:g},lam={l:g}', t, l)
             for t in taus for l in lams]
    chunk = min(args.chunk_size, args.games)
    n_arms = len(arms)
    print(f"[sweep] {n_arms} bras x {args.games} parties (chunks de "
          f"{chunk}, seed {args.seed}, lookahead ON imposé = champ de la "
          f"tête et de la référence) | porte : dng -"
          f"{100 * args.gate_dng:g}% ET Δmr <= +{args.gate_mr:g}", flush=True)
    print(f"[sweep] ETA ~{n_arms * args.games * 5.6 / 60:.0f} min "
          f"(calibré sur 5.6 s/partie RTX 2060, affiné au 1er chunk)",
          flush=True)

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, 'results.json')
    results = {'config': {k: v for k, v in vars(args).items()},
               'ref_mr': REF_MR, 'arms': {}, 'gate': None}

    base_games = None
    t_sweep = time.time()
    try:
        for i, (label, tau, lam) in enumerate(arms):
            if tau is None:                    # ---- bras baseline ----
                agent.blend_active = False
                agent.tau = agent.lam = 0.0
            else:                              # ---- bras blend ----
                agent.tau, agent.lam = tau, lam
                agent.blend_active = True
            agent.reset_counters()
            per_game = run_arm(label, agent, args.games, chunk,
                               args.seed, device, args.grp)
            s = arm_summary(per_game)
            s['n_blend_decisions'] = agent.n_blend_decisions
            s['n_blend_swaps'] = agent.n_blend_swaps
            if tau is None:
                base_games = per_game
                print(f"[sweep] == baseline : mr={s['mr']:.3f} "
                      f"(±{s['mr_ic95']:.3f}) wr={s['wr']:.3f} | "
                      f"dng {s['dng_pct']:.2f}% ({s['n_dd']}/{s['n_rd']}) "
                      f"| PAYÉS {s['n_dpid']} ({s['paid_pct']:.2f}%, "
                      f"{s['pts_paid']:.0f} pts) | chankans "
                      f"{s['n_ckan']} ({s['pts_ckan']:.0f} pts)",
                      flush=True)
                dev = s['mr'] - REF_MR
                print(f"[sweep] contrôle non-régression : mr - {REF_MR} = "
                      f"{dev:+.3f} (attendu +0.000 si mêmes réglages que "
                      f"la référence ; sinon vérifier games/chunk/seed)",
                      flush=True)
                if s['n_rd'] == 0:
                    raise SystemExit("[sweep] baseline sans défausse "
                                     "instrumentée -- danger_labels inactif")
            else:
                pc = paired_compare(base_games, per_game,
                                    n_boot=args.n_boot, seed=args.seed)
                s['paired'] = pc
                swap_pct = (100.0 * s['n_blend_swaps']
                            / max(s['n_blend_decisions'], 1))
                gate_ok = (pc['rel_dng'] >= args.gate_dng
                           and pc['dmr'] <= args.gate_mr)
                s['gate_pass'] = bool(gate_ok)
                print(f"[sweep] == {label} : mr={s['mr']:.3f} "
                      f"(±{s['mr_ic95']:.3f}) wr={s['wr']:.3f} | dng "
                      f"{s['dng_pct']:.2f}% | swaps {swap_pct:.1f}% "
                      f"({s['n_blend_swaps']}/{s['n_blend_decisions']})",
                      flush=True)
                rel_lo, rel_hi = pc['rel_dng_ic95']
                rel_ci = (f"[{100 * rel_lo:+.1f}..{100 * rel_hi:+.1f}]"
                          if rel_lo == rel_lo else "[NA : trop peu de dng]")
                pts_lo, pts_hi = pc['d_dng_pts_ic95']
                sv_lo, sv_hi = pc['pts_saved_ic95']
                print(f"[sweep]    vs baseline : Δmr={pc['dmr']:+.3f} "
                      f"[IC95 {pc['dmr_ic95'][0]:+.3f}.."
                      f"{pc['dmr_ic95'][1]:+.3f}] | réduction dng "
                      f"{100 * pc['rel_dng']:+.1f}% {rel_ci} "
                      f"(Δ {pc['d_dng_pts']:+.2f} pt, moins = mieux, "
                      f"[{pts_lo:+.2f}..{pts_hi:+.2f}]) | "
                      f"payés {pc['paid_base']}→{pc['paid_arm']} "
                      f"(Δ {pc['d_paid']:+d}), économie "
                      f"{pc['pts_saved']:+.0f} pts "
                      f"[{sv_lo:+.0f}..{sv_hi:+.0f}] | "
                      f"PORTE : {'PASSÉE' if gate_ok else 'échouée'}",
                      flush=True)
            results['arms'][label] = {
                k: v for k, v in s.items() if k != 'per_game'}
            results['arms'][label]['per_game'] = {
                f'{c}.{e}': {'rank': g['rank_of_seat0'],
                             'rd': int(g['n_rl_discards']),
                             'dd': int(g['n_danger_discards']),
                             'dpid': int(g.get('n_dealins_paid', 0)),
                             'pts': float(g.get('points_paid', 0.0)),
                             'ckan': int(g.get('n_chankan_paid', 0)),
                             'ckpts': float(g.get('points_chankan', 0.0))}
                for (c, e), g in per_game.items()}
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        # ---- verdict final (porte pré-enregistrée) ------------------- #
        passing = [(lbl, results['arms'][lbl])
                   for lbl, _, _ in arms[1:]
                   if results['arms'][lbl].get('gate_pass')]
        results['gate'] = {
            'n_pass': len(passing),
            'best': max(passing, key=lambda x: x[1]['paired']['rel_dng'])[0]
            if passing else None}
        print()
        if passing:
            best = results['gate']['best']
            bp = results['arms'][best]['paired']
            print(f"[sweep] VERDICT : {len(passing)}/{len(arms) - 1} bras "
                  f"passent la porte ; meilleur = {best} "
                  f"(réduction dng {100 * bp['rel_dng']:+.1f}%, "
                  f"Δmr {bp['dmr']:+.3f})", flush=True)
            print("[sweep] étape suivante : geler (τ, λ) et intégrer le "
                  "blend dans l'agent de déploiement (patch séparé).",
                  flush=True)
        else:
            print("[sweep] VERDICT : AUCUN bras ne passe la porte "
                  "pré-enregistrée (dng −20% ET Δmr <= +0.05). Échec "
                  "assumé : pas de rattrapage improvisé, on analyse "
                  "(swaps vs λ, faux swaps, Δmr par τ) avant de "
                  "re-discuter le plan.", flush=True)
    finally:
        results['elapsed_s'] = time.time() - t_sweep
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[sweep] résultats : {results_path}", flush=True)


if __name__ == '__main__':
    main()
