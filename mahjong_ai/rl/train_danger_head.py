"""[PLAN DANGER - ÉTAPE 2] Entraînement supervisé de la tête danger.

Entrée : les shards de l'étape 1 (mahjong_ai/rl/dump_danger_data.py) —
(features observables 279 canaux, labels contrefactuels 34 voies
« défausser cette tuile donne ron à un tenpai non-furiten »). Là où le
D-2 event_pen n'a vu que ~1500 événements bruités par la chance adverse
(dng% plat, mr 2.643 vs baseline 2.484), un dump de 4000 parties porte
~276k transitions × 34 labels, SANS bruit (label exact calculé des
mains cachées).

Architecture : tronc SL FIGÉ (BatchNorm en eval sur les running stats,
requires_grad=False — leçon D-2 : ne jamais modifier le tronc par
gradient) + tête danger neuve Conv1x1(C→H) + ReLU + Conv1x1(H→1) ->
34 logits, une par tuile. Seule la tête reçoit des gradients (~17k
params) : le comportement SL reste intangible, la tête ne peut rien
casser — le dommage éventuel est borné par construction à l'étape 3
(blend uniquement sur les quasi-égalités).

Loss : BCE pondérée par pos_weight = (1-p)/p (p = prior danger des
tuiles LÉGALES du split train, ~3% -> pos_weight ~32), masquée aux
seules tuiles légales (le blend de l'étape 3 ne reranke que des
candidates jouables).

Split : PAR PARTIE et par shard (game % 10 == --val-fold) — jamais par
transition (fuites intra-partie). Le prior, le pos_weight et l'init du
biais (logit du prior) sont calculés sur le split train uniquement.

Métriques val (la porte de rentabilité est pré-enregistrée) :
  * AUC (tuiles légales) — PORTE : >= 0.75 ET >= plancher heuristique
    « canal unseen » + 0.05, sinon la tête n'apprend rien au-delà de
    la visibilité des tuiles ;
  * AUC décomposée riichi actif / silencieux (le danger contre un
    riichi déclaré est lisible dans les rivières ; le tenpai silencieux
    est la partie dure) ;
  * ECE (calibration — le λ du blend étape 3 s'interprète en unités
    de probabilité) ;
  * métriques DE DÉPLOIEMENT : le SL joue a* = argmax ; pour chaque
    Δ (quasi-égalité en logits SL), on mesure (i) « swap_ok » : parmi
    les états où a* est DANGEREUX avec alternative sûre quasi-égalité,
    la tête classe-t-elle au moins une alternative sûre moins
    dangereuse que a* (le blend peut éviter le deal-in) ; (ii)
    « faux_swap » : parmi les états où a* est sûre avec alternative
    dangereuse quasi-égalité, la tête pousse-t-elle vers la tuile
    dangereuse (le risque du blend). Ces deux taux dimensionnent le
    sweep (τ, λ) de l'étape 3.

Usage (racine du repo, APRÈS le dump complet de l'étape 1 ; training
lui-même : quelques minutes sur GPU) :
    py -3.14 -m mahjong_ai.rl.train_danger_head ^
      --data checkpoints/danger_data ^
      --sl-discard checkpoints/sl_tonpusen_la/discard.pt ^
      --out checkpoints/danger_head

IMPORTANT : le champ du dump doit matcher le déploiement (lookahead ON
par défaut à l'étape 1 = conditions de l'éval de référence 2.484).
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mahjong_ai.agents.suphx_cnn import build_model, adapt_in_channels
from mahjong_ai.envs.suphx_features import (
    N_CHANNELS, N_TILES, CHANNEL_GROUPS)

N_FOLDS = 10        # split val par partie : game % 10 == --val-fold
DELTAS = (1.0, 2.0, 4.0)     # quasi-égalités en logits SL


def _channel_span(name):
    """(index, largeur) d'un groupe de canaux dans obs_normal."""
    idx = 0
    for gname, n in CHANNEL_GROUPS:
        if gname == name:
            return idx, n
        idx += n
    raise KeyError(name)


I_UNSEEN = _channel_span('unseen')[0]          # (4-vues)/4 : proxy danger
I_OPPRIICHI, W_OPPRIICHI = _channel_span('opp_riichi')


# ------------------------------------------------------------------ #
# Métriques numpy
# ------------------------------------------------------------------ #
def auc_rank(y, s):
    """AUC (Mann-Whitney) avec gestion exacte des ex æquo.
    y : booléen ; s : score (grand = plus « dangereux »)."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    s_sorted = s[order]
    change = np.empty(len(s), dtype=bool)
    change[0] = True
    if len(s) > 1:
        change[1:] = s_sorted[1:] != s_sorted[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(s))
    avg = (starts + ends + 1) / 2.0              # rangs moyens 1-indexés
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.repeat(avg, ends - starts)
    return float((ranks[y].sum()
                  - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece_bins(p, y, n_bins=10):
    """Erreur de calibration pondérée par bin (bins équiréguliers)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(p) == 0:
        return float('nan')
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            e += float(m.mean()) * abs(float(p[m].mean())
                                       - float(y[m].mean()))
    return e


def deployment_metrics(sl_logits, mask, y, p, deltas=DELTAS):
    """Métriques au niveau du déploiement : le SL joue a* = argmax des
    logits masqués ; le blend étape 3 reranke score = logit_SL − λ·p."""
    mb = np.asarray(mask) > 0
    yb = np.asarray(y) > 0
    masked = np.asarray(sl_logits, dtype=np.float64) + (mb - 1.0) * 1e9
    n = masked.shape[0]
    a = masked.argmax(axis=1)
    rows = np.arange(n)
    p_star = p[rows, a]
    y_star = yb[rows, a]
    out = {'n_states': int(n),
           'sl_argmax_dng_pct': (100.0 * float(y_star.mean())
                                 if n else float('nan'))}
    for d in deltas:
        top1 = masked.max(axis=1, keepdims=True)
        near = (masked > top1 - d) & mb
        near[rows, a] = False                   # alternatives seulement
        safe_alt = near & ~yb
        dng_alt = near & yb
        min_p_safe = np.where(safe_alt, p, np.inf).min(axis=1)
        min_p_dng = np.where(dng_alt, p, np.inf).min(axis=1)
        # (i) le SL allait défausser DANGEREUX, une sûre existe :
        #     la tête permet-elle de l'éviter ?
        m_hit = y_star & safe_alt.any(axis=1)
        # (ii) le SL jouait SÛR, une dangereuse existe : risque que le
        #      blend pousse vers la tuile dangereuse
        m_fs = (~y_star) & dng_alt.any(axis=1)
        cand = near.copy()
        cand[rows, a] = True
        out[f'delta={d:g}'] = {
            'n_alt_mean': float(near.sum(axis=1).mean()),
            'swap_ok': (int((m_hit & (min_p_safe < p_star)).sum()),
                        int(m_hit.sum())),
            'faux_swap': (int((m_fs & (min_p_dng < p_star)).sum()),
                          int(m_fs.sum())),
            'auc_near': auc_rank(yb[cand], p[cand]),
        }
    return out


# ------------------------------------------------------------------ #
# Modèle : tronc SL figé + tête danger
# ------------------------------------------------------------------ #
class DangerHead(nn.Module):
    """Conv1x1(C→H) + ReLU + Conv1x1(H→1) -> 34 logits (une par tuile).
    Miroir du DiscardHead du SL, avec une couche cachée."""

    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x):                        # (B, C, 34, 1)
        return self.conv2(self.relu(self.conv1(x)))\
                 .squeeze(-1).squeeze(1)         # (B, 34)


class DangerNet(nn.Module):
    """SL complet figé (tronc + tête discard : les logits SL servent
    aux quasi-égalités) + tête danger entraînable sur les features."""

    def __init__(self, sl_discard_path, hidden=128, device='cpu'):
        super().__init__()
        self.device = device
        ck = torch.load(sl_discard_path, map_location=device,
                        weights_only=False)
        cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
        self.arch = {'channels': cfg.get('channels', 256),
                     'n_blocks': cfg.get('n_blocks', 16)}
        self.sl = build_model('discard',
                              channels=self.arch['channels'],
                              n_blocks=self.arch['n_blocks']).to(device)
        try:
            self.sl.load_state_dict(ck['model_state_dict'])
        except RuntimeError:
            # checkpoint ancien (268 canaux) -> adaptation, cf. agent.py
            sd = dict(ck['model_state_dict'])
            w = sd.get('trunk.input_conv.weight')
            if w is not None and w.shape[1] != N_CHANNELS:
                sd['trunk.input_conv.weight'] = adapt_in_channels(
                    w, N_CHANNELS)
            self.sl.load_state_dict(sd)
        self.sl.eval()
        for q in self.sl.parameters():
            q.requires_grad = False
        self.danger = DangerHead(self.arch['channels'], hidden).to(device)
        print(f"[danger] tronc SL figé ({self.arch['channels']}x"
              f"{self.arch['n_blocks']}, val_acc={ck.get('val_acc', '?')})"
              f" + tête danger {self.arch['channels']}->{hidden}->1",
              flush=True)

    def train(self, mode=True):
        # le tronc reste TOUJOURS en eval : BatchNorm sur les running
        # stats du SL (même philosophie que PPOAgent)
        super().train(mode)
        self.sl.eval()
        return self

    def features(self, obs):                     # (B, 279, 34) float32
        with torch.no_grad():
            x = obs.unsqueeze(-1) if obs.dim() == 3 else obs
            return self.sl.trunk(x)

    def danger_logits(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32,
                              device=self.device)
        return self.danger(self.features(obs))

    def sl_logits(self, obs_np):
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, dtype=torch.float32,
                                  device=self.device)
            return self.sl(obs)


# ------------------------------------------------------------------ #
# Données : shards de l'étape 1
# ------------------------------------------------------------------ #
def scan_shards(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir,
                                          'danger_shard_*.npz')))
    if not files:
        raise SystemExit(
            f"[danger] aucun shard danger_shard_*.npz dans {data_dir} "
            f"-- lancer d'abord mahjong_ai.rl.dump_danger_data")
    return files


def load_shard(path):
    z = np.load(path)
    for k in ('obs_normal', 'mask', 'danger', 'action', 'game'):
        if k not in z:
            raise SystemExit(f"[danger] shard {path} sans clé '{k}' "
                             f"-- format étape 1 attendu")
    if z['obs_normal'].shape[1:] != (N_CHANNELS, N_TILES):
        raise SystemExit(
            f"[danger] shard {path} : obs_normal {z['obs_normal'].shape}"
            f" != (N, {N_CHANNELS}, {N_TILES})")
    return z


def _fmt_pct(x):
    return 'nan' if x != x else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser(
        description='[PLAN DANGER - ÉTAPE 2] tête danger supervisée '
                    'sur les shards de dump_danger_data (tronc SL figé)')
    ap.add_argument('--data', default='checkpoints/danger_data',
                    help='dossier des shards de l\'étape 1')
    ap.add_argument('--sl-discard',
                    default='checkpoints/sl_tonpusen_la/discard.pt',
                    help='discard.pt SL (tronc figé + logits '
                         'quasi-égalité)')
    ap.add_argument('--out', default='checkpoints/danger_head',
                    help='dossier de sortie (danger_head.pt)')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--val-fold', type=int, default=0,
                    help=f'parties avec game %% {N_FOLDS} == fold -> val '
                         f'(split PAR PARTIE, défaut 0)')
    ap.add_argument('--pos-weight-cap', type=float, default=50.0)
    ap.add_argument('--gate-auc', type=float, default=0.75,
                    help='porte : AUC val minimale')
    ap.add_argument('--gate-margin', type=float, default=0.05,
                    help='porte : marge minimale sur le plancher unseen')
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    files = scan_shards(args.data)
    model = DangerNet(args.sl_discard, hidden=args.hidden, device=device)

    # --- inventaire + prior (split train uniquement) ---------------- #
    tot = {'tr': 0, 'va': 0, 'legal': 0, 'pos': 0, 'games': 0}
    for f in files:
        z = load_shard(f)
        g = z['game']
        tr = (g % N_FOLDS) != args.val_fold
        m = z['mask'] > 0
        y = z['danger'] > 0
        tot['tr'] += int(tr.sum())
        tot['va'] += int((~tr).sum())
        tot['legal'] += int((m & tr[:, None]).sum())
        tot['pos'] += int((y & m & tr[:, None]).sum())
        tot['games'] += int(np.unique(g).size)
    if tot['tr'] == 0 or tot['va'] == 0:
        raise SystemExit(f"[danger] split vide (train {tot['tr']} / "
                         f"val {tot['va']}) -- vérifier --val-fold")
    if tot['pos'] == 0:
        raise SystemExit("[danger] AUCUNE tuile légale dangereuse dans "
                         "le split train -- dump à refaire")
    prior = tot['pos'] / tot['legal']
    pos_w = min((1.0 - prior) / prior, args.pos_weight_cap)
    print(f"[danger] {len(files)} shards, {tot['games']} parties | "
          f"transitions train={tot['tr']} val={tot['va']}", flush=True)
    print(f"[danger] prior danger (train, tuiles légales) "
          f"p={100 * prior:.3f}% -> pos_weight={pos_w:.1f} "
          f"(cap {args.pos_weight_cap})", flush=True)

    # init du biais au logit du prior : la tête démarre calibrée
    with torch.no_grad():
        model.danger.conv2.bias.fill_(math.log(prior / (1.0 - prior)))

    pos_w_t = torch.tensor(float(pos_w), device=device)
    optim = torch.optim.AdamW(model.danger.parameters(), lr=args.lr,
                              weight_decay=args.wd)

    def train_loss(logits, y, m):
        bce = F.binary_cross_entropy_with_logits(
            logits, y, reduction='none')
        w = 1.0 + y * (pos_w_t - 1.0)
        return (bce * w * m).sum() / m.sum().clamp(min=1.0)

    def val_pass():
        """Streaming des lignes val : on ne garde QUE les petits arrays
        (logits/prédiction/masques) — les obs ne restent pas en RAM."""
        model.eval()
        Ls, Ys, Ms, Ss, Us, Rs = [], [], [], [], [], []
        with torch.no_grad():
            for f in files:
                z = load_shard(f)
                idx = np.flatnonzero((z['game'] % N_FOLDS)
                                     == args.val_fold)
                if len(idx) == 0:
                    continue
                obs = z['obs_normal']
                for s in range(0, len(idx), args.batch_size):
                    j = idx[s:s + args.batch_size]
                    Ls.append(model.danger_logits(obs[j])
                              .cpu().numpy().astype(np.float32))
                    Ss.append(model.sl_logits(obs[j])
                              .cpu().numpy().astype(np.float32))
                    Ys.append(z['danger'][j])
                    Ms.append(z['mask'][j])
                    Us.append(obs[j][:, I_UNSEEN, :].astype(np.float32))
                    Rs.append(obs[j][:, I_OPPRIICHI:
                                     I_OPPRIICHI + W_OPPRIICHI, :]
                              .max(axis=(1, 2)) > 0)
        L = np.concatenate(Ls)
        S = np.concatenate(Ss)
        Y = np.concatenate(Ys)
        M = np.concatenate(Ms)
        U = np.concatenate(Us)
        R = np.concatenate(Rs)
        p = 1.0 / (1.0 + np.exp(-L.astype(np.float64)))
        mb, yb = M > 0, Y > 0
        # loss val (même pondération que le train)
        yy = yb.astype(np.float64)
        w = 1.0 + yy * (pos_w - 1.0)
        bce = -(yy * np.log(np.clip(p, 1e-7, 1.0))
                + (1.0 - yy) * np.log(np.clip(1.0 - p, 1e-7, 1.0)))
        metrics = {
            'val_loss': float((bce * w)[mb].sum() / max(int(mb.sum()), 1)),
            'auc': auc_rank(yb[mb], p[mb]),
            'auc_floor': auc_rank(yb[mb], U[mb]),
            'ece': ece_bins(p[mb], yy[mb]),
            'n_val_tr': int(M.shape[0]),
        }
        floor = metrics['auc_floor']
        metrics['auc_floor_eff'] = (floor if not (floor < 0.5)
                                    else 1.0 - floor)
        m_r = mb & R[:, None]
        m_s = mb & (~R[:, None])
        metrics['auc_riichi'] = (auc_rank(yb[m_r], p[m_r])
                                 if m_r.any() else float('nan'))
        metrics['auc_silent'] = (auc_rank(yb[m_s], p[m_s])
                                 if m_s.any() else float('nan'))
        metrics['pct_riichi_states'] = float(R.mean())
        metrics['deployment'] = deployment_metrics(S, M, Y, p)
        return metrics

    # --- boucle d'entraînement -------------------------------------- #
    best_auc, best_metrics, best_epoch = -1.0, None, -1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for f in rng.permutation(files):
            z = load_shard(f)
            idx = np.flatnonzero((z['game'] % N_FOLDS)
                                 != args.val_fold)
            if len(idx) == 0:
                continue
            obs = z['obs_normal']
            y_all, m_all = z['danger'], z['mask']
            perm = rng.permutation(idx)
            for s in range(0, len(perm), args.batch_size):
                j = perm[s:s + args.batch_size]
                logits = model.danger_logits(obs[j])
                y = torch.as_tensor(y_all[j], dtype=torch.float32,
                                    device=device)
                m = torch.as_tensor(m_all[j], dtype=torch.float32,
                                    device=device)
                loss = train_loss(logits, y, m)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
        metrics = val_pass()
        print(f"[danger] ep{ep}/{args.epochs} train_loss="
              f"{np.mean(losses):.4f} val_loss={metrics['val_loss']:.4f}"
              f" AUC={metrics['auc']:.4f} "
              f"(plancher unseen {metrics['auc_floor_eff']:.4f}, "
              f"{time.time() - t0:.0f}s)", flush=True)
        if metrics['auc'] > best_auc:
            best_auc, best_metrics, best_epoch = \
                metrics['auc'], metrics, ep
            os.makedirs(args.out, exist_ok=True)
            torch.save({
                'head': model.danger.state_dict(),
                'config': {'channels': model.arch['channels'],
                           'n_blocks': model.arch['n_blocks'],
                           'hidden': args.hidden,
                           'in_channels': N_CHANNELS,
                           'n_tiles': N_TILES},
                'trunk_source': os.path.abspath(args.sl_discard),
                'data': os.path.abspath(args.data),
                'val_fold': args.val_fold,
                'pos_weight': float(pos_w),
                'prior': float(prior),
                'epoch': ep,
                'metrics': metrics,
            }, os.path.join(args.out, 'danger_head.pt'))

    # --- rapport final (meilleure époque) ---------------------------- #
    m = best_metrics
    d = m['deployment']
    print()
    print(f"[danger] MEILLEURE ÉPOQUE : {best_epoch} "
          f"(AUC {m['auc']:.4f})", flush=True)
    print(f"[danger] porte : AUC {m['auc']:.4f} vs seuil "
          f"{args.gate_auc} | plancher unseen "
          f"{m['auc_floor_eff']:.4f} + marge {args.gate_margin}", flush=True)
    gate_pass = (m['auc'] >= args.gate_auc
                 and m['auc'] >= m['auc_floor_eff'] + args.gate_margin)
    print(f"[danger] AUC riichi actif {m['auc_riichi']:.4f} / "
          f"silencieux {m['auc_silent']:.4f} "
          f"({100 * m['pct_riichi_states']:.0f}% des états val ont un "
          f"riichi adverse) | ECE {m['ece']:.4f}", flush=True)
    print(f"[danger] déploiement : le SL argmax joue une tuile "
          f"dangereuse sur {d['sl_argmax_dng_pct']:.2f}% des "
          f"{d['n_states']} états val", flush=True)
    for k in [f'delta={d_:g}' for d_ in DELTAS]:
        dm = d[k]
        hit, n_hit = dm['swap_ok']
        fs, n_fs = dm['faux_swap']
        print(f"[danger] {k} : {dm['n_alt_mean']:.1f} alternatives/état | "
              f"swap_ok {hit}/{n_hit} ({_fmt_pct(hit / max(n_hit, 1))}) "
              f"| faux_swap {fs}/{n_fs} "
              f"({_fmt_pct(fs / max(n_fs, 1))}) | "
              f"AUC_near {dm['auc_near']:.4f}", flush=True)
    if gate_pass:
        print(f"[danger] PORTE PASSÉE -> patch étape 3 : blend "
              f"logit_SL − λ·p_danger sur les quasi-égalités "
              f"(sweep τ/λ sur l'éval appariée 384 seed 777 ; λ initial "
              f"suggéré ~0.5-1.0 en logits, prior p="
              f"{100 * prior:.2f}%)", flush=True)
    elif m['auc'] >= args.gate_auc - 0.05:
        print("[danger] PORTE LIMITÉE (AUC marginal) : étendre le dump "
              "(nouvelle --seed + --shard-offset à l'étape 1) et/ou "
              "--epochs 5 avant de statuer", flush=True)
    else:
        print("[danger] PORTE ÉCHOUÉE : la tête ne sépare pas le danger "
              "au-delà de la visibilité -> le signal observable est "
              "insuffisant, re-discuter avant l'étape 3", flush=True)
    print(f"[danger] checkpoint : {os.path.join(args.out, 'danger_head.pt')}",
          flush=True)


if __name__ == '__main__':
    main()
