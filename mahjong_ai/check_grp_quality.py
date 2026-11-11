"""check_grp_quality.py -- Audit qualitatif complet du GRP entraine.

Repond a : "le GRP predit-il bien les parties ?" sur VOTRE checkpoint et
VOTRE dataset, avec des metriques honnetes :

  1. Metadonnees du checkpoint (feature_dim doit etre 12, hidden_dim,
     val_loss, epoch) ;
  2. Loss masquee + R2 par position de manche (1..12) sur des parties
     TENUES DE COTE (split PAR PARTIE, pas par sample) ;
  3. Prediction du vainqueur par partie : top-1, ordre exact, Spearman
     (les 4 joueurs d'une meme partie sont compares entre eux) ;
  4. Calibration : Phi moyen par vrai reward final (+15/+5/-5/-15) ;
  5. Checks comportementaux (credit temporel) : un +8000 vaut-il plus
     en E4 qu'en E2 ? un deal-in coute-t-il plus cher a l'orasu ?

Verdicts automatiques :
  - R2 faible (< 0.05) aux manches 1-2 + calibration plate  =>  le
    checkpoint a probablement ete entraint avec l'ANCIENNE loss (seul le
    dernier pas etait supervise, cf. [FIX GRP] dans train_grp_sl.py) :
    les Phi intermediaires -- exactement ce que le RL consomme --
    n'ont jamais ete entraintes. Reentrainer.
  - R2 qui croit de ~0.1 (E1) a >0.8 (fin) + calibration monotone  =>
    GRP sain.

Usage (depuis la racine du depot) :
    py -3.14 -m mahjong_ai.check_grp_quality ^
        --checkpoint checkpoints/grp_tonpusen.pt ^
        --cache grp_dataset_cache.npy

    # sans cache (re-parse les logs, plus lent) :
    py -3.14 -m mahjong_ai.check_grp_quality ^
        --checkpoint checkpoints/grp_tonpusen.pt ^
        --data-dir data/tenhou_tonpusen --max-files 2000
"""
import argparse
import sys

import numpy as np
import torch

from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM
from mahjong_ai.agents.train_grp_sl import (
    EXTREME_REWARD_THRESHOLD, EXTREME_REWARD_WEIGHT,
)

results = {"pass": 0, "fail": 0, "warn": 0}


def check(name, cond, detail="", warn=False):
    key = "pass" if cond else ("warn" if warn else "fail")
    results[key] += 1
    tag = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[key]
    print(f"  [{tag}] {name}" + ("" if cond else f"   {detail}"))
    return cond


# --------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------- #
def load_samples(args):
    if args.cache:
        data = np.load(args.cache, allow_pickle=True).item()
        samples = data["samples"]
        print(f"[data] {len(samples)} samples charges depuis {args.cache}")
        return samples
    if args.data_dir:
        from mahjong_ai.agents.train_grp_sl import GRPDataset
        ds = GRPDataset(args.data_dir, max_files=args.max_files)
        return ds.samples
    raise SystemExit("--cache ou --data-dir requis")


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    feat_dim = ck.get("feature_dim", None)
    hidden = ck.get("hidden_dim", None)
    print(f"[ckpt] {path}")
    print(f"[ckpt] feature_dim={feat_dim}  hidden_dim={hidden}  "
          f"val_loss={ck.get('val_loss', '?')}  epoch={ck.get('epoch', '?')}")
    if feat_dim is not None and feat_dim != FEATURE_DIM:
        raise SystemExit(
            f"\n[ERREUR] checkpoint feature_dim={feat_dim} != "
            f"{FEATURE_DIM} (format actuel).\n"
            "Ce GRP a ete entraine avec l'ancien format (15 features, "
            "avant la modif 2026-10-13).\nIl est INCOMPATIBLE avec "
            "game_env actuel : reentrainez-le (train_grp_sl.py).")
    if hidden is None:
        # devine depuis les poids du GRU
        hidden = sd["gru.weight_ih_l0"].shape[0] // 3
        print(f"[ckpt] hidden_dim deduit des poids : {hidden}")
    model = RewardPredictor(hidden_dim=hidden)
    model.load_state_dict(sd)
    model.eval()
    return model


# --------------------------------------------------------------------- #
# Batching (padding par batch, longueurs preservees)
# --------------------------------------------------------------------- #
def batch_forward(model, samples, batch=512):
    maxT = max(len(s) for s, _ in samples)
    preds, ys, lens = [], [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            x = np.zeros((len(chunk), maxT, FEATURE_DIM), dtype=np.float32)
            ln = np.zeros(len(chunk), dtype=np.int64)
            for k, (s, _) in enumerate(chunk):
                x[k, :len(s)] = s
                ln[k] = len(s)
            p = model(torch.as_tensor(x)).numpy()
            preds.append(p)
            ys.append(np.array([r for _, r in chunk], dtype=np.float32))
            lens.append(ln)
    return (np.concatenate(preds), np.concatenate(ys),
            np.concatenate(lens))


# --------------------------------------------------------------------- #
def masked_mse(pred, y, lens):
    T = pred.shape[1]
    steps = np.arange(T)[None, :]
    mask = (steps < lens[:, None]).astype(np.float32)
    w = np.where(np.abs(y) > EXTREME_REWARD_THRESHOLD,
                 EXTREME_REWARD_WEIGHT, 1.0)[:, None]
    per = ((pred - y[:, None]) ** 2 * w * mask).sum(1) / mask.sum(1)
    return float(per.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache', default=None,
                    help='cache .npy cree par train_grp_sl (--cache-path)')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--max-files', type=int, default=2000)
    ap.add_argument('--val-games', type=float, default=0.25,
                    help='fraction de parties gardees pour l eval')
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    samples = load_samples(args)

    # regroupement par partie : GRPDataset ajoute les 4 vues consecutives
    if len(samples) % 4 != 0:
        print(f"[WARN] {len(samples)} samples non multiple de 4 : "
              "regroupement par 4 approximatif")
    games = [samples[i:i + 4] for i in range(0, len(samples) - 3, 4)]

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(games))
    n_val = max(4, int(len(games) * args.val_games))
    val_games = [games[i] for i in idx[:n_val]]
    print(f"[split] {len(games)} parties : "
          f"{len(games) - n_val} train / {n_val} val "
          f"(split PAR PARTIE, les 4 vues d'une partie restent ensemble)")

    val_samples = [s for g in val_games for s in g]
    pred, y, lens = batch_forward(model, val_samples)
    var_y = float(y.var())

    print("\n=== 1. Loss et variance expliquee ===")
    mse = masked_mse(pred, y, lens)
    print(f"  MSE masquee (eq.4) = {mse:.3f}")
    print(f"  baseline predict-0 = {var_y:.3f}")
    r2_global = 1 - mse / var_y
    print(f"  => variance expliquee (toutes manches confondues) : "
          f"{100 * r2_global:.1f}%")

    print("\n=== 2. R2 par position de manche ===")
    r2_early = None
    maxT = pred.shape[1]
    for j in range(min(12, maxT)):
        m = lens > j
        if m.sum() < 20:
            break
        err = ((pred[m, j] - y[m]) ** 2).mean()
        r2 = 1 - err / var_y
        if j == 0:
            r2_early = r2
        print(f"    manche {j+1:2d}: n={m.sum():5d}  R2={r2:+.3f}")

    print("\n=== 3. Prediction du vainqueur par partie ===")
    # val_samples est construit partie par partie (4 vues consecutives) ;
    # pred/lens de batch_forward conservent cet ordre.
    for j in range(min(12, maxT)):
        top1 = exact = n_ok = 0
        for gi, g in enumerate(val_games):
            if any(len(s) <= j for s, _ in g):
                continue
            base = gi * 4
            phi = pred[base:base + 4, j]
            rew = np.array([r for _, r in g])
            po, to = np.argsort(-phi), np.argsort(-rew)
            n_ok += 1
            top1 += po[0] == to[0]
            exact += list(po) == list(to)
        if n_ok < 20:
            break
        print(f"    manche {j+1:2d}: n={n_ok:4d}  top1={100*top1/n_ok:5.1f}%"
              f"  ordre exact={100*exact/n_ok:5.1f}%  (chance 25% / 4.2%)")

    print("\n=== 4. Calibration (Phi moyen par vrai reward final) ===")
    mono_ok = True
    for j, lbl in [(0, "manche 1"), (None, "derniere manche")]:
        means = []
        for target in (15.0, 5.0, -5.0, -15.0):
            m = (y == target)
            if j is not None:
                m = m & (lens > j)
            if m.sum() == 0:
                continue
            vals = (pred[np.where(m)[0], j] if j is not None
                    else np.array([pred[i, lens[i] - 1]
                                   for i in np.where(m)[0]]))
            means.append(vals.mean())
            print(f"    {lbl:15s} vrai={target:+5.1f}  n={len(vals):5d}  "
                  f"Phi moy={vals.mean():+6.2f}  std={vals.std():5.2f}")
        mono_ok &= all(means[i] > means[i + 1] for i in range(len(means) - 1))
    check("calibration monotone (1er > 2e > 3e > 4e)", mono_ok)

    print("\n=== 5. Credit temporel (comportement attendu du shaping) ===")
    behav = behavioral_checks(model)
    check("un gain certain (E4) vaut plus qu'un gain totif (E2)",
          behav["gain_tard"] > behav["gain_tot"])
    check("un deal-in a l'orasu coute plus cher qu'en E2",
          behav["perte_tard"] < behav["perte_tot"])

    print("\n=== VERDICT ===")
    if r2_early is None:
        print("  [?] pas assez de donnees pour juger")
        return 1
    if r2_early < 0.05:
        print("  [!!] R2 quasi nul des les premieres manches : le checkpoint")
        print("       a probablement ete entraint avec l'ANCIENNE loss")
        print("       (seul le dernier pas supervise). Les Phi intermediaires")
        print("       que le RL consomme (Phi_k - Phi_{k-1}) n'ont jamais ete")
        print("       entraintes -> REENTRAINER avec train_grp_sl.py actuel.")
        results["fail"] += 1
    else:
        print(f"  [OK] R2 manche 1 = {r2_early:+.3f} : les Phi intermediaires")
        print("       sont bien entraintes (loss masquee par manche).")
        results["pass"] += 1
    print(f"\n{results['pass']} OK / {results['warn']} WARN / "
          f"{results['fail']} FAIL")
    return 0 if results["fail"] == 0 else 1


def pred_val_at(model, seq, j):
    """Phi au pas j d'une sequence (cache simple par appel)."""
    x = torch.as_tensor(np.asarray(seq[:j + 1], dtype=np.float32)
                        ).unsqueeze(0)
    with torch.no_grad():
        return float(model(x)[0, -1])


def behavioral_checks(model):
    """Reprend les tests de check_grp_behavior.py (credit temporel)."""

    def phi_prefix(seq):
        x = torch.as_tensor(np.array(seq, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            return float(model(x)[0, -1])

    def feat(delta=0, score=20000, rank=4, rr=3, gap_leader=5000,
             gap_last=0):
        return [delta / 1000, score / 1000, 0.0, rr / 8.0, 0.0, rank / 4.0,
                gap_leader / 1000, gap_last / 1000, 1.0 if rr == 0 else 0.0,
                0.0, 0.0, 0.0]

    apres = dict(score=28000, rank=2, gap_leader=2000)
    saut_tot = (phi_prefix([feat(rr=3), feat(delta=8000, rr=2, **apres)])
                - phi_prefix([feat(rr=3)]))
    saut_tard = (phi_prefix([feat(rr=3), feat(rr=2), feat(rr=1),
                              feat(delta=8000, rr=0, **apres)])
                 - phi_prefix([feat(rr=3), feat(rr=2), feat(rr=1)]))
    lead32 = dict(score=32000, rank=1, gap_leader=0, gap_last=7000)
    apres3 = dict(score=20000, rank=3, gap_leader=10000)
    saut_de = (phi_prefix([feat(**lead32, rr=3),
                           feat(delta=-12000, rr=2, **apres3)])
               - phi_prefix([feat(**lead32, rr=3)]))
    saut_dl = (phi_prefix([feat(**lead32, rr=3), feat(**lead32, rr=2),
                           feat(**lead32, rr=1),
                           feat(delta=-12000, rr=0, **apres3)])
               - phi_prefix([feat(**lead32, rr=3), feat(**lead32, rr=2),
                             feat(**lead32, rr=1)]))
    print(f"    DeltaPhi(+8000 en E2) = {saut_tot:+.2f} ; "
          f"en E4 = {saut_tard:+.2f}")
    print(f"    DeltaPhi(-12000 en E2) = {saut_de:+.2f} ; "
          f"en E4 = {saut_dl:+.2f}")
    return {"gain_tot": saut_tot, "gain_tard": saut_tard,
            "perte_tot": saut_de, "perte_tard": saut_dl}


if __name__ == '__main__':
    sys.exit(main())
