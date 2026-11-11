"""
Entrainement supervise du Global Reward Predictor (GRP) sur des logs humains.

[MODIF 2026-10-13] Suppression de la double vue hanchan/tonpuu et des
flags de format. Une seule vue par partie, format detecte automatiquement.
FEATURE_DIM = 12 (rounds_remaining absolu remplace progression + format).

Usage :
python -m mahjong_ai.agents.train_grp_sl \
    --data-dir data/tenhou --max-files 100000 --epochs 15
"""
import argparse
import glob
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM
from mahjong_ai.sl.mjai_parser import parse_mjai_zst_file, _detect_format

# ====================================================================== #
# Constantes
# ====================================================================== #

REWARD_SCALE = 1.0 / 1000.0
RANK_REWARDS = {1: 15.0, 2: 5.0, 3: -5.0, 4: -15.0}
MAX_ROUNDS = 12
MAX_PHYSICAL_ROUNDS = 12
EXTREME_REWARD_THRESHOLD = 10.0
EXTREME_REWARD_WEIGHT = 3.0

# ====================================================================== #
# Extraction  [MODIFIE]
# ====================================================================== #

def _rank_from_scores(scores):
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    ranks = [0] * 4
    for rank, player in enumerate(order, start=1):
        ranks[player] = rank
    return ranks

def extract_games_from_events(events):
    """Extrait les features par manche pour les 4 joueurs.
    [MODIFIE] Une seule vue par partie, format detecte automatiquement.
    [SUPPRIME] Double vue hanchan/tonpuu et flags de format.
    """
    game_format = _detect_format(events)
    normal_rounds = 4 if game_format == "tonpuu" else 8

    scores = [25000.0] * 4
    current_riichi = [False] * 4
    oya = 0
    round_idx = 0
    n_rounds = 0
    normal_round_index = 0
    is_last_normal_round = False
    is_extension = False
    honba = 0
    kyotaku = 0
    feats_by_player = [[] for _ in range(4)]

    def _emit_round(deltas, dealer_seat):
        nonlocal scores, current_riichi, round_idx
        for p in range(4):
            scores[p] += deltas[p]
        scores_arr = np.array(scores, dtype=np.float32)
        ranks = _rank_from_scores(scores_arr)
        leader = float(scores_arr.max())
        last = float(scores_arr.min())

        # [MODIFIE] rounds_remaining ABSOLU
        rounds_remaining_abs = max(normal_rounds - (round_idx + 1), 0)

        for p in range(4):
            feat = np.array([
                deltas[p] * REWARD_SCALE,
                scores[p] * REWARD_SCALE,
                1.0 if p == dealer_seat else 0.0,
                rounds_remaining_abs / 8.0,  # [REPLACE] etait progression
                1.0 if current_riichi[p] else 0.0,
                ranks[p] / 4.0,
                (leader - scores[p]) * REWARD_SCALE,
                (scores[p] - last) * REWARD_SCALE,
                # [SUPPRIME] format_flags[0],
                # [SUPPRIME] format_flags[1],
                1.0 if is_last_normal_round else 0.0,
                1.0 if is_extension else 0.0,
                min(max(honba, 0), 4) / 4.0,
                min(max(kyotaku, 0), 4) / 4.0,
            ], dtype=np.float32)
            feats_by_player[p].append(feat)
        round_idx += 1

    for ev in events:
        ev_type = ev.get("type")

        if ev_type == "start_game":
            scores = [25000.0] * 4
            continue

        if ev_type == "start_kyoku":
            if n_rounds >= MAX_PHYSICAL_ROUNDS:
                break
            s = ev.get("scores")
            if isinstance(s, list) and len(s) == 4:
                scores = [float(x) for x in s]
            oya = int(ev.get("oya", 0) or 0)
            kyoku = int(ev.get("kyoku", 1) or 1)
            if ev.get("bakaze") == "E":
                normal_round_index = kyoku
            elif ev.get("bakaze") == "S":
                normal_round_index = 4 + kyoku
            else:
                normal_round_index = 8 + kyoku
            is_last_normal_round = normal_round_index == normal_rounds
            is_extension = normal_round_index > normal_rounds
            honba = int(ev.get("honba", 0) or 0)
            kyotaku = int(ev.get("kyotaku", 0) or 0)
            current_riichi = [False] * 4
            n_rounds += 1
            continue

        if ev_type == "reach":
            actor = ev.get("actor")
            if actor is not None and 0 <= actor < 4:
                current_riichi[actor] = True
            continue

        if ev_type in ("hora", "ryukyoku"):
            deltas = ev.get("deltas")
            if not (isinstance(deltas, list) and len(deltas) == 4):
                deltas = [0] * 4
            _emit_round([float(x) for x in deltas], oya)
            continue

    if round_idx == 0:
        return None
    if any(len(f) == 0 for f in feats_by_player):
        return None

    final_ranks = _rank_from_scores([float(x) for x in scores])
    rewards_final = [RANK_REWARDS[final_ranks[p]] for p in range(4)]
    return feats_by_player, rewards_final

# ====================================================================== #
# Dataset PyTorch
# ====================================================================== #

class GRPDataset(Dataset):
    def __init__(self, data_dir, max_files=None, shuffle_files=True,
                 cache_path=None, max_rounds=MAX_ROUNDS):
        self.max_rounds = max_rounds
        self.samples = []

        if cache_path and os.path.isfile(cache_path):
            print(f"[dataset] chargement du cache : {cache_path}")
            t0 = time.time()
            data = np.load(cache_path, allow_pickle=True).item()
            if data.get("feature_dim") != FEATURE_DIM:
                raise RuntimeError(
                    f"Cache GRP incompatible: "
                    f"feature_dim={data.get('feature_dim')}, "
                    f"attendu={FEATURE_DIM}. Utilisez un nouveau cache-path."
                )
            self.samples = data["samples"]
            print(f"[dataset] {len(self.samples)} echantillons charges en "
                  f"{time.time()-t0:.1f}s")
            return

        files = sorted(glob.glob(
            os.path.join(data_dir, "**", "*.zst"), recursive=True))
        print(f"[dataset] {len(files)} fichiers trouvés dans {data_dir}")
        if not files:
            raise RuntimeError(f"Aucun fichier .zst dans {data_dir}")
        if shuffle_files:
            random.Random(42).shuffle(files)
        if max_files:
            files = files[:max_files]

        print(f"[dataset] parsing de {len(files)} fichiers...")
        t0 = time.time()
        n_games = 0
        n_failed = 0
        n_skipped_incomplete = 0

        for i, path in enumerate(files):
            if (i + 1) % max(1, len(files) // 10) == 0:
                print(f"  ... {i+1}/{len(files)} fichiers  "
                      f"({n_games} parties, {len(self.samples)} samples, "
                      f"{n_skipped_incomplete} parties incompletes) "
                      f"[{time.time()-t0:.0f}s]")
            try:
                events = parse_mjai_zst_file(path)
                if not events:
                    n_failed += 1
                    continue

                # [MODIFIE] Une seule vue par partie
                result = extract_games_from_events(events)
                if result is None:
                    n_skipped_incomplete += 1
                    n_failed += 1
                    continue

                feats_by_player, rewards_final = result
                for p in range(4):
                    feats = feats_by_player[p]
                    T = len(feats)
                    if T == 0:
                        continue
                    if T > self.max_rounds:
                        feats = feats[:self.max_rounds]
                    feat_arr = np.stack(feats, axis=0).astype(np.float32)
                    self.samples.append(
                        (feat_arr, float(rewards_final[p])))
                n_games += 1
            except Exception:
                n_failed += 1

        print(f"[dataset] {n_games} parties parsees, "
              f"{len(self.samples)} echantillons, "
              f"{n_failed} echecs, "
              f"{n_skipped_incomplete} incompletes "
              f"[{time.time()-t0:.0f}s]")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            np.save(cache_path, {
                "samples": self.samples,
                "feature_dim": FEATURE_DIM,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, reward = self.samples[idx]
        T = len(feats)
        if T < self.max_rounds:
            padded = np.zeros((self.max_rounds, FEATURE_DIM), dtype=np.float32)
            padded[:T] = feats
        else:
            padded = feats[:self.max_rounds]
            T = self.max_rounds
        # [FIX GRP] on renvoie aussi la longueur reelle (pour masquer le
        # padding dans la loss : prediction a CHAQUE manche reelle).
        return padded, reward, T

# ====================================================================== #
# Entrainement
# ====================================================================== #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--max-files', type=int, default=None)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--cache-path', type=str, default=None)
    p.add_argument('--val-fraction', type=float, default=0.05)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[GRP] device={device}")

    ds = GRPDataset(args.data_dir, max_files=args.max_files,
                    cache_path=args.cache_path)

    # [FIX GRP] vraie validation (avant : 'val_loss' = loss d'entrainement)
    n_val = max(1, int(len(ds) * args.val_fraction))
    gen = torch.Generator().manual_seed(0)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=gen)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = RewardPredictor(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def masked_loss(pred, target, lens):
        """[FIX GRP] Eq. 4 du papier Suphx : MSE entre Phi(x^1..x^j) et le
        reward final R, moyennee sur TOUTES les manches j reelles d'une
        partie (padding exclu), puis sur les parties.
        Avant : seule la sortie du DERNIER pas (souvent un pas de padding)
        etait supervisee -> Phi(x^k) des manches intermediaires, qui sert
        de reward RL (Phi_k - Phi_{k-1}), n'etait jamais entraine."""
        T = pred.shape[1]
        steps = torch.arange(T, device=pred.device)[None, :]
        mask = (steps < lens[:, None]).float()
        err = (pred - target[:, None]) ** 2
        w = torch.where(
            torch.abs(target) > EXTREME_REWARD_THRESHOLD,
            torch.full_like(target, EXTREME_REWARD_WEIGHT),
            torch.ones_like(target))[:, None]
        per_game = (err * w * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return per_game.mean()

    def evaluate():
        """Loss de validation + R2 par position de manche (1..4)."""
        model.eval()
        tot, n = 0.0, 0
        se = np.zeros(MAX_ROUNDS); cnt = np.zeros(MAX_ROUNDS)
        ys = []
        with torch.no_grad():
            for x, y, lens in val_loader:
                x = x.to(device); y = y.float().to(device)
                lens = lens.to(device)
                pred = model(x)
                tot += masked_loss(pred, y, lens).item() * len(y)
                n += len(y)
                ys.append(y.cpu().numpy())
                err = ((pred - y[:, None]) ** 2).cpu().numpy()
                for j in range(MAX_ROUNDS):
                    m = (lens.cpu().numpy() > j)
                    se[j] += err[m, j].sum(); cnt[j] += m.sum()
        var = float(np.concatenate(ys).var()) if ys else 1.0
        r2 = [1.0 - (se[j] / cnt[j]) / var if cnt[j] else float('nan')
              for j in range(min(6, MAX_ROUNDS))]
        return tot / max(n, 1), r2

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y, lens in loader:
            x = x.to(device)
            y = y.float().to(device)
            lens = lens.to(device)
            pred = model(x)                       # (B, T) toutes les manches
            loss = masked_loss(pred, y, lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_loss, r2 = evaluate()
        print(f"[GRP] epoch {epoch+1}/{args.epochs} train={avg_loss:.4f} "
              f"val={val_loss:.4f}  R2 par manche 1..{len(r2)} : "
              + " ".join(f"{v:.2f}" for v in r2))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_dim': FEATURE_DIM,
                'hidden_dim': args.hidden_dim,
                'val_loss': val_loss,
                'epoch': epoch + 1,
            }, args.output)
            print(f"  -> save ({args.output}, val={val_loss:.4f})")

if __name__ == '__main__':
    main()