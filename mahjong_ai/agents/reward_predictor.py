"""
Predicteur de reward Phi (section 3.2 du papier Suphx) : un GRU a 2 couches
suivi de 2 couches fully-connected, qui apprend a predire le reward final de
partie (base sur le classement) a partir de l'historique des manches jouees.

[MODIF 2026-10-13] Suppression des flags de format (tonpuu/hanchan) des
features. Remplacement de la progression relative par rounds_remaining ABSOLU
(normalise /8). FEATURE_DIM passe de 15 a 12.

[Amelioration] Ajout d'un dropout leger (GRU inter-couches + tete FC) et
d'une tranche de validation dans train_reward_predictor().
"""
import numpy as np
import torch
import torch.nn as nn

# [MODIFIE] 12 features (suppression des 2 flags format + 1 progression)
FEATURE_DIM = 12

class RewardPredictor(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """x: (B, T, input_dim) -> predictions (B, T), UNE par pas de temps."""
        h, _ = self.gru(x)
        return self.fc(h).squeeze(-1)

    @torch.no_grad()
    def predict_step(self, feat, hidden_state):
        """Inference incrementale pour UNE manche."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(feat, dtype=torch.float32,
                            device=device).view(1, 1, -1)
        h_out, new_hidden = self.gru(x, hidden_state)
        pred = self.fc(h_out[:, -1, :]).squeeze().item()
        return pred, new_hidden

def train_reward_predictor(model, optimizer, buffer, batch_size=256,
                           epochs=2, device="cpu", val_fraction=0.1):
    """Entrainement supervise (eq. 4 du papier)."""
    if len(buffer) < 100:
        return None, None

    n_val = max(1, int(len(buffer) * val_fraction))
    n_train = len(buffer) - n_val

    train_seqs = buffer.sequences[:n_train]
    train_rewards = buffer.final_rewards[:n_train]
    val_seqs = buffer.sequences[n_train:]
    val_rewards = buffer.final_rewards[n_train:]

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        indices = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < 2:
                continue
            batch_seqs = [train_seqs[j] for j in batch_idx]
            batch_rewards = [train_rewards[j] for j in batch_idx]

            max_len = max(len(s) for s in batch_seqs)
            padded = np.zeros((len(batch_seqs), max_len, FEATURE_DIM),
                              dtype=np.float32)
            for k, s in enumerate(batch_seqs):
                padded[k, :len(s)] = s

            x = torch.as_tensor(padded, device=device)
            y = torch.as_tensor(batch_rewards, device=device,
                                dtype=torch.float32)
            pred = model(x)[:, -1]
            loss = torch.nn.functional.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        if val_seqs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(val_seqs), batch_size):
                    batch_seqs = val_seqs[i:i + batch_size]
                    batch_rewards = val_rewards[i:i + batch_size]
                    if not batch_seqs:
                        continue
                    max_len = max(len(s) for s in batch_seqs)
                    padded = np.zeros(
                        (len(batch_seqs), max_len, FEATURE_DIM),
                        dtype=np.float32)
                    for k, s in enumerate(batch_seqs):
                        padded[k, :len(s)] = s
                    x = torch.as_tensor(padded, device=device)
                    y = torch.as_tensor(batch_rewards, device=device,
                                        dtype=torch.float32)
                    pred = model(x)[:, -1]
                    val_loss += torch.nn.functional.mse_loss(
                        pred, y).item()
            val_loss /= max(1, (len(val_seqs) + batch_size - 1) // batch_size)
        else:
            val_loss = None

        train_loss = total_loss / max(n_batches, 1)
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss

    return train_loss, val_loss