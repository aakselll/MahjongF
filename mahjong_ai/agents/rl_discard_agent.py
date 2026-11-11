"""
Agent RL pour Suphx - Phase 1 (RL-1 avec GRP).

Architecture :
    - Policy    : SuphxModel('discard'), initialisee avec le SL discard
    - Critic    : MLP separe prenant la meme obs (250, 34)
    - Frozen SL : riichi/chow/pong/kong (charges depuis le SL, non entraines)

Reference : Suphx paper section 3.1 (PPO + entropy regularization).
"""
import os

import numpy as np
import torch
import torch.nn as nn

from mahjong_ai.agents.suphx_cnn import build_model, N_CHANNELS
from mahjong_ai.envs.suphx_features import N_TILES


FROZEN_HEADS = ['riichi', 'chow', 'pong', 'kong']


class Critic(nn.Module):
    """Value function V(s). MLP separe prenant l'obs (N_CHANNELS, 34)."""

    def __init__(self, in_channels=N_CHANNELS, in_tiles=N_TILES, hidden=512):
        super().__init__()
        flat = in_channels * in_tiles
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'net.1.weight'
        w = state_dict.get(key)
        if w is not None and w.shape[1] != self.net[1].in_features:
            from mahjong_ai.agents.suphx_cnn import adapt_flat_in_features
            state_dict[key] = adapt_flat_in_features(w, self.net[1].in_features)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RLDiscardAgent:
    """Wrapper qui regroupe la policy (discard), le critic et les SL figes."""

    def __init__(self, sl_discard_path, sl_frozen_dir, device='cpu',
                 lr=1e-4, critic_lr=1e-3):
        self.device = device

        # --- 1. Policy : charge le discard SL ---
        self.policy = build_model('discard').to(device)
        ckpt = torch.load(sl_discard_path, map_location=device,
                          weights_only=False)
        cfg = ckpt.get('config', {})
        # Note : build_model ci-dessus utilise les defaults ; si le SL a une
        # config differente, il faudra passer les params (channels, n_blocks).
        self.policy.load_state_dict(ckpt['model_state_dict'])
        print(f"[rl] discard SL charge (val_acc={ckpt.get('val_acc', '?')})",
              flush=True)

        # --- 2. Critic ---
        self.critic = Critic().to(device)

        # --- 3. Frozen SL heads ---
        self.frozen_sl = {}
        for head in FROZEN_HEADS:
            path = os.path.join(sl_frozen_dir, f'{head}.pt')
            if not os.path.isfile(path):
                print(f"[rl] WARN : {path} introuvable, skip {head}",
                      flush=True)
                continue
            m = build_model(head).to(device)
            ck = torch.load(path, map_location=device, weights_only=False)
            m.load_state_dict(ck['model_state_dict'])
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            self.frozen_sl[head] = m
            print(f"[rl] {head} SL fige (val_acc={ck.get('val_acc', '?')})",
                  flush=True)

        # --- 4. Optimizers ---
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                              lr=critic_lr)

    # ------------------------------------------------------------------ #
    def select_action(self, obs, mask, deterministic=False):
        """Forward sur la policy discard.

        obs : np.array (N_CHANNELS, 34) float32
        mask : np.array (34,) float32   (1 = tuile discardable)

        Retourne (action_idx, log_prob, value, entropy).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)

        logits = self.policy(obs_t)                 # (1, 34)
        # Masque : les tuiles illegales recoivent -inf
        masked_logits = logits + (mask_t - 1.0) * 1e9
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = (torch.argmax(masked_logits, dim=-1) if deterministic
                  else dist.sample())

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        with torch.no_grad():
            value = self.critic(obs_t)

        return (int(action.item()),
                float(log_prob.item()),
                float(value.item()),
                float(entropy.item()))

    # ------------------------------------------------------------------ #
    def evaluate_actions(self, obs_batch, mask_batch, actions_batch):
        """Batch forward pour PPO update.

        obs_batch    : (B, N_CHANNELS, 34)
        mask_batch   : (B, 34)
        actions_batch: (B,) long

        Retourne (log_probs, values, entropies).
        """
        logits = self.policy(obs_batch)
        masked_logits = logits + (mask_batch - 1.0) * 1e9
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions_batch)
        entropies = dist.entropy()
        values = self.critic(obs_batch)
        return log_probs, values, entropies

    # ------------------------------------------------------------------ #
    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck['policy'])
        self.critic.load_state_dict(ck['critic'])