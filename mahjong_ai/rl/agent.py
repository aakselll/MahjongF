"""Agent RL reworké : policy + critic asymétrique + têtes SL figées, BATCHÉS.

Différences vs agents/oracle_agent.py (legacy) :
    * toutes les méthodes de décision prennent un BATCH de requêtes et
      font UN seul forward par modèle (l'ancien code faisait un forward
      batch-1 par requête : ~5-10x plus lent, et les têtes SL étaient
      appelées une à une DANS l'env via _sl_decide, donc jamais batchables) ;
    * le discard SL des adversaires est un modèle 279 canaux chargé tel
      quel (comportement identique à l'ancien select_action_sl qui
      remplissait l'oracle de zéros) ;
    * le critic asymétrique (voit l'oracle complet) est conservé : c'est
      une bonne idée du pipeline v9, il prédit G_t / reward_scale en unités
      normalisées ;
    * sauvegarde au format enrichi {'policy','critic','config',...} tout en
      gardant les clés 'policy'/'critic' -> eval_stable_rank.py et
      export_rl_discard.py continuent de fonctionner tels quels.
"""
import copy
import os

import numpy as np
import torch

from mahjong_ai.agents.suphx_cnn import (
    build_model, N_CHANNELS, ORACLE_DIM, IN_CHANNELS_ORACLE,
    adapt_in_channels,
)
from mahjong_ai.agents.oracle_agent import Critic          # réutilisé tel quel
from mahjong_ai.envs.suphx_features import N_TILES

FROZEN_HEADS = ['riichi', 'chow', 'pong', 'kong']


class PPOAgent:
    """Policy discard (283 canaux = 279 + 4 oracle) + critic + SL figés.

    La policy reste TOUJOURS en mode eval (BatchNorm figée sur les running
    stats du SL) : la policy qui joue == celle qu'on entraîne, le ratio PPO
    n'est pas bruité. Les gradients passent normalement en eval.
    """

    def __init__(self, sl_discard_path, sl_frozen_dir, device='cpu',
                 policy_lr=1e-4, critic_lr=1e-3,
                 oracle_init='kaiming', oracle_scale=0.25):
        self.device = device

        # ---------------- Policy (283 canaux, warm-start SL) ----------------
        ck = torch.load(sl_discard_path, map_location=device,
                        weights_only=False)
        cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
        self.arch = {
            'channels': cfg.get('channels', 256),
            'n_blocks': cfg.get('n_blocks', 16),
        }
        self.policy = build_model(
            'discard', use_oracle=True,
            channels=self.arch['channels'],
            n_blocks=self.arch['n_blocks'],
        ).to(device)
        self._load_partial_sl(ck['model_state_dict'],
                              oracle_init=oracle_init,
                              oracle_scale=oracle_scale)
        with torch.no_grad():
            self.oracle_init_norm = self.oracle_weight_norm()
        self.policy.eval()
        print(f"[ppo_agent] policy discard SL chargée "
              f"(val_acc={ck.get('val_acc', '?')}, "
              f"arch={self.arch['channels']}x{self.arch['n_blocks']}, "
              f"oracle_init={oracle_init}x{oracle_scale}, "
              f"||W_o||={self.oracle_init_norm:.4f})", flush=True)

        # ------------- Discard SL des adversaires (279 canaux) --------------
        # Modèle à part, chargé du checkpoint SL : comportement identique à
        # l'ancien select_action_sl (oracle=0 en entrée -> contribution
        # nulle des canaux oracle, quelle que soit leur init).
        self.sl_discard = build_model(
            'discard',
            channels=self.arch['channels'],
            n_blocks=self.arch['n_blocks'],
        ).to(device)
        try:
            self.sl_discard.load_state_dict(ck['model_state_dict'])
        except RuntimeError:
            # checkpoint 268/283 canaux -> adaptation (blocs absents=0)
            sd = dict(ck['model_state_dict'])
            w = sd.get('trunk.input_conv.weight')
            if w is not None and w.shape[1] != N_CHANNELS:
                sd['trunk.input_conv.weight'] = adapt_in_channels(
                    w, N_CHANNELS)
            self.sl_discard.load_state_dict(sd)
        self.sl_discard.eval()
        for p in self.sl_discard.parameters():
            p.requires_grad = False

        # --------------------------- Critic (283) ----------------------------
        self.critic = Critic().to(device)
        self.critic.eval()

        # ------------------ Têtes SL figées riichi/chow/pong/kong ------------
        self.frozen_sl = {}
        for head in FROZEN_HEADS:
            path = os.path.join(sl_frozen_dir, f'{head}.pt')
            if not os.path.isfile(path):
                print(f"[ppo_agent] tête SL {head} MANQUANTE ({path}) : "
                      f"l'env recevra l'action sûre 0", flush=True)
                continue
            try:
                hck = torch.load(path, map_location=device,
                                 weights_only=False)
                hcfg = hck.get('config', {}) if isinstance(hck, dict) else {}
                m = build_model(
                    head,
                    channels=hcfg.get('channels', 256),
                    n_blocks=hcfg.get('n_blocks', 16),
                    binary_hidden=hcfg.get('binary_hidden', 256),
                    in_channels=N_CHANNELS,
                ).to(device)
                m.load_state_dict(hck['model_state_dict'])
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                self.frozen_sl[head] = m
            except Exception as e:
                print(f"[ppo_agent] tête SL {head} non chargée : {e}",
                      flush=True)

        # ---------------------- Ancre KL optionnelle -------------------------
        # Par défaut le SL de départ (comportement de l'ancien pipeline) ;
        # --kl-ref permet d'ancrer sur un checkpoint de fin de phase 1.
        self.ref_policy = copy.deepcopy(self.policy).eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # --------------------------- Optimizers ------------------------------
        self.policy_optim = torch.optim.Adam(self.policy.parameters(),
                                             lr=policy_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)
        self.policy_lr0 = policy_lr
        self.critic_lr0 = critic_lr

    # ------------------------------------------------------------------ #
    # Chargement
    # ------------------------------------------------------------------ #
    def _load_partial_sl(self, state_dict, oracle_init='kaiming',
                         oracle_scale=0.25):
        """Copie les poids SL dans la policy 283 canaux ; canaux oracle
        absents du SL initialisés en kaiming x scale (FIX ORACLE TAKEOFF
        de l'ancien pipeline : une init nulle laisse l'oracle muet)."""
        my_state = self.policy.state_dict()
        for k, v in state_dict.items():
            if k not in my_state:
                continue
            if my_state[k].shape == v.shape:
                my_state[k].copy_(v)
            elif k == 'trunk.input_conv.weight':
                new_w = adapt_in_channels(
                    v, my_state[k].shape[1],
                    oracle_init=oracle_init, oracle_scale=oracle_scale)
                my_state[k].copy_(new_w)
        self.policy.load_state_dict(my_state)

    def load_kl_ref(self, state_dict):
        self.ref_policy.load_state_dict(state_dict)
        self.ref_policy.eval()

    def oracle_weight_norm(self):
        w = self.policy.trunk.input_conv.weight
        return float(w[:, -ORACLE_DIM:, :, :].norm().item())

    # ------------------------------------------------------------------ #
    # Décisions BATCHÉES (collection)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act_rl(self, obs_n, obs_o, masks, gamma_t, deterministic=False):
        """Décisions du siège RL (discard), batchées.

        obs_n : (B, 279, 34) float32
        obs_o : (B, 4, 34) float32 (oracle COMPLET)
        masks : (B, 34) float32
        gamma_t : keep-prob du dropout oracle (1 = oracle complet)

        Dropout oracle PAR ÉLÉMENT (même granularité que l'ancien
        select_action : un masque Bernoulli indépendant par (canal, tuile)).
        L'oracle masqué est RETOURNÉ pour être stocké dans la transition :
        l'évaluation PPO doit revoir exactement la même entrée (ratio=1).

        Retour (actions, logps, values, oracle_masked) ; values en unités
        normalisées (critic cible = G_t / reward_scale).
        """
        on = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        oo = torch.as_tensor(obs_o, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(masks, dtype=torch.float32,
                                 device=self.device)
        B = on.shape[0]

        if gamma_t >= 1.0:
            oo_masked = oo
        elif gamma_t <= 0.0:
            oo_masked = torch.zeros_like(oo)
        else:
            dropout = torch.bernoulli(
                torch.full_like(oo, gamma_t))
            oo_masked = oo * dropout

        x = torch.cat([on, oo_masked], dim=1)
        logits = self.policy(x)
        masked_logits = logits + (mask_t - 1.0) * 1e9
        if deterministic:
            actions = masked_logits.argmax(dim=-1)
            logps = torch.log_softmax(masked_logits, dim=-1)\
                .gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            dist = torch.distributions.Categorical(logits=masked_logits)
            actions = dist.sample()
            logps = dist.log_prob(actions)

        # critic asymétrique : voit l'oracle COMPLET (le Critic squeeze
        # déjà sa dernière dim -> (B,))
        values = self.critic(torch.cat([on, oo], dim=1))

        return (actions.cpu().numpy().astype(np.int64),
                logps.cpu().numpy().astype(np.float32),
                values.cpu().numpy().astype(np.float32),
                oo_masked.cpu().numpy().astype(np.float16))

    @torch.no_grad()
    def act_sl_discard(self, obs_n, masks, sample=False):
        """Décisions de discard des adversaires SL figés, batchées."""
        on = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(masks, dtype=torch.float32,
                                 device=self.device)
        logits = self.sl_discard(on)
        masked = logits + (mask_t - 1.0) * 1e9
        if sample:
            probs = torch.softmax(masked, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(1)
        else:
            actions = masked.argmax(dim=-1)
        return actions.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def act_head(self, head, obs_n, masks):
        """Décisions des têtes SL figées (riichi/chow/pong/kong), batchées.

        Retourne des 0 (action sûre = passer / non) si la tête est absente,
        comme l'ancien _sl_decide en cas d'échec.
        """
        model = self.frozen_sl.get(head)
        B = len(obs_n)
        if model is None:
            return np.zeros(B, dtype=np.int64)
        on = torch.as_tensor(np.stack(obs_n), dtype=torch.float32,
                             device=self.device)
        mask_t = torch.as_tensor(np.stack(masks), dtype=torch.float32,
                                 device=self.device)
        logits = model(on)
        masked = logits + (mask_t - 1.0) * 1e9
        return masked.argmax(dim=-1).cpu().numpy().astype(np.int64)

    # ------------------------------------------------------------------ #
    # Évaluation PPO
    # ------------------------------------------------------------------ #
    def evaluate_actions(self, obs_n, obs_o_masked, obs_o_full, masks,
                         actions, kl_to_ref=False):
        """Log-probs / valeurs / entropie des actions sous la policy
        courante, pour la loss PPO.

        obs_o_masked : oracle TEL QUE VU À LA COLLECTION (dropout bake-in)
        -> PAS de nouveau dropout ici (l'ancien pipeline le faisait
        correctement, on conserve la sémantique).
        obs_o_full   : oracle complet, entrée du critic asymétrique.
        """
        on = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        oo = torch.as_tensor(obs_o_masked, dtype=torch.float32,
                             device=self.device)
        mask_t = torch.as_tensor(masks, dtype=torch.float32,
                                 device=self.device)
        a = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        x = torch.cat([on, oo], dim=1)
        logits = self.policy(x)
        masked_logits = logits + (mask_t - 1.0) * 1e9
        log_probs_all = torch.log_softmax(masked_logits, dim=-1)
        logp = log_probs_all.gather(1, a.unsqueeze(1)).squeeze(1)

        oof = torch.as_tensor(obs_o_full, dtype=torch.float32,
                              device=self.device)
        values = self.critic(torch.cat([on, oof], dim=1))

        probs = log_probs_all.exp()
        entropy = -(probs * log_probs_all).sum(dim=-1)

        kl = None
        if kl_to_ref:
            with torch.no_grad():
                ref_logits = self.ref_policy(x) + (mask_t - 1.0) * 1e9
                ref_lp = torch.log_softmax(ref_logits, dim=-1)
            kl = (probs * (log_probs_all - ref_lp)).sum(dim=-1).mean()

        return logp, values, entropy, kl

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #
    def save(self, path, step=0, alpha=0.0, phase=1, extra=None):
        ck = {
            # clés compatibles avec eval_stable_rank.py / export_rl_discard.py
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            # format enrichi
            'step': step,
            'alpha': alpha,
            'phase': phase,
            'config': {
                'channels': self.arch['channels'],
                'n_blocks': self.arch['n_blocks'],
                'in_channels': IN_CHANNELS_ORACLE,
                'oracle_dim': ORACLE_DIM,
                'n_tiles': N_TILES,
            },
            'optimizer_policy': self.policy_optim.state_dict(),
            'optimizer_critic': self.critic_optim.state_dict(),
        }
        if extra:
            ck.update(extra)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(ck, path)

    def load(self, path, load_optimizers=True):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck['policy'])
        self.critic.load_state_dict(ck['critic'])
        if load_optimizers and 'optimizer_policy' in ck:
            try:
                self.policy_optim.load_state_dict(ck['optimizer_policy'])
                self.critic_optim.load_state_dict(ck['optimizer_critic'])
            except Exception as e:
                print(f"[ppo_agent] optimizers non restaurés : {e}",
                      flush=True)
        self.policy.eval()
        self.critic.eval()
        return ck
