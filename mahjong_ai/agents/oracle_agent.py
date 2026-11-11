"""
Agent RL Suphx : policy (discard) + critic + oracle channels.

Conforme au papier §3.3 (oracle guiding) :
    - input : [x_n(s) ; δ_t · x_o(s)]  (386 canaux)
    - δ_t ~ Bernoulli(γ_t) sur les 136 canaux oracle
    - γ_t décroît de 1 à 0 sur gamma_decay_steps

Une fois γ=0, l'agent devient "normal" (250 canaux effectifs, les 136
autres à zéro). C'est ce qui permet de déployer l'agent sans info parfaite.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import os
from mahjong_ai.agents.suphx_cnn import (
    build_model, N_CHANNELS, ORACLE_DIM, IN_CHANNELS_ORACLE,
    adapt_in_channels,
)


class Critic(nn.Module):
    """[FIX CRITIC CONV] Ancien critic = MLP sur vecteur aplati
    (9622 -> 512 -> 256 -> 1) : aucune notion de structure des 34 tuiles,
    il convergait vers une prediction quasi constante (FLAT_VALUE :
    val_std ~0.06, EV 1-5 %) et l'avantage A = R - V(s) degenerait en
    R - moyenne, la meme pathologie que --adv-mode batch_mean.

    Nouveau critic conv, miroir reduit du tronc de la policy : blocs
    residuels Conv (3,1) comme le SL (pas de normalisation : le critic
    reste TOUJOURS en mode eval, la BN y serait une operation figee),
    puis aggregation moyenne+max par canal sur les 34 tuiles et tete
    lineaire. ~160k parametres (contre ~5M pour l'ancien MLP), cout de
    forward comparable, bien meilleure hypothese de structure.
    """
    def __init__(self, in_channels=IN_CHANNELS_ORACLE, channels=64,
                 n_blocks=4):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, channels, (3, 1),
                                    padding=(1, 0), bias=True)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (3, 1), padding=(1, 0),
                          bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, (3, 1), padding=(1, 0),
                          bias=True),
            ) for _ in range(n_blocks)
        ])
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        # Tete finale initialisee a zero : V(s) = 0 exactement au depart,
        # l'ecart de valeur ne peut apparaitre que s'il reduit la loss.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # [COMPAT] Ne charge que les cles du critic conv dont la forme
        # correspond ; ignore l'ancien critic MLP (net.*) et les anciens
        # formats de canaux. Un --resume d'un run anterieur fait alors
        # repartir le critic de son initialisation (apres le warmup il
        # est redeja bon) au lieu de crasher le load strict.
        local = self.state_dict()
        for key in [k for k in state_dict if k.startswith(prefix)]:
            name = key[len(prefix):]
            if name not in local or local[name].shape != state_dict[key].shape:
                del state_dict[key]
        for name, param in local.items():
            state_dict.setdefault(prefix + name, param)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        if x.dim() == 3:                      # (B, C, 34) -> (B, C, 34, 1)
            x = x.unsqueeze(-1)
        h = self.act(self.input_conv(x))
        for block in self.blocks:
            h = h + block(h)                  # residuel
        h = self.act(h).squeeze(-1)           # (B, C, 34)
        pooled = torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=1)
        return self.head(pooled).squeeze(-1)


class OracleDiscardAgent:
    """Policy (discard) + critic + SL figés (riichi/chow/pong/kong).

    [FIX 2026-10-14] L'architecture est lue depuis la config sauvegardee
    dans chaque checkpoint (train_policy_sl.py stocke {'config': ...})
    au lieu des defauts de build_model. Sans ca :
    - les tetes legeres (128 canaux / 6 blocs) ne chargent pas ;
    - la policy RL (defaut 16 blocs) ne warm-start que partiellement
      depuis un SL a 12 blocs -- les blocs 12-15 restent aleatoires.
    """
    def __init__(self, sl_discard_path, sl_frozen_dir, device='cpu',
                 policy_lr=1e-4, critic_lr=1e-3,
                 oracle_init='kaiming', oracle_scale=0.25):
        self.device = device

        # --- Policy : archi lue depuis le checkpoint SL discard ---
        ck = torch.load(sl_discard_path, map_location=device,
                        weights_only=False)
        cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
        channels = cfg.get('channels', 256)
        n_blocks = cfg.get('n_blocks', 16)

        self.policy = build_model('discard', use_oracle=True,
                                  channels=channels,
                                  n_blocks=n_blocks).to(device)
        self._load_partial_sl(ck['model_state_dict'],
                              oracle_init=oracle_init,
                              oracle_scale=oracle_scale)
        # [FIX ORACLE TAKEOFF] norme de reference des colonnes oracle pour
        # monitorer la croissance (log du trainer).
        with torch.no_grad():
            self.oracle_init_norm = self.oracle_weight_norm()
        print(f"[oracle_agent] discard SL charge "
              f"(val_acc={ck.get('val_acc','?')}, "
              f"arch={channels}x{n_blocks}, "
              f"oracle_init={oracle_init}x{oracle_scale} "
              f"||W_o||={self.oracle_init_norm:.4f})", flush=True)

        # [ANCRE KL] Copie GELEE de la policy SL de depart. Sert de
        # reference pour la penalite KL(pi || pi_SL) : sans ancrage le
        # self-play symetrique derive vers des politiques passives (run
        # 15M : 38 pourcent d'exhaustive draws, rang moyen 2.68 vs 2.50
        # pour le SL). Construite ICI (depuis le SL), donc inchangee par un
        # --resume qui ne charge que les poids de self.policy.
        self.ref_policy = copy.deepcopy(self.policy).eval()
        for _p in self.ref_policy.parameters():
            _p.requires_grad = False
        self.compute_kl = False       # active par la boucle (--kl-coef > 0)
        self.last_kl = None
        # [DANGER LOSS] E_pi[danger] (B,) avec gradient, et KL d'ablation
        # oracle (diagnostic : 0 = la policy ignore l'oracle)
        self.last_p_danger = None
        self.last_abl_kl = None

        # --- Critic ---
        self.critic = Critic().to(device)

        # --- SL figes : archi lue depuis CHAQUE checkpoint ---
        self.frozen_sl = {}
        for head in ['riichi', 'chow', 'pong', 'kong']:
            path = os.path.join(sl_frozen_dir, f'{head}.pt')
            try:
                hck = torch.load(path, map_location=device,
                                 weights_only=False)
                hcfg = hck.get('config', {}) if isinstance(hck, dict) else {}
                m = build_model(
                    head,
                    channels=hcfg.get('channels', 256),
                    n_blocks=hcfg.get('n_blocks', 16),
                    binary_hidden=hcfg.get('binary_hidden', 256),
                    # [ROUGES] toujours N_CHANNELS : un ancien checkpoint a
                    # 268 canaux est charge avec des poids nuls pour les
                    # 11 canaux rouges (comportement inchange).
                    in_channels=N_CHANNELS,
                ).to(device)
                m.load_state_dict(hck['model_state_dict'])
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                self.frozen_sl[head] = m
                print(f"[oracle_agent] {head} charge "
                      f"(arch={hcfg.get('channels','?')}x"
                      f"{hcfg.get('n_blocks','?')})", flush=True)
            except Exception as e:
                print(f"[oracle_agent] {head} non charge : {e}", flush=True)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(),
                                             lr=policy_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)

        # [FIX BATCHNORM] La policy contient des BatchNorm2d (tronc + 16
        # blocs). Un module fraichement construit est en mode train : en
        # rollout (select_action, batch de 1) la BN normaliserait alors
        # chaque canal sur les seuls 34 emplacements de l'etat courant au
        # lieu d'utiliser les stats du SL, et evaluate_actions (batch de
        # ~1000) utiliserait d'autres stats encore -> la policy qui joue
        # n'est pas celle du checkpoint SL et le ratio PPO est du bruit.
        # En eval, les stats SL sont figees (les gradients, eux, passent
        # normalement, et gamma/beta de la BN restent entrainables).
        self.policy.eval()

    def set_eval_mode(self):
        """Remet la policy en eval (BN figee). A rappeler apres tout
        load_state_dict / resume : load_state_dict ne change pas le mode,
        mais un .train() externe l'aurait change."""
        self.policy.eval()
        self.critic.eval()

    def load_kl_ref(self, state_dict):
        """[ANCRE KL] recharge la reference KL depuis un checkpoint.

        Par defaut ref_policy = copie du SL de depart (ses canaux oracle
        sont ceux de --oracle-init : zero = aveugle a l'oracle). En
        phase 2/3, ancrer plutot la policy de phase 1 (oracle-aware) via
        --kl-ref : sinon la penalite KL re-importe la pression
        anti-oracle du SL qui a fait chuter ||W_o|| en v10.
        """
        self.ref_policy.load_state_dict(state_dict)
        self.ref_policy.eval()
        with torch.no_grad():
            wo = self.ref_policy.trunk.input_conv.weight[
                :, -ORACLE_DIM:, :, :].norm().item()
        print(f"[oracle_agent] ancre KL rechargee : ||W_o||_ref={wo:.4f} "
              f"(0.0 = ancre aveugle a l'oracle)", flush=True)

    def oracle_weight_norm(self):
        """Norme L2 des poids input_conv des ORACLE_DIM canaux oracle.
        Indicateur d'usage de l'info cachee : reste a la valeur d'init si
        la policy n'apprend pas a s'en servir."""
        w = self.policy.trunk.input_conv.weight
        # [FIX] derniers ORACLE_DIM canaux seulement : N_CHANNELS: incluait
        # aussi les canaux ROUGES (init a zero) dans la norme.
        return float(w[:, -ORACLE_DIM:, :, :].norm().item())

    def _load_partial_sl(self, state_dict, oracle_init='kaiming',
                         oracle_scale=0.25):
        """Charge partiellement : le trunk (conv_in) attend 386 canaux mais
        le SL en a 250. On étend les poids d'entrée en initialisant les
        canaux oracle à 0.

        [FIX ORACLE TAKEOFF] les canaux oracle sont initialises en Kaiming
        x oracle_scale (defaut 0.25) au lieu de zero : a poids nuls
        l'oracle n'influence pas les logits et doit croitre par pas PG
        bruites -- trop lent a l'echelle du run (cf. oracle_weight_norm).
        Les canaux ROUGES restent a zero (comportement initial == SL sur
        la partie visible).
        """
        my_state = self.policy.state_dict()
        for k, v in state_dict.items():
            if k not in my_state:
                continue
            if my_state[k].shape == v.shape:
                my_state[k].copy_(v)
            elif k == 'trunk.input_conv.weight':
                # [ROUGES] (256, 283, 3, 1) attendu ; le SL a 268 (ancien)
                # ou 279 (avec rouges) canaux : blocs absents = zeros,
                # SAUF oracle (kaiming x scale, cf. [FIX ORACLE TAKEOFF]).
                new_w = adapt_in_channels(v, my_state[k].shape[1],
                                          oracle_init=oracle_init,
                                          oracle_scale=oracle_scale)
                my_state[k].copy_(new_w)
                print(f"[oracle_agent] {k} etendu de {v.shape} a {new_w.shape} "
                      f"(oracle_init={oracle_init}x{oracle_scale})")
            # Les autres mismatch sont ignores (rare)
        self.policy.load_state_dict(my_state)

    # ------------------------------------------------------------------ #
    def forward_with_dropout(self, obs_normal, obs_oracle, gamma_t):
        """Construit l'input avec dropout oracle et forward.

        obs_normal : (B, N_CHANNELS, 34)
        obs_oracle : (B, ORACLE_DIM, 34)
        gamma_t    : float dans [0, 1]

        Retourne (logits_discard, value, oracle_mask).
        """
        B = obs_normal.shape[0]
        # Mask Bernoulli sur les canaux oracle
        if gamma_t <= 0:
            delta = torch.zeros(B, ORACLE_DIM, 1, device=self.device)
        elif gamma_t >= 1:
            delta = torch.ones(B, ORACLE_DIM, 1, device=self.device)
        else:
            delta = (torch.rand(B, ORACLE_DIM, 1, device=self.device)
                     < gamma_t).float()
        oracle_masked = obs_oracle * delta
        obs_full = torch.cat([obs_normal, oracle_masked], dim=1)  # (B, 386, 34)
        logits = self.policy(obs_full)
        value = self.critic(obs_full)
        return logits, value, delta

    # ------------------------------------------------------------------ #
    def select_action(self, obs_normal, obs_oracle, mask, gamma_t):
        """Retourne (action, logp, value, masked_obs_oracle).
        
        [FIX ORACLE GUIDING] Le masque Bernoulli appliqué à l'oracle fait
        partie de l'ETAT vu par la policy. Pour que l'importance sampling
        soit correct (ratio = 1 quand les poids n'ont pas change), il faut
        que l'evaluation utilise LE MEME masque que la generation.
        
        On retourne donc l'oracle MASQUE pour qu'il soit stocke dans la
        transition et reutilise tel quel pendant l'entrainement.
        """
        on = torch.as_tensor(obs_normal, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        oo = torch.as_tensor(obs_oracle, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

        # Appliquer le dropout Bernoulli per-element
        if gamma_t < 1.0:
            dropout = torch.bernoulli(
                torch.full_like(oo, gamma_t))
            oo_masked = oo * dropout
        else:
            oo_masked = oo

        x = torch.cat([on, oo_masked], dim=1)
        with torch.no_grad():
            logits = self.policy(x)
            masked_logits = logits + (mask_t - 1.0) * 1e9
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            
            # [CRITIC ASYMETRIQUE] le critic voit l'oracle COMPLET (non
            # masque) : il sert de baseline V(s) et n'est pas utilise au
            # deploiement, donc il peut tricher avec l'info cachee.
            value = self.critic(torch.cat([on, oo], dim=1)).item()

        # [FIX] Retourner l'oracle MASQUE (pas le brut)
        masked_obs_oracle = oo_masked.squeeze(0).cpu().numpy()
        
        return int(action.item()), logp.item(), value, masked_obs_oracle

    # ------------------------------------------------------------------ #
    def select_action_sl(self, obs_normal, mask, argmax=True):
        """Action de la policy SL GELEE (ref_policy), pour les adversaires.

        Entree = obs normale + canaux oracle a zero (le SL n'a jamais vu
        l'oracle : ses poids oracle sont a zero). Aucune transition n'est
        enregistree pour ces sieges.
        argmax=True : meme protocole que l'eval ; False : echantillonnage.
        """
        on = torch.as_tensor(obs_normal, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        oo = torch.zeros(1, ORACLE_DIM, on.shape[-1], dtype=torch.float32,
                         device=self.device)
        x = torch.cat([on, oo], dim=1)
        with torch.no_grad():
            logits = self.ref_policy(x) + (mask_t - 1.0) * 1e9
            if argmax:
                return int(logits.argmax(dim=1).item())
            return int(torch.distributions.Categorical(
                logits=logits).sample().item())

    # ------------------------------------------------------------------ #
    def evaluate_actions(self, obs_n, obs_o, masks, actions, gamma_t,
                         obs_o_full=None, danger=None, ablate=False):
        """Evalue les log-probs des actions sous la policy courante.
        
        [FIX ORACLE GUIDING] obs_o est DEJA MASQUE (stocke depuis la
        generation). On ne re-applique PAS de nouveau dropout -- sinon le
        ratio d'importance sampling serait faux (memes poids != ratio 1
        car les entrees different).
        
        gamma_t est garde en parametre pour compatibilite mais IGNORE :
        le masque est deja "bake in" dans obs_o.
        """
        on = torch.as_tensor(obs_n, dtype=torch.float32,
                            device=self.device)
        oo = torch.as_tensor(obs_o, dtype=torch.float32,
                            device=self.device)
        # [FIX] PAS de nouveau dropout -- obs_o est deja masque
        x = torch.cat([on, oo], dim=1)
        
        logits = self.policy(x)
        mask_t = torch.as_tensor(masks, dtype=torch.float32,
                                device=self.device)
        masked_logits = logits + (mask_t - 1.0) * 1e9
        dist = torch.distributions.Categorical(logits=masked_logits)
        logp = dist.log_prob(actions)
        
        # [CRITIC ASYMETRIQUE] si l'oracle complet est fourni, le critic
        # l'utilise (au lieu de l'oracle masque par le dropout).
        if obs_o_full is not None:
            oof = torch.as_tensor(obs_o_full, dtype=torch.float32,
                                  device=self.device)
            values = self.critic(torch.cat([on, oof], dim=1))
        else:
            values = self.critic(x)
        # Entropie sur les actions valides
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)

        # [DANGER LOSS] E_pi[danger(t)] : probabilite que la policy defausse
        # une tuile qui ferait ron. Esperance EXACTE (pas d'echantillonnage),
        # gradient direct vers les logits -> seule une policy qui LIT
        # l'oracle (mains adverses) peut la reduire sans perdre ailleurs.
        self.last_p_danger = None
        if danger is not None:
            d = torch.as_tensor(danger, dtype=torch.float32,
                                device=self.device)
            self.last_p_danger = (probs * d).sum(dim=-1)

        # [DIAG ABLATION] KL(pi_oracle || pi_oracle=0) sur ce batch.
        # ~0 => la policy n'utilise pas l'oracle.
        if ablate:
            with torch.no_grad():
                x0 = torch.cat([on, torch.zeros_like(oo)], dim=1)
                lg0 = self.policy(x0) + (mask_t - 1.0) * 1e9
                lp0 = torch.log_softmax(lg0, dim=-1)
                self.last_abl_kl = float(
                    (probs.detach() * (log_probs.detach() - lp0))
                    .sum(dim=-1).mean().item())

        # [ANCRE KL] KL(pi || pi_SL) moyenne sur les etats du batch, avec
        # gradient vers pi. Stockee dans self.last_kl (la signature de
        # retour est inchangee). Memes etats et memes masques que pi.
        if self.compute_kl:
            with torch.no_grad():
                ref_logits = self.ref_policy(x) + (mask_t - 1.0) * 1e9
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            self.last_kl = (probs * (log_probs - ref_log_probs)
                            ).sum(dim=-1).mean()
        else:
            self.last_kl = None

        return logp, values.squeeze(-1), entropy

    # ------------------------------------------------------------------ #
    def save(self, path, step=None, alpha=None):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'step': step,
            'alpha': alpha,          # <-- AJOUT
        }, path)