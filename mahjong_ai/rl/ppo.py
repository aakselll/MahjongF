"""Update PPO on-policy : minibatches, value clipping, entropie optionnelle.

Corrige les deux défauts algorithmiques fatals de l'ancien pipeline
(train_rl_suphx.py) :

1. OFF-POLICY : l'ancien code échantillonnait 20 parties dans un buffer
   FIFO de 150 générées par des poids publiés toutes les N steps, avec
   réutilisation illimitée. Le ratio d'importance était structurellement
   != 1 ("rdev" dans les logs), les avantages ne correspondaient à aucune
   policy cohérente, et le clip PPO annulait alors la plupart des
   gradients. Ici : les transitions viennent EXCLUSIVEMENT du batch
   collecté avec la policy courante -> ratio = 1 exactement au premier
   passage (assertion dans test_rl_rework.py).

2. BUDGET DE GRADIENT : l'ancien code faisait 3 pas de gradient (full
   batch, 3 epochs) par 1000 transitions consommées -> ~1500 pas sur un
   run de 500k steps. Ici : epochs x ceil(B/minibatch) pas de gradient
   par itération (ex. 4096 transitions, 4 epochs, minibatch 512 ->
   32 pas), soit ~30x plus d'optimisation pour le même volume de données.

La baseline des avantages suit le papier Suphx (batch mean) par défaut ;
le critic asymétrique (oracle complet) est disponible via --adv-mode
critic avec repli automatique batch-mean si sa variance expliquée (EV)
est insuffisante.
"""
import numpy as np
import torch


def compute_advantages(agent, transitions, mode='batch_mean',
                       reward_scale=10.0, min_ev=0.0):
    """A = G_t - baseline, puis normalisation (x-mean)/std.

    mode='batch_mean' (papier Suphx) : baseline = mean(G_t du batch).
    mode='critic' : baseline = V(s) du critic asymétrique (oracle complet)
    en unités de reward (le critic prédit G_t/reward_scale). Si la
    variance expliquée EV < min_ev, repli sur batch_mean (hérité v9 : un
    critic non encore entraîné ajoute du bruit au lieu d'en retirer).

    Retour (advantages, info) ; pose t['advantage'] sur chaque transition.

    [V2] info contient adv_var_raw : variance de l'avantage AVANT
    normalisation (unités reward^2). C'est la métrique qui mesure ce que
    la baseline retire au bruit : batch_mean -> Var(G) ; critic ->
    Var(G - V(s)). Chute de adv_var_raw = le critic absorbe la variance
    d'état (hidden info) que la policy ne contrôle pas.
    """
    returns = np.array([t['return'] for t in transitions], dtype=np.float32)
    values = None
    if mode == 'critic':
        with torch.no_grad():
            on = torch.as_tensor(
                np.stack([t['obs_normal'] for t in transitions]),
                dtype=torch.float32, device=agent.device)
            oof = torch.as_tensor(
                np.stack([t['obs_oracle_full'] for t in transitions]),
                dtype=torch.float32, device=agent.device)
            v = agent.critic(torch.cat([on, oof], dim=1))
        values = v.cpu().numpy().astype(np.float32) * reward_scale

    var_r = float(returns.var())
    ev, used_critic = 0.0, False
    if values is None or var_r < 1e-8:
        advs = returns - returns.mean()
    else:
        resid = returns - values
        ev = 1.0 - float(resid.var()) / max(var_r, 1e-8)
        if ev < min_ev:
            advs = returns - returns.mean()
        else:
            advs = resid
            used_critic = True

    adv_var_raw = float(advs.var()) if len(advs) > 1 else 0.0
    if len(advs) > 1 and advs.std() > 1e-6:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for i, tr in enumerate(transitions):
        tr['advantage'] = float(advs[i])
    info = {'ev': ev, 'used_critic': used_critic,
            'ret_mean': float(returns.mean()),
            'ret_std': float(returns.std()),
            'adv_var_raw': adv_var_raw}
    return advs, info


def ppo_update(agent, transitions, epochs=4, minibatch=512,
               clip_eps=0.2, value_clip=0.2, value_coef=0.5,
               entropy_coef=0.0, kl_coef=0.0, grad_clip=0.5,
               reward_scale=10.0, iw_threshold=None, warmup=False,
               lr_policy=None, lr_critic=None, target_kl=None):
    """Une itération PPO complète sur le batch collecté.

    warmup=True : seul le critic est optimisé (policy figée) — utilisé au
    démarrage en --adv-mode critic.
    lr_policy/lr_critic : surcharge ponctuelle du LR (phases).
    target_kl : [V2] early-stop PPO — si la KL moyenne old->new dépasse
    1.5 x target_kl, les epochs/minibatchs restants sont sautés. Avec la
    baseline critic le gradient par update est plus fort : sans garde-fou,
    4 epochs peuvent détruire en un batch ce que l'ancre KL reconstruit
    en 20 (cf. effondrement rl_nooracle_v1 : kl~+0.02 PAR update, 3856
    updates). 0.015 = standard PPO.

    Retour : dict de métriques ('early_stop' : True si coupé).
    """
    B = len(transitions)
    if B < 2:
        return None

    if lr_policy is not None:
        for pg in agent.policy_optim.param_groups:
            pg['lr'] = lr_policy
    if lr_critic is not None:
        for pg in agent.critic_optim.param_groups:
            pg['lr'] = lr_critic

    obs_n = np.stack([t['obs_normal'] for t in transitions])
    obs_o = np.stack([t['obs_oracle'] for t in transitions])
    obs_f = np.stack([t['obs_oracle_full'] for t in transitions])
    masks = np.stack([t['mask'] for t in transitions])
    actions = np.array([t['action'] for t in transitions], dtype=np.int64)
    old_logp = np.array([t['logp'] for t in transitions], dtype=np.float32)
    old_value = np.array([t['value'] for t in transitions],
                         dtype=np.float32)          # unités normalisées
    returns = np.array([t['return'] for t in transitions],
                       dtype=np.float32)            # unités reward
    advs = np.array([t['advantage'] for t in transitions],
                    dtype=np.float32)

    mb = max(8, min(minibatch, B))
    metrics = {
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'ratio_dev': [], 'approx_kl': [], 'clip_frac': [],
        'pg_norm': [], 'cg_norm': [],
    }
    use_kl = kl_coef > 0
    early_stop = False

    for epoch in range(epochs):
        if early_stop:
            break
        perm = np.random.permutation(B)
        for s in range(0, B, mb):
            idx = perm[s:s + mb]
            on = torch.as_tensor(obs_n[idx], dtype=torch.float32,
                                 device=agent.device)
            oo = torch.as_tensor(obs_o[idx], dtype=torch.float32,
                                 device=agent.device)
            of = torch.as_tensor(obs_f[idx], dtype=torch.float32,
                                 device=agent.device)
            m = torch.as_tensor(masks[idx], dtype=torch.float32,
                                device=agent.device)
            a = torch.as_tensor(actions[idx], dtype=torch.long,
                                device=agent.device)
            olp = torch.as_tensor(old_logp[idx], dtype=torch.float32,
                                  device=agent.device)
            ov = torch.as_tensor(old_value[idx], dtype=torch.float32,
                                 device=agent.device)
            ret = torch.as_tensor(returns[idx], dtype=torch.float32,
                                  device=agent.device) / reward_scale
            adv = torch.as_tensor(advs[idx], dtype=torch.float32,
                                  device=agent.device)

            logp, values, entropy, kl = agent.evaluate_actions(
                on, oo, of, m, a, kl_to_ref=use_kl)

            ratio = torch.exp(logp - olp)
            if epoch == 0 and s == 0:
                metrics['ratio_dev'].append(
                    float((ratio.detach() - 1.0).abs().mean().item()))
            with torch.no_grad():
                metrics['approx_kl'].append(
                    float((olp - logp).mean().item()))
                metrics['clip_frac'].append(float(
                    ((ratio - 1.0).abs() > clip_eps).float().mean().item()))

            # ---- IW rejection (phase 3, papier §3.3) ----
            if iw_threshold is not None:
                keep = (ratio <= iw_threshold).float()
                if keep.sum().item() < 2:
                    continue
                keep = keep / keep.mean()
            else:
                keep = torch.ones_like(ratio)

            # ---- policy loss (clipped surrogate) ----
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -(torch.min(s1, s2) * keep).mean()

            # ---- value loss avec clipping (PPO standard) ----
            v_clipped = ov + torch.clamp(values - ov, -value_clip,
                                         value_clip)
            vl1 = (values - ret) ** 2
            vl2 = (v_clipped - ret) ** 2
            value_loss = 0.5 * torch.max(vl1, vl2).mean()

            # ---- entropie / KL ancre ----
            ent = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * ent
            if use_kl and kl is not None:
                loss = loss + kl_coef * kl

            agent.policy_optim.zero_grad()
            agent.critic_optim.zero_grad()
            loss.backward()
            pg = torch.nn.utils.clip_grad_norm_(
                agent.policy.parameters(), grad_clip)
            cg = torch.nn.utils.clip_grad_norm_(
                agent.critic.parameters(), grad_clip)
            if not warmup:
                agent.policy_optim.step()
            agent.critic_optim.step()

            metrics['policy_loss'].append(float(policy_loss.item()))
            metrics['value_loss'].append(float(value_loss.item()))
            metrics['entropy'].append(float(ent.item()))
            metrics['pg_norm'].append(float(pg))
            metrics['cg_norm'].append(float(cg))

            # ---- [V2] early-stop KL (garde-fou anti-dérive) ----
            if target_kl is not None and metrics['approx_kl']:
                if float(np.mean(metrics['approx_kl'])) > 1.5 * target_kl:
                    early_stop = True
                    break

    out = {k: (float(np.mean(v)) if v else None)
           for k, v in metrics.items()}
    out['n_grad_steps'] = len(metrics['policy_loss'])
    out['batch_size'] = B
    out['early_stop'] = early_stop
    return out


# ---------------------------------------------------------------------- #
# [V2.1] Entraînement du critic sur fenêtre de replay
# ---------------------------------------------------------------------- #
# Constat du run no-oracle v2 (checkpoints/rl_new, 225k steps) : le critic
# ne généralisait JAMAIS. Chaque update il recevait 16 pas de gradient sur
# le batch courant (~186 cibles distinctes : G_t est constant par manche),
# puis le batch était jeté. Résultat mesuré dans les logs :
#     vl ~ 0.17-0.29  = 0.5 * Var(G_t) / reward_scale^2  (bande du
#                       prédicteur-constant : le critic prédit ~la moyenne)
#     ev ~ 0.000 en out-of-sample (mesuré sur le batch AVANT de l'entraîner
#     dessus), parfois négatif ; repli batch_mean sur ~65 pour-cent des
#     updates -> le run tournait en "v1 + laisse KL" (stable, n'apprend pas).
#
# Correctif : le critic est une régression supervisée, PAS un composant
# on-policy. Seul le RATIO PPO exige l'on-policy (il reste calculé sur le
# batch courant exclusivement) ; la baseline peut être apprise sur des
# données légèrement stales sans biais (Pinto & al. 2018, Robust Adversarial
# RL : même asymétrie entraînement/déploiement). Avec l'ancre KL la policy
# dérive de ~0.003 KL par update : une fenêtre de 10 batches (~21k steps de
# dérive policy) reste représentative.
class CriticReplay:
    """Fenêtre FIFO des K derniers batches, pour l'entraînement du critic.

    Stocke par batch : obs normale (f16), oracle complet (f16), retour
    G_t (f32, unités reward). ~19 Ko par transition -> 10 batches de
    ~2100 transitions ~ 400 Mo en float16.
    """

    def __init__(self, capacity_batches=10, seed=0):
        self.capacity = max(0, int(capacity_batches))
        self._obs_n = []     # list[np.ndarray (B, 279, 34) f16]
        self._obs_of = []    # list[np.ndarray (B, 4, 34) f16]
        self._rets = []      # list[np.ndarray (B,) f32]
        self._n = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self._n

    @property
    def n_batches(self):
        return len(self._rets)

    def push(self, transitions):
        """Ajoute un batch (celui qui vient servi l'update PPO)."""
        if not transitions or self.capacity == 0:
            return
        self._obs_n.append(np.stack(
            [t['obs_normal'] for t in transitions]).astype(np.float16))
        self._obs_of.append(np.stack(
            [t['obs_oracle_full'] for t in transitions]).astype(np.float16))
        self._rets.append(np.array(
            [t['return'] for t in transitions], dtype=np.float32))
        self._n += len(transitions)
        if len(self._rets) > self.capacity:
            dropped = len(self._rets[0])
            self._obs_n.pop(0)
            self._obs_of.pop(0)
            self._rets.pop(0)
            self._n -= dropped

    def sample_minibatch(self, size, skip_newest=0):
        """Tire `size` transitions uniformément dans la fenêtre (avec
        remise). Retour (obs_n, obs_of, returns) ou None si vide.

        [V2.2] skip_newest > 0 : exclut les k batches les plus récents
        (réservés au jeu de validation du critic_fit).
        """
        if self._n == 0 or size <= 0:
            return None
        k = max(0, min(int(skip_newest), len(self._rets) - 1))
        bounds = np.cumsum([len(r) for r in self._rets])
        n_avail = self._n if k == 0 else (bounds[-(k + 1)]
                                          if len(bounds) > k else bounds[0])
        if n_avail <= 0:
            return None
        flat = self.rng.integers(0, n_avail, size=int(size))
        bi = np.searchsorted(bounds, flat, side='right')
        # pour bi==0 la branche bounds[-1] est écartée par le where
        lo = np.where(bi > 0, bounds[np.maximum(bi - 1, 0)], 0)
        local = flat - lo
        obs_n = np.empty(
            (len(flat),) + self._obs_n[0].shape[1:], dtype=np.float16)
        obs_of = np.empty(
            (len(flat),) + self._obs_of[0].shape[1:], dtype=np.float16)
        rets = np.empty(len(flat), dtype=np.float32)
        for b in np.unique(bi):
            m = bi == b
            li = local[m]
            obs_n[m] = self._obs_n[b][li]
            obs_of[m] = self._obs_of[b][li]
            rets[m] = self._rets[b][li]
        return obs_n, obs_of, rets


def critic_fit(agent, replay, n_steps=48, minibatch=512, reward_scale=10.0,
               lr=None, weight_decay=0.0, val_batches=2, patience=3):
    """Pas de gradient pour le critic SEUL, sur la fenêtre de replay.

    MSE pure ; le policy n'est pas touché (aucun forward policy).

    [V2.2] Régularisation, réponse au run v2.1 (checkpoints/rl_v21,
    265k steps) : la fenêtre de 10 batches a réglé « le critic ne voit
    jamais deux fois les mêmes données » mais produit l'échec INVERSE :
    cfit 0.104 -> 0.024 (le critic mémorise la fenêtre, ~1900 cibles
    distinctes seulement, G_t constant par manche) pendant que l'EV sur
    batch frais passait de +0.000 a -0.44 : prédictions CONFIANTES mais
    non corrélées -> Var(A) = Var(G) + Var(V), la porte min_ev rejetait
    le critic a 122 updates sur 122. Correctifs :
      * weight_decay : L2 dans la loss (découplé de l'optimizer PPO) ;
      * validation sur les `val_batches` batches les plus RÉCENTS de la
        fenêtre — les plus proches de la distribution du batch frais,
        celle-là même que mesure la porte EV ;
      * early-stop : le fit s'arrête si la val remonte `patience` fois
        de suite (l'optimum de généralisation est derrière nous).

    Duo de diagnostic cfit/cval dans les logs DRIFT :
        cfit << cval           -> mémorisation (monter wd, couper plus tôt)
        cfit ~ cval ~ Var/2    -> sous-entraîné (plus de steps)
        cfit ~ cval, ev > 0.1  -> sain : le critic généralise.

    Retour {'loss', 'val_loss', 'n_steps', 'stopped_at', 'window'} —
    val_loss/stopped_at None si pas de split possible (fenêtre < 2
    batches) ; loss en unités normalisées (comparable a vl).
    """
    if replay is None or len(replay) == 0 or n_steps <= 0:
        return None
    if lr is not None:
        for pg in agent.critic_optim.param_groups:
            pg['lr'] = lr

    n_val = min(int(val_batches), max(0, replay.n_batches - 1))

    def _val_loss():
        """MSE (0.5) moyenne sur les n_val batches les plus récents."""
        if n_val <= 0:
            return None
        tot, w = 0.0, 0
        with torch.no_grad():
            for an, ao, ar in zip(replay._obs_n[-n_val:],
                                  replay._obs_of[-n_val:],
                                  replay._rets[-n_val:]):
                on = torch.as_tensor(an, dtype=torch.float32,
                                      device=agent.device)
                oof = torch.as_tensor(ao, dtype=torch.float32,
                                       device=agent.device)
                ret = torch.as_tensor(ar, dtype=torch.float32,
                                       device=agent.device) / reward_scale
                v = agent.critic(torch.cat([on, oof], dim=1))
                tot += float(0.5 * ((v - ret) ** 2).mean().item()) * len(ar)
                w += len(ar)
        return tot / max(w, 1)

    mb = max(8, int(minibatch))
    losses = []
    best_val, since_best, stopped_at = None, 0, None
    check_every = max(4, int(n_steps) // 12)
    for it in range(int(n_steps)):
        batch = replay.sample_minibatch(mb, skip_newest=n_val)
        if batch is None:
            break
        obs_n, obs_of, rets = batch
        on = torch.as_tensor(obs_n, dtype=torch.float32,
                              device=agent.device)
        oof = torch.as_tensor(obs_of, dtype=torch.float32,
                               device=agent.device)
        ret = torch.as_tensor(rets, dtype=torch.float32,
                               device=agent.device) / reward_scale
        v = agent.critic(torch.cat([on, oof], dim=1))
        loss = 0.5 * ((v - ret) ** 2).mean()
        if weight_decay > 0:
            l2 = sum(torch.sum(p * p) for p in agent.critic.parameters())
            loss = loss + weight_decay * l2
        agent.critic_optim.zero_grad()
        loss.backward()
        agent.critic_optim.step()
        losses.append(float(loss.item()))

        if n_val > 0 and (it + 1) % check_every == 0:
            vnow = _val_loss()
            if vnow is None:
                continue
            if best_val is None or vnow < best_val - 1e-6:
                best_val, since_best = vnow, 0
            else:
                since_best += 1
                if since_best >= max(1, int(patience)):
                    stopped_at = it + 1
                    break

    if not losses:
        return None
    val_final = _val_loss() if n_val > 0 else None
    return {'loss': float(np.mean(losses)), 'val_loss': val_final,
            'n_steps': len(losses), 'stopped_at': stopped_at,
            'window': len(replay)}
