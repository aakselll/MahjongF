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

    if len(advs) > 1 and advs.std() > 1e-6:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for i, tr in enumerate(transitions):
        tr['advantage'] = float(advs[i])
    info = {'ev': ev, 'used_critic': used_critic,
            'ret_mean': float(returns.mean()),
            'ret_std': float(returns.std())}
    return advs, info


def ppo_update(agent, transitions, epochs=4, minibatch=512,
               clip_eps=0.2, value_clip=0.2, value_coef=0.5,
               entropy_coef=0.0, kl_coef=0.0, grad_clip=0.5,
               reward_scale=10.0, iw_threshold=None, warmup=False,
               lr_policy=None, lr_critic=None):
    """Une itération PPO complète sur le batch collecté.

    warmup=True : seul le critic est optimisé (policy figée) — utilisé au
    démarrage en --adv-mode critic.
    lr_policy/lr_critic : surcharge ponctuelle du LR (phases).

    Retour : dict de métriques.
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

    for epoch in range(epochs):
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

    out = {k: (float(np.mean(v)) if v else None)
           for k, v in metrics.items()}
    out['n_grad_steps'] = len(metrics['policy_loss'])
    out['batch_size'] = B
    return out
