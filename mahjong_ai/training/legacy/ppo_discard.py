"""
Boucle PPO pour entrainer le modele discard Suphx en RL-1 (avec GRP).

Reference : Suphx paper section 3.1 (policy gradient + entropy reg) et
section 4.2 (variante RL-1 : reward GRP, autres tetes figees SL).

Usage :
    py -3.14 -m mahjong_ai.training.ppo_discard ^
        --sl-checkpoint checkpoints/sl_2015_run1/discard.pt ^
        --sl-frozen-dir checkpoints/sl_2015_run1 ^
        --grp-checkpoint checkpoints/grp_sl.pt ^
        --total-steps 200000 ^
        --output checkpoints/rl_discard.pt
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from mahjong_ai.agents.rl_discard_agent import RLDiscardAgent
from mahjong_ai.training.game_env import GameEnv, PredictorHolder, GameLogBuffer
from mahjong_ai.envs.multi_agent import DISCARD
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM


# ---------------------------------------------------------------------- #
# Collecte de trajectoires
# ---------------------------------------------------------------------- #
def collect_one_episode(env, agent, max_rounds=8):
    """Joue UNE partie complete et retourne les transitions pour les
    4 sieges (self-play classique).

    Chaque transition est stockee avec SON siege, et le reward de la manche
    est attribue a la transition du bon siege.
    """
    transitions = []
    # Reward en attente par siege (dict)
    pending_reward = {s: 0.0 for s in range(4)}

    req = env.reset_round()
    while True:
        seat = req['seat']
        action = None
        logp = value = 0.0
        obs = mask = None

        if req['action_type'] == DISCARD:
            obs = req['obs']
            mask = req['mask']
            action, logp, value, _ = agent.select_action(obs, mask)

        result = env.step(action)

        if action is not None:
            # Stocke la transition avec le siege courant
            transitions.append({
                'seat': seat,
                'obs': obs, 'mask': mask, 'action': action,
                'logp': logp, 'value': value,
                'reward': pending_reward[seat],   # reward deja accumule
                'done': False,
            })
            pending_reward[seat] = 0.0            # consomme le pending

        if result['round_done']:
            round_rewards = result['final_rewards']   # dict {seat: float}

            # Distribue les rewards de la manche aux 4 sieges
            # (chaque siege accumule pour sa PROCHAINE transition discard)
            for s in range(4):
                pending_reward[s] += round_rewards.get(s, 0.0)

            if result.get('next_request') is None:
                # Fin de partie : flush les pendings restants
                for t in transitions:
                    s = t['seat']
                    if pending_reward[s] != 0.0:
                        t['reward'] += pending_reward[s]
                        pending_reward[s] = 0.0
                break
            req = env.reset_round()
        else:
            req = result['next_request']

    if transitions:
        transitions[-1]['done'] = True
    return transitions


# ---------------------------------------------------------------------- #
# Calcul des advantages (GAE)
# ---------------------------------------------------------------------- #
def compute_gae(transitions, gamma=0.99, lam=0.95):
    """GAE sur une trajectoire. Ajoute 'advantage' et 'return' a chaque trans."""
    advantages = []
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(transitions))):
        tr = transitions[t]
        if tr['done']:
            next_value = 0.0
            delta = tr['reward'] - tr['value']
            gae = delta
        else:
            delta = tr['reward'] + gamma * next_value - tr['value']
            gae = delta + gamma * lam * gae
        tr['advantage'] = gae
        tr['return'] = gae + tr['value']
        next_value = tr['value']
        advantages.append(gae)

    advantages = np.array(advantages[::-1], dtype=np.float32)
    # Normalise les advantages (reduit la variance)
    if len(advantages) > 1 and advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


# ---------------------------------------------------------------------- #
# PPO update
# ---------------------------------------------------------------------- #
def ppo_update(agent, transitions, advantages, epochs, batch_size,
                clip_eps, entropy_coef, value_coef):
    """Met a jour policy et critic sur les transitions collectees.

    Eq. 2 du papier : L = L_PG + c_v * L_VF - c_h * H(pi)
    """
    n = len(transitions)
    obs = torch.as_tensor(
        np.stack([t['obs'] for t in transitions]), dtype=torch.float32,
        device=agent.device)
    masks = torch.as_tensor(
        np.stack([t['mask'] for t in transitions]), dtype=torch.float32,
        device=agent.device)
    actions = torch.as_tensor(
        [t['action'] for t in transitions], dtype=torch.long,
        device=agent.device)
    old_logp = torch.as_tensor(
        [t['logp'] for t in transitions], dtype=torch.float32,
        device=agent.device)
    returns = torch.as_tensor(
        [t['return'] for t in transitions], dtype=torch.float32,
        device=agent.device)
    adv_t = torch.as_tensor(advantages, dtype=torch.float32,
                             device=agent.device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            if len(mb) < 2:
                continue
            mb_obs = obs[mb]
            mb_masks = masks[mb]
            mb_actions = actions[mb]
            mb_old_logp = old_logp[mb]
            mb_returns = returns[mb]
            mb_adv = adv_t[mb]

            new_logp, values, entropies = agent.evaluate_actions(
                mb_obs, mb_masks, mb_actions)

            # Ratio
            ratio = torch.exp(new_logp - mb_old_logp)
            # Clipped surrogate
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = ((values - mb_returns) ** 2).mean()

            # Entropy bonus (Eq. 2)
            entropy_bonus = entropies.mean()

            # Loss combinee
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            agent.policy_optim.zero_grad()
            agent.critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 5.0)
            agent.policy_optim.step()
            agent.critic_optim.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_bonus.item()
            n_updates += 1

    return {
        'policy_loss': total_policy_loss / max(n_updates, 1),
        'value_loss': total_value_loss / max(n_updates, 1),
        'entropy': total_entropy / max(n_updates, 1),
    }


# ---------------------------------------------------------------------- #
# Boucle principale
# ---------------------------------------------------------------------- #
def train(args):
    device = 'cpu'
    print(f"[ppo] device={device}", flush=True)

    # --- 1. Agent RL (policy discard + critic + SL figes) ---
    agent = RLDiscardAgent(
        sl_discard_path=args.sl_checkpoint,
        sl_frozen_dir=args.sl_frozen_dir,
        device=device,
        lr=args.lr,
        critic_lr=args.critic_lr,
    )

    # --- 2. GRP pour le reward ---
    grp_holder = PredictorHolder()
    if args.grp_checkpoint and os.path.isfile(args.grp_checkpoint):
        grp = RewardPredictor(input_dim=FEATURE_DIM, hidden_dim=128,
                               num_layers=2).to(device)
        ck = torch.load(args.grp_checkpoint, map_location=device,
                        weights_only=False)
        grp.load_state_dict(ck)
        grp.eval()
        grp_holder.model = grp
        print(f"[ppo] GRP charge depuis {args.grp_checkpoint}", flush=True)

    # --- 3. Env ---
    log_buffer = GameLogBuffer(max_size=10000)
    env = GameEnv(
        seed=42,
        predictor_holder=grp_holder,
        game_log_buffer=log_buffer,
        n_rounds=8,
    )
    # Remplace l'env interne par un env avec frozen SL
    # (GameEnv expose round_env, on y injecte les modeles figes)
    env.round_env.frozen_sl = agent.frozen_sl

    # --- 4. Entropy coefficient adaptatif (Eq. 3 du papier) ---
    alpha = args.alpha_init
    H_target = args.h_target
    beta = args.beta

    # --- 5. Boucle principale ---
    print(f"[ppo] debut de l'entrainement : {args.total_steps} steps max",
          flush=True)
    step = 0
    episode = 0
    recent_entropies = deque(maxlen=50)

    t_start = time.time()
    while step < args.total_steps:
        # --- Collecte ---
        transitions = collect_one_episode(env, agent, max_rounds=8)
        episode += 1
        if not transitions:
            continue
        step += len(transitions)

        # --- GAE ---
        advantages = compute_gae(transitions, gamma=args.gamma, lam=args.lam)

        # --- PPO update ---
        metrics = ppo_update(
            agent, transitions, advantages,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            clip_eps=args.clip_eps,
            entropy_coef=alpha,
            value_coef=args.value_coef,
        )

        # --- Entropy adaptatif (Eq. 3) ---
        recent_entropies.append(metrics['entropy'])
        if len(recent_entropies) >= 10:
            H_bar = np.mean(recent_entropies)
            alpha = max(0.0, alpha + beta * (H_target - H_bar))

        # --- Logs ---
        if episode % args.log_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[ep {episode:5d} | step {step:7d}] "
                f"pl={metrics['policy_loss']:+.4f} "
                f"vl={metrics['value_loss']:.4f} "
                f"H={metrics['entropy']:.4f} "
                f"alpha={alpha:.4f} "
                f"|T|={len(transitions)} "
                f"({elapsed:.0f}s)",
                flush=True)

        # --- Evaluation periodique ---
        if step >= (step // args.eval_every) * args.eval_every + len(transitions) \
                and step // args.eval_every > (step - len(transitions)) // args.eval_every:
            # Snapshot
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            agent.save(args.output)
            print(f"[ppo] checkpoint sauvegarde : {args.output} "
                  f"(step={step})", flush=True)

    # --- Save final ---
    agent.save(args.output)
    print(f"[ppo] entrainement termine. Checkpoint final : {args.output}",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sl-checkpoint', required=True,
                   help='checkpoint SL discard (init de la policy)')
    p.add_argument('--sl-frozen-dir', required=True,
                   help='dossier des 4 checkpoints SL figes')
    p.add_argument('--grp-checkpoint', default='checkpoints/grp_sl.pt')
    p.add_argument('--output', required=True)

    p.add_argument('--total-steps', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--critic-lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--ppo-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--h-target', type=float, default=1.5,
                   help='entropie cible (Eq. 3)')
    p.add_argument('--alpha-init', type=float, default=0.01)
    p.add_argument('--beta', type=float, default=0.001)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--eval-every', type=int, default=10000)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()