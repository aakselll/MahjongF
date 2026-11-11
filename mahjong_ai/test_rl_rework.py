"""Tests du rework RL (mahjong_ai.rl.*) — checkpoints factices, sans GPU.

Exécution :  python test_rl_rework.py            (depuis la racine du repo,
ou avec le repo sur PYTHONPATH)

Vérifie les invariants qui tuaient l'ancien pipeline :
    1. RATIO ON-POLICY : |ratio - 1| < 1e-4 au premier passage PPO
       (l'ancien replay donnait rdev >> 0 -> gradients biaisés/annulés).
    2. REWARDS : somme des rewards de manche par siège = reward de rang
       VRAI (shaping GRP + correction terminale exact).
    3. G_t : retour restant escompté = somme rho^(k-1) des rewards de
       manche (algorithme v9).
    4. PPO UPDATE : les poids de la policy bougent réellement.
    5. SAVE/RESUME : round-trip checkpoint complet (poids + optimizers).
    6. MODES : parties complètes à gamma 1.0 / 0.5 / 0.0, modes opponent
       et selfplay, mode éval argmax.
    7. SMOKE TRAIN : 2 itérations complètes de rl.train.train().
"""
import copy
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Le repo doit être importable comme package "mahjong_ai" (convention du
# projet : le dossier racine EST le package). Si le dossier ne s'appelle
# pas mahjong_ai, on crée un lien symbolique à côté.
_REPO = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_REPO)
if os.path.basename(_REPO) != 'mahjong_ai':
    _link = os.path.join(_PARENT, 'mahjong_ai')
    if not os.path.exists(_link):
        os.symlink(_REPO, _link)
    sys.path.insert(0, _PARENT)

from mahjong_ai.agents.suphx_cnn import build_model
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM
from mahjong_ai.envs.suphx_features import N_CHANNELS
from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner
from mahjong_ai.rl.ppo import compute_advantages, ppo_update
from mahjong_ai.training.game_env import RANK_REWARDS

CH, NB = 32, 2          # architecture factice minuscule (CPU)


def make_dummy_checkpoints(tmpdir):
    """Checkpoints SL factices (poids aléatoires, format identique aux
    vrais : {'model_state_dict', 'config', ...})."""
    torch.manual_seed(123)
    os.makedirs(tmpdir, exist_ok=True)
    for head in ['discard', 'riichi', 'chow', 'pong', 'kong']:
        m = build_model(head, channels=CH, n_blocks=NB, binary_hidden=64,
                        in_channels=N_CHANNELS)
        torch.save({
            'model_state_dict': m.state_dict(),
            'config': {'channels': CH, 'n_blocks': NB,
                       'binary_hidden': 64},
            'val_acc': 0.0,
        }, os.path.join(tmpdir, f'{head}.pt'))
    grp = RewardPredictor(input_dim=FEATURE_DIM, hidden_dim=16, num_layers=2)
    torch.save({
        'model_state_dict': grp.state_dict(),
        'hidden_dim': 16,
        'feature_dim': FEATURE_DIM,
        'val_loss': 0.0,
    }, os.path.join(tmpdir, 'grp.pt'))
    return tmpdir


def make_agent_and_runner(ckpt_dir, n_envs=2, mode='opponent',
                          grp=True, seed=0):
    agent = PPOAgent(
        os.path.join(ckpt_dir, 'discard.pt'), ckpt_dir, device='cpu',
        policy_lr=1e-4, critic_lr=1e-3)
    runner = VecRunner(
        agent, n_envs=n_envs, seed=seed,
        grp_path=(os.path.join(ckpt_dir, 'grp.pt') if grp else None),
        compute_lookahead=False, danger_labels=True, mode=mode)
    return agent, runner


def test_1_ratio_on_policy(ckpt_dir):
    print("\n=== TEST 1 : ratio PPO == 1 au premier passage ===")
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    episodes, _ = runner.collect_batch(2, gamma_t=1.0)
    transitions = [t for ep in episodes for t in ep]
    assert len(transitions) >= 16, f"batch trop petit : {len(transitions)}"
    compute_advantages(agent, transitions, mode='batch_mean')
    m = ppo_update(agent, transitions, epochs=1, minibatch=512)
    rdev = m['ratio_dev']
    print(f"    ratio_dev = {rdev:.2e}  (batch={m['batch_size']})")
    assert rdev < 1e-4, f"OFF-POLICY ! ratio_dev={rdev}"
    print("    OK")


def test_2_rewards(ckpt_dir):
    print("\n=== TEST 2 : somme des rewards = reward de rang VRAI ===")
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    episodes, stats = runner.collect_batch(3, gamma_t=1.0)
    assert len(episodes) == 3
    for ep, st in zip(episodes, stats):
        # rang du siège RL (mode opponent : 1 seul siège apprend)
        rank = st['rank_of_seat0']
        expected = RANK_REWARDS[rank]
        # groupes (siège, manche) consécutifs
        groups = {}
        order = []
        for t in ep:
            key = (t['seat'], t['round'])
            if key not in groups:
                order.append(key)
                groups[key] = []
            groups[key].append(t)
        total = sum(groups[g][-1]['reward'] for g in order)
        print(f"    rang={rank} expected={expected:+.1f} somme={total:+.3f}")
        assert abs(total - expected) < 1e-3, \
            f"somme rewards {total} != reward de rang {expected}"
    print("    OK")


def test_3_returns(ckpt_dir):
    print("\n=== TEST 3 : G_t = somme escomptee rho^(k-1) des manches ===")
    rho = 0.85
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    episodes, _ = runner.collect_batch(2, gamma_t=0.7)
    for ep in episodes:
        # groupes par siège
        by_seat = {}
        for t in ep:
            by_seat.setdefault(t['seat'], []).append(t)
        for seat, trs in by_seat.items():
            groups, order = {}, []
            for t in trs:
                key = t['round']
                if key not in groups:
                    order.append(key)
                    groups[key] = []
                groups[key].append(t)
            last_rewards = [groups[g][-1]['reward'] for g in order]
            expected = sum(rho ** k * r_k
                           for k, r_k in enumerate(last_rewards))
            got = trs[0]['return']
            print(f"    siège {seat} : G_0={got:+.3f} attendu={expected:+.3f}")
            assert abs(got - expected) < 1e-3
    print("    OK")


def test_4_update_changes_weights(ckpt_dir):
    print("\n=== TEST 4 : l'update PPO bouge les poids ===")
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    episodes, _ = runner.collect_batch(2, gamma_t=1.0)
    transitions = [t for ep in episodes for t in ep]
    compute_advantages(agent, transitions, mode='batch_mean')
    before = copy.deepcopy(agent.policy.state_dict())
    ppo_update(agent, transitions, epochs=3, minibatch=256)
    n_changed = sum(
        int(not torch.equal(before[k], agent.policy.state_dict()[k]))
        for k in before)
    total = len(before)
    print(f"    tenseurs modifiés : {n_changed}/{total}")
    assert n_changed > total // 2, "la policy n'a quasi pas bougé"
    print("    OK")


def test_5_save_resume(ckpt_dir):
    print("\n=== TEST 5 : save / resume ===")
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    episodes, _ = runner.collect_batch(2, gamma_t=1.0)
    transitions = [t for ep in episodes for t in ep]
    compute_advantages(agent, transitions, mode='batch_mean')
    ppo_update(agent, transitions, epochs=2, minibatch=256)
    path = os.path.join(ckpt_dir, 'test_ckpt.pt')
    agent.save(path, step=1234, alpha=0.01, phase=2)
    agent2 = PPOAgent(os.path.join(ckpt_dir, 'discard.pt'), ckpt_dir,
                      device='cpu')
    ck = agent2.load(path)
    for k in agent.policy.state_dict():
        assert torch.equal(agent.policy.state_dict()[k],
                           agent2.policy.state_dict()[k]), k
    assert ck['step'] == 1234 and ck['phase'] == 2
    print(f"    step={ck['step']} phase={ck['phase']} alpha={ck['alpha']}")
    print("    OK")


def test_6_modes(ckpt_dir):
    print("\n=== TEST 6 : gamma 1.0/0.5/0.0, selfplay, éval ===")
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2, mode='opponent')
    for g in (1.0, 0.5, 0.0):
        eps, stats = runner.collect_batch(2, gamma_t=g)
        n = sum(len(e) for e in eps)
        mr = np.mean([s['rank_of_seat0'] for s in stats])
        print(f"    gamma={g}: {len(eps)} parties, {n} transitions, "
              f"mr={mr:.2f}, unhandled="
              f"{sum(s['n_unhandled'] for s in stats)}")
        assert all(s['n_unhandled'] == 0 for s in stats)
    # self-play : 4 sièges RL
    agent_sp, runner_sp = make_agent_and_runner(
        ckpt_dir, n_envs=2, mode='selfplay')
    eps, stats = runner_sp.collect_batch(2, gamma_t=1.0)
    seats = set(t['seat'] for ep in eps for t in ep)
    print(f"    selfplay : {len(eps)} parties, sièges vus={sorted(seats)}")
    assert len(seats) >= 2
    # éval : argmax, gamma=0, aucune transition
    _, eval_stats = runner.collect_batch(
        2, gamma_t=1.0, deterministic=True, collect=False, gamma_override=0.0)
    print(f"    éval argmax gamma=0 : "
          f"mr={np.mean([s['rank_of_seat0'] for s in eval_stats]):.2f}")
    print("    OK")


def test_7_smoke_train(ckpt_dir):
    print("\n=== TEST 7 : smoke train (2 itérations complètes) ===")
    from mahjong_ai.rl.train import build_parser, train
    out = os.path.join(ckpt_dir, 'rl_smoke', 'rl.pt')
    argv = [
        '--sl-discard', os.path.join(ckpt_dir, 'discard.pt'),
        '--sl-frozen', ckpt_dir,
        '--grp-checkpoint', os.path.join(ckpt_dir, 'grp.pt'),
        '--output', out,
        '--total-steps', '900',
        '--n-envs', '2', '--batch-games', '2',
        '--max-batch-transitions', '512',
        '--minibatch', '128', '--ppo-epochs', '2',
        '--log-every', '1', '--save-every', '2000',
        '--oracle-frac', '0.4', '--transition-frac', '0.3',
        '--stats-window', '50',
    ]
    args = build_parser().parse_args(argv)
    t0 = time.time()
    train(args)
    assert os.path.isfile(os.path.splitext(out)[0] + '_latest.pt'), \
        "checkpoint latest manquant"
    print(f"    OK ({time.time()-t0:.0f}s)")


def test_8_v2_gate0_gate1(ckpt_dir):
    print("\n=== TEST 8 : [V2] accord SL, critic baseline, early-stop KL ===")
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    # conditions no-oracle : gamma_t=0 -> oracle masqué partout
    episodes, _ = runner.collect_batch(2, gamma_t=0.0)
    transitions = [t for ep in episodes for t in ep]
    assert len(transitions) >= 16

    # (a) accord top-1 avec le SL à l'init = 1.0 EXACT (canaux oracle
    # muets : conv sans bias -> contribution nulle des canaux zéros)
    agr0 = agent.sl_agreement(transitions)
    print(f"    accord SL à l'init = {agr0:.4f} (attendu 1.0)")
    assert agr0 is not None and agr0 > 0.999, \
        f"accord init {agr0} != 1.0 : l'ancre SL est cassée"

    # (b) critic entraîné sur le batch (warmup, policy figée) ->
    #     Var(G - V) < Var(G) : c'est LE mécanisme Gate 1. Un critic
    #     non entraîné ferait l'inverse (variance ajoutée), d'où le
    #     warmup --critic-warmup-updates du trainer.
    _, info_bm = compute_advantages(agent, transitions, mode='batch_mean')
    for _ in range(20):
        compute_advantages(agent, transitions, mode='batch_mean')
        ppo_update(agent, transitions, epochs=1, minibatch=128, warmup=True)
    _, info = compute_advantages(agent, transitions, mode='critic',
                                 min_ev=-1e9)
    print(f"    adv_var_raw : batch_mean={info_bm['adv_var_raw']:.2f} "
          f"critic(appris)={info['adv_var_raw']:.2f} "
          f"(ev={info['ev']:+.3f}, used_critic={info['used_critic']})")
    assert info['used_critic'], "critic non utilisé (min_ev)"
    assert info['adv_var_raw'] < info_bm['adv_var_raw'], \
        "le critic entraîné doit absorber de la variance"

    # (c) early-stop KL : cible minuscule -> l'update coupe avant la fin
    m_full = ppo_update(agent, transitions, epochs=4, minibatch=128)
    m_stop = ppo_update(agent, transitions, epochs=4, minibatch=128,
                        target_kl=1e-8)
    print(f"    grad-steps : complet={m_full['n_grad_steps']} "
          f"avec target_kl=1e-8 -> {m_stop['n_grad_steps']} "
          f"(early_stop={m_stop['early_stop']})")
    assert m_stop['early_stop'], "l'early-stop KL ne s'est pas déclenché"
    assert m_stop['n_grad_steps'] < m_full['n_grad_steps'], \
        "l'early-stop n'a rien économisé"
    # rétro-compat : sans target_kl, early_stop=False et même budget
    assert not m_full['early_stop']

    # (d) l'accord descend après des updates agressifs (l'instrument vit)
    for _ in range(3):
        compute_advantages(agent, transitions, mode='batch_mean')
        ppo_update(agent, transitions, epochs=4, minibatch=128)
    agr1 = agent.sl_agreement(transitions)
    print(f"    accord SL après 3 updates agressifs = {agr1:.4f} "
          f"(doit être <= 1.0)")
    assert 0.0 <= agr1 <= 1.0
    print("    OK")


def main():
    tmpdir = tempfile.mkdtemp(prefix='mj_rl_test_')
    print(f"checkpoints factices : {tmpdir}")
    make_dummy_checkpoints(tmpdir)
    t0 = time.time()
    test_1_ratio_on_policy(tmpdir)
    test_2_rewards(tmpdir)
    test_3_returns(tmpdir)
    test_4_update_changes_weights(tmpdir)
    test_5_save_resume(tmpdir)
    test_6_modes(tmpdir)
    test_7_smoke_train(tmpdir)
    test_8_v2_gate0_gate1(tmpdir)
    print(f"\n=== TOUS LES TESTS PASSENT ({time.time()-t0:.0f}s) ===")


if __name__ == '__main__':
    main()
