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
    8. [V2] accord SL à l'init = 1.0, critic qui absorbe la variance,
       early-stop KL.
    9. [V2.1] CRITIC REPLAY : fenêtre FIFO (éviction), échantillonnage
       uniforme, pas dédiés critic sans toucher la policy, loss
       in-sample qui baisse sur fenêtre fixe.
    10. [V2.2] RÉGULARISATION : sampler jamais dans la val (skip_newest),
        early-stop sur val qui remonte, weight decay qui rétrécit les
        poids, intégration train() (bannière + cfit/cval dans DRIFT).
    11. [V2.3] DIAG : round-trip du cache npz (f16/f32 exacts),
        BackboneCritic (V=0 à l'init, tronc SL figé, seule la partie
        entraînable bouge).
    12. [D-2] EVENT-PEN : avantages par événement (masque exact,
        non centré/non normalisé, garde sans pénalité), ppo_update
        event_pen (minibatchs sans événement, critic figé, policy qui
        bouge), porte de plomberie _event_pen_selfcheck, intégration
        train() complète.
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


def test_9_critic_replay(ckpt_dir):
    print("\n=== TEST 9 : [V2.1] fenêtre replay du critic ===")
    from mahjong_ai.rl.ppo import CriticReplay, critic_fit
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    episodes, _ = runner.collect_batch(2, gamma_t=0.0)
    transitions = [t for ep in episodes for t in ep]
    assert len(transitions) >= 60, "batch trop petit"

    # (a) push / éviction FIFO / échantillonnage : formes + contenu
    rep = CriticReplay(capacity_batches=4, seed=0)
    half = len(transitions) // 2
    rep.push(transitions[:half])
    rep.push(transitions[half:])
    assert len(rep) == len(transitions) and rep.n_batches == 2
    for _ in range(6):
        rep.push(transitions[:10])          # 8 pushes, capacité 4
    assert len(rep) == 40 and rep.n_batches == 4, \
        f"éviction FIFO cassée : len={len(rep)} batches={rep.n_batches}"
    on, oof, rets = rep.sample_minibatch(37)
    assert on.shape == (37, N_CHANNELS, 34), on.shape
    assert oof.shape == (37, 4, 34) and rets.shape == (37,)
    w_rets = np.concatenate(rep._rets)      # accès white-box volontaire
    assert rets.min() >= w_rets.min() - 1e-6, "échantillon hors fenêtre"
    assert rets.max() <= w_rets.max() + 1e-6, "échantillon hors fenêtre"

    # (b) le fit critic ne touche PAS la policy
    agr_before = agent.sl_agreement(transitions)
    w_pol = copy.deepcopy(agent.policy.state_dict())
    info = critic_fit(agent, rep, n_steps=25, minibatch=32)
    assert info and info['n_steps'] == 25 and info['loss'] is not None
    for k in w_pol:
        assert torch.equal(w_pol[k], agent.policy.state_dict()[k]), k
    assert agent.sl_agreement(transitions) == agr_before, \
        "le fit a déplacé la policy"

    # (c) in-sample : la loss baisse en réentraînant sur la même fenêtre
    l0 = info['loss']
    for _ in range(3):
        info = critic_fit(agent, rep, n_steps=25, minibatch=32)
    print(f"    cfit in-sample : {l0:.4f} -> {info['loss']:.4f} "
          f"(fenêtre fixe de 40)")
    assert info['loss'] < l0, "la loss critic ne baisse pas in-sample"

    # (d) intégration train() : --adv-mode critic + replay actif (le
    #     chemin exact du prochain run), 2 itérations complètes
    from mahjong_ai.rl.train import build_parser, train
    out = os.path.join(ckpt_dir, 'rl_smoke_v21', 'rl.pt')
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
        '--adv-mode', 'critic',
        '--critic-warmup-updates', '2',
        '--critic-replay-batches', '3',
        '--critic-replay-steps', '8',
    ]
    args = build_parser().parse_args(argv)
    train(args)
    assert os.path.isfile(os.path.splitext(out)[0] + '_latest.pt'), \
        "checkpoint latest manquant (intégration v2.1)"
    print("    OK")


def test_10_critic_v22(ckpt_dir):
    print("\n=== TEST 10 : [V2.2] régularisation du critic_fit ===")
    import io
    from contextlib import redirect_stdout
    from mahjong_ai.rl.ppo import CriticReplay, critic_fit
    from mahjong_ai.agents.oracle_agent import Critic
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)

    batches = []
    for _ in range(3):
        eps, _ = runner.collect_batch(2, gamma_t=0.0)
        batches.append([t for ep in eps for t in ep])
    assert all(len(b) >= 30 for b in batches), "batch trop petit"

    # (a) skip_newest : le sampler ne tire JAMAIS dans le batch réservé
    marker = 424242.0
    val_batch = [{**t, 'return': marker} for t in batches[-1]]
    rep = CriticReplay(capacity_batches=10, seed=0)
    for b in batches[:-1]:
        rep.push(b)
    rep.push(val_batch)
    for _ in range(50):
        s = rep.sample_minibatch(48, skip_newest=1)
        assert s is not None and (s[2] != marker).all(), \
            "le sampler a tiré dans la val"
    hit = False
    for _ in range(300):
        s = rep.sample_minibatch(48, skip_newest=0)
        if (s[2] == marker).any():
            hit = True
            break
    assert hit, "batch val inaccessible même sans skip (test vide)"

    # (b) early-stop : cibles constantes opposées (train +5, val -5)
    #     -> la val REMONTE a chaque check -> le fit coupe avant la fin
    tb = [{**t, 'return': 5.0} for t in batches[0]]
    vb = [{**t, 'return': -5.0} for t in batches[1]]
    rep2 = CriticReplay(capacity_batches=10, seed=1)
    rep2.push(tb)
    rep2.push(vb)
    info = critic_fit(agent, rep2, n_steps=120, minibatch=64,
                      reward_scale=10.0, val_batches=1, patience=3)
    assert info and info['stopped_at'] is not None, "early-stop muet"
    assert info['n_steps'] < 120, "l'early-stop n'a rien coupé"
    assert info['val_loss'] is not None and info['val_loss'] > info['loss'], \
        "la val doit être > train (cibles opposées)"
    print(f"    early-stop : coupé a {info['n_steps']}/120 steps "
          f"(val {info['val_loss']:.3f} > cfit {info['loss']:.3f})")

    # (c) weight decay : mêmes poids initiaux, mêmes steps -> ||W|| plus petit
    torch.manual_seed(7)
    base = Critic().state_dict()
    saved = (agent.critic, agent.critic_optim)
    norms = {}
    for wd in (0.0, 1e-2):
        c = Critic()
        c.load_state_dict(base)
        c.eval()
        agent.critic = c
        agent.critic_optim = torch.optim.Adam(
            c.parameters(), lr=1e-3)
        rep3 = CriticReplay(capacity_batches=10, seed=2)
        rep3.push([{**t, 'return': 5.0} for t in batches[0]])
        critic_fit(agent, rep3, n_steps=60, minibatch=64,
                   reward_scale=10.0, weight_decay=wd, val_batches=0)
        norms[wd] = sum(float((p * p).sum())
                        for p in agent.critic.parameters())
    agent.critic, agent.critic_optim = saved
    print(f"    ||W||² après 60 steps : wd=0 -> {norms[0.0]:.1f}, "
          f"wd=1e-2 -> {norms[1e-2]:.1f}")
    assert norms[1e-2] < norms[0.0], "le weight decay ne rétrécit rien"

    # (d) intégration train() : bannière V2.2 + cfit/cval dans les DRIFT
    from mahjong_ai.rl.train import build_parser, train
    out = os.path.join(ckpt_dir, 'rl_smoke_v22', 'rl.pt')
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
        '--adv-mode', 'critic',
        '--critic-warmup-updates', '2',
        '--critic-replay-batches', '3',
        '--critic-replay-steps', '8',
        '--critic-wd', '0.001',
        '--critic-val-batches', '1',
    ]
    args = build_parser().parse_args(argv)
    buf = io.StringIO()
    with redirect_stdout(buf):
        train(args)
    txt = buf.getvalue()
    assert '[V2.2]' in txt, "bannière V2.2 absente des logs"
    assert 'cval=' in txt, "cval absent des lignes DRIFT"
    assert os.path.isfile(os.path.splitext(out)[0] + '_latest.pt')
    print("    OK")


def test_11_diag_v23(ckpt_dir):
    print("\n=== TEST 11 : [V2.3] diag : cache + BackboneCritic ===")
    from mahjong_ai.diag_critic_learnability import (
        BackboneCritic, save_cache, load_cache, _batch_from_transitions)
    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    batches = []
    for _ in range(2):
        eps, _ = runner.collect_batch(2, gamma_t=0.0)
        batches.append(_batch_from_transitions(
            [t for ep in eps for t in ep]))
    assert all(len(b['ret']) >= 30 for b in batches), "batch trop petit"

    # (a) round-trip du cache : octet-exact sur les 4 tableaux
    path = os.path.join(ckpt_dir, 'diag_cache_test.npz')
    save_cache(path, batches)
    loaded = load_cache(path)
    assert len(loaded) == 2
    for a, b in zip(loaded, batches):
        assert a['obs_n'].dtype == np.float16
        assert np.array_equal(a['obs_n'], b['obs_n'])
        assert np.array_equal(a['obs_of'], b['obs_of'])
        assert np.array_equal(a['ret'], b['ret'])
        assert np.array_equal(a['rew'], b['rew'])

    # (b) BackboneCritic : V=0 exactement à l'init, tronc figé
    bc = BackboneCritic(agent.sl_discard).eval()
    on = torch.as_tensor(batches[0]['obs_n'][:8].astype(np.float32))
    oof = torch.as_tensor(batches[0]['obs_of'][:8].astype(np.float32))
    with torch.no_grad():
        v0 = bc(on, oof)
    assert v0.shape == (8,), v0.shape
    assert float(v0.abs().max()) == 0.0, "la tête zero-init doit donner V=0"
    assert all(not p.requires_grad for p in bc.trunk.parameters()), \
        "le tronc SL doit être figé"
    n_trainable = sum(1 for p in bc.parameters() if p.requires_grad)
    assert n_trainable > 0, "aucune partie entraînable"

    # (c) 5 pas de fit : le tronc ne bouge PAS, la partie entraînable si
    w0 = {k: t.clone() for k, t in bc.trunk.state_dict().items()}
    opt = torch.optim.Adam(
        [p for p in bc.parameters() if p.requires_grad], lr=1e-3)
    tgt = torch.as_tensor(
        batches[0]['ret'][:8] / 10.0, dtype=torch.float32)
    for _ in range(5):
        v = bc(on, oof)
        loss = 0.5 * ((v - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for k in w0:
        assert torch.equal(w0[k], bc.trunk.state_dict()[k]), k
    with torch.no_grad():
        v1 = bc(on, oof)
    assert float(v1.abs().max()) > 0.0, "la partie entraînable n'a pas vécu"
    print(f"    tronc figé intact, V: 0.0 -> {float(v1.abs().max()):.4f} "
          f"({n_trainable} tenseurs entraînables)")
    print("    OK")


def test_12_event_pen(ckpt_dir):
    print("\n=== TEST 12 : [D-2] avantages par événement (event_pen) ===")
    import argparse
    import io
    from contextlib import redirect_stdout
    from mahjong_ai.rl.ppo import compute_event_pen_advantages
    from mahjong_ai.rl.train import (build_parser, train,
                                     _event_pen_selfcheck)

    agent, runner = make_agent_and_runner(ckpt_dir, n_envs=2)
    eps, _ = runner.collect_batch(4, gamma_t=0.0, danger_penalty=2.5)
    transitions = [t for ep in eps for t in ep]
    assert len(transitions) >= 60, "batch trop petit"
    assert all(t.get('danger') is not None for t in transitions), \
        "labels danger absents alors que danger_labels=True"

    # (a) unitaire : 5 événements garantis (mutation white-box du label,
    # comme le marker 424242 des tests V2.2) + éventuels événements
    # naturels du batch.
    for t in transitions[:5]:
        d = t['danger'].copy()
        d[int(t['action'])] = 1
        t['danger'] = d
    advs, info = compute_event_pen_advantages(transitions, 2.5)
    n_ev = sum(1 for t in transitions if t['is_event'])
    assert n_ev >= 5 and info['n_events'] == n_ev
    assert abs(info['adv_ev'] + 2.5) < 1e-6
    # non centré : 0 EXACT sur les non-événements (pas de +p*(1-p))
    assert all(t['advantage'] == 0.0
               for t in transitions if not t['is_event'])
    assert all(t['advantage'] == -2.5
               for t in transitions if t['is_event'])
    # non normalisé : Var brute = p(1-p) * d^2, pas 1
    p = n_ev / len(transitions)
    assert abs(info['adv_var_raw'] - p * (1 - p) * 2.5 ** 2) < 1e-4
    # garde : sans pénalité -> refus immédiat (policy loss vide)
    try:
        compute_event_pen_advantages(transitions, 0.0)
        raise AssertionError("danger_penalty=0 accepté en event_pen")
    except SystemExit:
        pass
    print(f"    (a) n_ev={n_ev}/{len(transitions)} "
          f"({info['ev_pct']:.2f}%) adv_ev={info['adv_ev']:+.2f} "
          f"advvar={info['adv_var_raw']:.3f}")

    # (b) ppo_update(event_pen=True) : minibatch=8 -> la plupart des
    # minibatchs SANS événement (l'ancien crash potentiel) ; la policy
    # bouge (ancre KL), le critic ne bouge PAS (value_coef forcé à 0).
    w0 = copy.deepcopy(agent.policy.state_dict())
    c0 = copy.deepcopy(agent.critic.state_dict())
    m = ppo_update(agent, transitions, epochs=2, minibatch=8,
                   event_pen=True, kl_coef=0.05)
    assert m['n_grad_steps'] > 0 and m['batch_size'] == len(transitions)
    n_changed = sum(
        int(not torch.equal(w0[k], agent.policy.state_dict()[k]))
        for k in w0)
    assert n_changed > 0, "la policy n'a pas bougé (masque mort ?)"
    assert all(torch.equal(c0[k], agent.critic.state_dict()[k])
               for k in c0), "le critic a bougé (value_coef != 0 ?)"
    print(f"    (b) grad-steps={m['n_grad_steps']} "
          f"(mb=8, plupart sans événement) policy modifiée, critic figé")

    # (c) porte de plomberie : passe au régime nominal, échoue sur
    # champ absent / part aberrante / signe inversé.
    ns = argparse.Namespace(danger_penalty=2.5)
    _event_pen_selfcheck(
        {'n_events': 37, 'ev_pct': 1.7, 'adv_ev': -2.5}, ns,
        n_transitions=2190)
    for bad in (
            {'n_events': 0, 'ev_pct': 0.0, 'adv_ev': 0.0},      # champ absent
            {'n_events': 400, 'ev_pct': 40.0, 'adv_ev': -2.5},   # part aberrante
            {'n_events': 37, 'ev_pct': 1.7, 'adv_ev': +2.5}):    # signe
        try:
            _event_pen_selfcheck(bad, ns, n_transitions=2190)
            raise AssertionError(f"selfcheck muet sur {bad}")
        except SystemExit:
            pass
    print("    (c) selfcheck : OK au nominal, SystemExit sur les 3 casses")

    # (d) intégration train() : --adv-mode event_pen, itérations
    # complètes (chemin exact du run D-2, --no-oracle + ancre KL).
    # Les checkpoints factices produisent ~0.07% d'événements danger
    # naturels (mesuré : 3/4330 ; adversaires aléatoires quasi jamais
    # tenpai) contre ~1.7% avec les vrais SL (diag B) : la porte
    # [0.5%, 5%] ne peut qu'échouer ici, et c'est son travail. Le
    # wrapper white-box ci-dessous (même esprit que le marker 424242
    # des tests V2.2) garantit >= 1 événement par batch pour exercer
    # le chemin masqué COMPLET de train() ; la logique non masquée est
    # déjà couverte par (a)-(c). B < 800 -> seule la porte plomberie
    # (n_ev > 0, signe) s'applique, de façon déterministe.
    import mahjong_ai.rl.train as TR
    real_cepa = TR.compute_event_pen_advantages

    def cepa_with_events(transitions, pen):
        advs, info = real_cepa(transitions, pen)
        if info['n_events'] == 0:
            d = transitions[0]['danger'].copy()
            d[int(transitions[0]['action'])] = 1
            transitions[0]['danger'] = d
            advs, info = real_cepa(transitions, pen)
        return advs, info

    TR.compute_event_pen_advantages = cepa_with_events
    try:
        out = os.path.join(ckpt_dir, 'rl_smoke_d2', 'rl.pt')
        argv = [
            '--sl-discard', os.path.join(ckpt_dir, 'discard.pt'),
            '--sl-frozen', ckpt_dir,
            '--grp-checkpoint', os.path.join(ckpt_dir, 'grp.pt'),
            '--output', out,
            '--total-steps', '700',
            '--n-envs', '2', '--batch-games', '2',
            '--max-batch-transitions', '512',
            '--minibatch', '32', '--ppo-epochs', '2',
            '--log-every', '1', '--save-every', '5000',
            '--oracle-frac', '0.4', '--transition-frac', '0.3',
            '--stats-window', '50',
            '--no-oracle',
            '--adv-mode', 'event_pen',
            '--danger-penalty', '2.5',
            '--kl-coef', '0.05',
        ]
        args = build_parser().parse_args(argv)
        buf = io.StringIO()
        with redirect_stdout(buf):
            train(args)
    finally:
        TR.compute_event_pen_advantages = real_cepa
    txt = buf.getvalue()
    assert '[D-2]' in txt, "bannière D-2 absente des logs"
    assert '[event_pen][check]' in txt, "selfcheck batch 1 absent"
    check_line = [l for l in txt.splitlines()
                  if '[event_pen][check]' in l][0]
    assert 'OK' in check_line, f"selfcheck échoué : {check_line}"
    assert 'ev%=' in txt, "ev%/n_ev absents des lignes de log"
    assert os.path.isfile(os.path.splitext(out)[0] + '_latest.pt'), \
        "checkpoint latest manquant (intégration D-2)"
    print(f"    (d) intégration : {check_line.strip()}")
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
    test_9_critic_replay(tmpdir)
    test_10_critic_v22(tmpdir)
    test_11_diag_v23(tmpdir)
    test_12_event_pen(tmpdir)
    print(f"\n=== TOUS LES TESTS PASSENT ({time.time()-t0:.0f}s) ===")


if __name__ == '__main__':
    main()
