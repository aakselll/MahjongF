"""
Parametric Monte-Carlo Policy Adaptation (pMCPA).

Reference : Suphx paper section 3.4, Eq. 6.

    theta_a = arg max_theta  Sum_{tau ~ theta_o . T(h)}  R(tau) * p(tau;theta) / p(tau;theta_o)

Au debut de chaque manche :
    1. Simulations : K trajectoires avec la policy offline (theta_o)
       -> adversaires et mur tires aleatoirement (hors notre main)
    2. Adaptation  : gradient updates avec importance sampling pondere par R(tau)
    3. Inference   : utilise theta_a pour cette manche

Apres la manche, on reset sur theta_o.

Note : s'applique UNIQUEMENT au modele discard (comme pour le RL).

[FIX 2025-XX] Helper _prepare_obs_for_model : detecte la taille d'input
attendue par le modele (272 = SL standard sans oracle, 276 = RL avec
oracle). Ajoute les 4 canaux oracle de zeros UNIQUEMENT si le modele les
attend. Utilise partout (simulation, adaptation, inference) pour eviter
les "Channel mismatch 272 vs 276".

[FIX 2025-XX-2] Dealer recoit 14 tuiles a ActionNewRound (13 + pioche).
adapt_from_sniffer_state retire la derniere piochee pour ramener a 13.
"""
import copy
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures import as_completed

import numpy as np
import torch
import torch.nn as nn


# ====================================================================== #
# Constantes
# ====================================================================== #
N_TILES = 34
HAND_SIZE = 13
COPIES_PER_TILE = 4
LIVE_WALL_SIZE = 70
DEAD_WALL_SIZE = 14
RINSHAN_SIZE = 4

DEFAULT_ROUND_REWARD_SCALE = 1.0 / 1000.0


# ====================================================================== #
# Helper : prepare obs selon le modele
# ====================================================================== #
def _prepare_obs_for_model(obs, model):
    """[FIX] Detecte la taille d'input attendue par le modele et ajoute
    les canaux oracle de zeros UNIQUEMENT si le modele les attend.

    Args:
        obs   : np.array (C, 34) avec C canaux normaux
        model : nn.Module avec input_conv.weight

    Retourne :
        np.array (C', 34) pret pour le forward, ou C' = C ou C+4.

    Raises:
        RuntimeError si in_channels ne matche ni C ni C+4.
    """
    expected_c = None
    for name, p in model.named_parameters():
        if name.endswith('input_conv.weight'):
            expected_c = p.shape[1]
            break
    if expected_c is None:
        raise RuntimeError("input_conv.weight introuvable dans le modele")

    actual_c = obs.shape[0]
    if expected_c == actual_c:
        return obs   # pas d'oracle
    if expected_c == actual_c + 4:
        # Modele avec oracle -> ajoute 4 canaux de zeros
        zeros = np.zeros((4, obs.shape[1]), dtype=obs.dtype)
        return np.concatenate([obs, zeros], axis=0)
    raise RuntimeError(
        f"Channel mismatch : modele attend {expected_c}, obs a {actual_c}"
    )


# ====================================================================== #
# Simulation d'une manche isolee
# ====================================================================== #
def _sample_round_state(our_hand_counts, rng):
    """Genere un etat initial de manche avec notre main fixe.

    Args:
        our_hand_counts : np.array (34,) int -- main de notre agent (13 tuiles)
        rng             : np.random.RandomState

    Retourne un dict avec :
        hands   : liste de 4 np.array (34,) int -- indexee par siege REEL
        wall    : liste de tuiles restantes (70)
        dead_wall : liste de tuiles du mur mort (14)
        dora_indicators : liste d'indicators
    """
    pool = []
    for t in range(N_TILES):
        remaining = COPIES_PER_TILE - int(our_hand_counts[t])
        if remaining > 0:
            pool.extend([t] * remaining)
    rng.shuffle(pool)

    idx = 0
    opp_hands = []
    for _ in range(3):
        h = np.zeros(N_TILES, dtype=np.int32)
        for t in pool[idx:idx + HAND_SIZE]:
            h[t] += 1
        idx += HAND_SIZE
        opp_hands.append(h)

    dead_wall = pool[idx:idx + DEAD_WALL_SIZE]
    idx += DEAD_WALL_SIZE
    live_wall = pool[idx:]

    dora_indicators = [dead_wall[0]] if dead_wall else []
    ura_dora = [dead_wall[1]] if len(dead_wall) > 1 else []

    hands = [
        our_hand_counts.astype(np.int32),
        opp_hands[0], opp_hands[1], opp_hands[2],
    ]

    return {
        'hands': hands,
        'wall': live_wall,
        'dead_wall': dead_wall,
        'dora_indicators': dora_indicators,
        'ura_dora_indicators': ura_dora,
    }


# ====================================================================== #
# Agent pMCPA
# ====================================================================== #
class PMCPA:
    """Adaptation run-time de la policy discard a la main initiale."""

    def __init__(self, models, head_sizes, device='cpu',
                 k=50, lr=1e-4, grad_steps=3, timeout_s=20.0,
                 reward_scale=DEFAULT_ROUND_REWARD_SCALE,
                 verbose=True, n_workers=None):
        self.device = device
        self.k = int(k)
        self.lr = float(lr)
        self.grad_steps = int(grad_steps)
        self.timeout_s = float(timeout_s)
        self.reward_scale = float(reward_scale)
        self.verbose = bool(verbose)
        configured_workers = (
            os.environ.get("PMCPA_N_WORKERS", "4")
            if n_workers is None else n_workers
        )
        self.n_workers = max(1, int(configured_workers))

        self.models = models
        self.discard = models['discard']
        self.frozen = {h: models[h] for h in
                        ['riichi', 'chow', 'pong', 'kong']
                        if h in models}

        self._offline_discard_state = copy.deepcopy(
            self.discard.state_dict()
        )

        self._adapt_optim = None

        self.is_adapted = False
        self._initial_hand = None
        self._thread = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def reset(self):
        """Restaure les poids OFFLINE. Appele au debut de chaque manche."""
        with self._lock:
            self.discard.load_state_dict(self._offline_discard_state)
            self.discard.eval()
            self.is_adapted = False
            self._initial_hand = None
            self._adapt_optim = None
        if self.verbose:
            print("[pMCPA] reset sur poids offline", flush=True)

    # ------------------------------------------------------------------ #
    def adapt_from_sniffer_state(self, state):
        """Lance l'adaptation a la main initiale.

        [FIX] Dealer recoit 14 tuiles (13 + pioche). On retire la derniere
        pour ramener a 13.
        """
        if not state.hand:
            if self.verbose:
                print("[pMCPA] main vide, adaptation ignoree", flush=True)
            return

        hand = list(state.hand)
        # [FIX] Dealer : 14 tuiles au premier tour, retire la derniere piochee
        if len(hand) == HAND_SIZE + 1:
            hand = hand[:-1]

        hand_counts = np.zeros(N_TILES, dtype=np.int32)
        for t in hand:
            if 0 <= t < N_TILES:
                hand_counts[t] += 1

        if int(hand_counts.sum()) != HAND_SIZE:
            if self.verbose:
                print(f"[pMCPA] main invalide ({hand_counts.sum()} tuiles, "
                      f"attendu {HAND_SIZE}), adaptation ignoree", flush=True)
            return

        self._initial_hand = hand_counts
        self.is_adapted = False

        self._thread = threading.Thread(
            target=self._adapt_worker,
            args=(hand_counts, state),
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    def compute_contextual_reward(self, sniffer_state, round_score):
        """Reward contextuel pour pMCPA.

        - Si pas la derniere manche : score brut.
        - Derniere manche : evaluer l'impact sur le rang final.
        """
        my_seat = getattr(sniffer_state, "my_seat", None)
        if my_seat is None:
            return round_score

        round_idx = getattr(sniffer_state, "round_idx", 0)
        is_last_round = (round_idx >= 3)

        scores = list(getattr(sniffer_state, "scores", [0.0] * 4))
        if len(scores) != 4:
            return round_score

        def _rank_of(seat, sc):
            return 1 + sum(1 for s in sc if s > sc[seat])

        current_rank = _rank_of(my_seat, scores)

        if not is_last_round:
            return round_score

        new_scores = scores.copy()
        new_scores[my_seat] += round_score
        new_rank = _rank_of(my_seat, new_scores)
        rank_delta = current_rank - new_rank

        if current_rank == 1:
            if new_rank == 1:
                return +15.0
            else:
                return -15.0
        elif current_rank == 4:
            if new_rank == 4:
                return -15.0
            else:
                return +10.0 * rank_delta
        else:
            return 5.0 * rank_delta

    # ------------------------------------------------------------------ #
    def _adapt_worker(self, hand_counts, state):
        """Thread d'adaptation."""
        t_start = time.time()
        try:
            if self.verbose:
                print(f"[pMCPA] debut adaptation (main={hand_counts.sum()} "
                      f"tuiles, K={self.k}, workers={self.n_workers})",
                      flush=True)

            trajectories = self._simulate_k_rounds(hand_counts, state)

            if not trajectories:
                if self.verbose:
                    print("[pMCPA] aucune trajectoire generee", flush=True)
                return

            self._adapt_with_importance_sampling(trajectories)
            self.is_adapted = True

            if self.verbose:
                dt = time.time() - t_start
                print(f"[pMCPA] ADAPTATION TERMINEE en {dt:.1f}s "
                      f"({len(trajectories)} trajectoires)", flush=True)

        except Exception as e:
            if self.verbose:
                print(f"[pMCPA] erreur adaptation : {e}", flush=True)
                import traceback
                traceback.print_exc()
            self.is_adapted = False

    # ------------------------------------------------------------------ #
    def _simulate_k_rounds(self, our_hand, sniffer_state):
        """Simule K manches avec la policy offline en parallele.

        Chaque worker possede son propre environnement et son propre RNG.
        Les simulations ne modifient pas les poids : l'adaptation est
        effectuee ensuite dans le thread principal.
        """
        trajectories = []
        rng = np.random.RandomState(int(time.time() * 1000) & 0x7fffffff)

        offline_state = copy.deepcopy(self.discard.state_dict())
        self.discard.eval()

        t_start = time.time()

        # Les seeds sont generees dans le thread principal afin d'eviter
        # tout partage de RandomState entre workers.
        tasks = [
            (
                i,
                np.random.RandomState(int(rng.randint(0, 2**31 - 1))),
            )
            for i in range(self.k)
        ]

        def simulate(task):
            simulation_id, simulation_rng = task
            try:
                trajectory = self._simulate_one_round(
                    our_hand, sniffer_state, simulation_rng, offline_state)
                return simulation_id, trajectory, None
            except Exception as exc:
                return simulation_id, None, exc

        with ThreadPoolExecutor(
                max_workers=min(self.n_workers, max(1, self.k)),
                thread_name_prefix="pmcpa-sim") as executor:
            futures = [executor.submit(simulate, task) for task in tasks]
            try:
                for future in as_completed(futures, timeout=self.timeout_s):
                    simulation_id, trajectory, error = future.result()
                    if error is not None:
                        if self.verbose:
                            print(f"[pMCPA] sim {simulation_id} echouee : "
                                  f"{error}", flush=True)
                        continue
                    if trajectory is not None:
                        trajectories.append(trajectory)
                        if self.verbose:
                            elapsed = time.time() - t_start
                            print(f"[pMCPA] sim {simulation_id + 1}/{self.k} "
                                  f"OK ({elapsed:.1f}s, "
                                  f"{len(trajectory['trajectory'])} "
                                  f"transitions, "
                                  f"reward={trajectory['reward']:+.1f})",
                                  flush=True)
            except FuturesTimeoutError:
                if self.verbose:
                    completed = len(trajectories)
                    print(f"[pMCPA] timeout sims a {completed}/{self.k} "
                          f"({time.time()-t_start:.1f}s)", flush=True)
                for future in futures:
                    future.cancel()

        self.discard.train()
        return trajectories

    # ------------------------------------------------------------------ #
    def _simulate_one_round(self, our_hand, sniffer_state, rng,
                            offline_state=None, base_env=None):
        """Simule UNE manche complete dans un environnement independant."""
        from mahjong_ai.envs.multi_agent import (
            DISCARD, PONG, CHOW, KONG, RIICHI, RON,
        )
        from mahjong_ai.envs.multi_agent import MultiAgentMahjongEnv

        MAX_STEPS = 500

        env = MultiAgentMahjongEnv(
            seed=int(rng.randint(0, 2**31 - 1)),
            compute_lookahead=False,
        )
        env.rng = np.random.RandomState(int(rng.randint(0, 2**31 - 1)))

        sampled = _sample_round_state(our_hand, rng)
        original_deal = env._deal

        def custom_deal():
            original_deal()
            env.hands[0] = sampled['hands'][0].copy()
            env.hands[1] = sampled['hands'][1].copy()
            env.hands[2] = sampled['hands'][2].copy()
            env.hands[3] = sampled['hands'][3].copy()
            # [ROUGES] les mains echantillonnees n'ont pas de rouge
            if hasattr(env, '_clear_reds'):
                env._clear_reds()
            env.wall = list(sampled['wall'])
            env.rinshan_pool = list(sampled['dead_wall'][:RINSHAN_SIZE])
            env.dora_indicators = list(sampled['dora_indicators'])
            env.ura_dora_indicators = list(sampled['ura_dora_indicators'])
            env._initial_wall_size = len(env.wall)

        env._deal = custom_deal

        trajectory = []
        t0 = time.time()
        final_reward = 0.0
        step_count = 0

        try:
            req = env.reset_round()

            while True:
                step_count += 1
                if step_count > MAX_STEPS:
                    if self.verbose:
                        print(f"[pMCPA-DBG] MAX_STEPS atteint ({MAX_STEPS})",
                              flush=True)
                    break

                seat = req['seat']
                obs = req.get('obs')
                mask = req.get('mask')
                atype = req.get('action_type')

                if atype == DISCARD:
                    model = self.discard
                elif atype == KONG:
                    model = self.frozen['kong']
                elif atype == RIICHI:
                    model = self.frozen['riichi']
                elif atype == PONG:
                    model = self.frozen['pong']
                elif atype == CHOW:
                    model = self.frozen['chow']
                elif atype == RON:
                    action = (1 if mask is not None and len(mask) > 1
                              and mask[1] > 0.5 else 0)
                    result = env.step(action)
                    if result.get('round_done'):
                        fr = result.get('final_rewards', {}) or {}
                        final_reward = self.compute_contextual_reward(
                            sniffer_state, float(fr.get(0, 0.0)))
                        break
                    req = result['next_request']
                    continue
                else:
                    result = env.step(None)
                    if result.get('round_done'):
                        fr = result.get('final_rewards', {}) or {}
                        final_reward = self.compute_contextual_reward(
                            sniffer_state, float(fr.get(0, 0.0)))
                        break
                    req = result['next_request']
                    continue

                if obs is None or mask is None:
                    result = env.step(None)
                    if result.get('round_done'):
                        fr = result.get('final_rewards', {}) or {}
                        final_reward = self.compute_contextual_reward(
                            sniffer_state, float(fr.get(0, 0.0)))
                        break
                    req = result['next_request']
                    continue

                # [FIX] Prepare obs selon le modele (272 ou 276 canaux)
                obs_model = _prepare_obs_for_model(obs, model)
                obs_t = torch.as_tensor(obs_model, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                         device=self.device).unsqueeze(0)

                with torch.no_grad():
                    logits = model(obs_t)
                    masked = logits + (mask_t - 1.0) * 1e9
                    dist = torch.distributions.Categorical(logits=masked)
                    action = dist.sample()
                    logp = dist.log_prob(action).item()

                action_int = int(action.item())

                if atype == DISCARD:
                    trajectory.append({
                        'obs': obs.copy(),
                        'mask': mask.copy(),
                        'action': action_int,
                        'logp': logp,
                    })

                result = env.step(action_int)

                if result.get('round_done'):
                    fr = result.get('final_rewards', {}) or {}
                    final_reward = self.compute_contextual_reward(
                        sniffer_state, float(fr.get(0, 0.0)))
                    break
                req = result['next_request']

            return {
                'trajectory': trajectory,
                'reward': final_reward,
                't0': t0,
            }

        except Exception as e:
            if self.verbose:
                print(f"[pMCPA] sim echouee apres {step_count} steps : {e}",
                      flush=True)
                import traceback
                traceback.print_exc()
            return None

    # ------------------------------------------------------------------ #
    def _adapt_with_importance_sampling(self, trajectories):
        """grad_steps updates avec IS pondere par R(tau).

        [FIX] Modele en eval() pendant tout le bloc (BN buffers figes,
        autograd stable). Utilise _prepare_obs_for_model pour chaque obs.
        """
        if self._adapt_optim is None:
            self._adapt_optim = torch.optim.Adam(
                self.discard.parameters(), lr=self.lr)

        self.discard.eval()

        for step_i in range(self.grad_steps):
            self._adapt_optim.zero_grad()

            total_loss = 0.0
            n_valid = 0

            for traj in trajectories:
                transitions = traj['trajectory']
                if not transitions:
                    continue

                R = float(traj['reward'])
                if R == 0.0:
                    continue

                logp_sum = None
                for t in transitions:
                    # [FIX] Prepare obs selon le modele (272 ou 276)
                    obs_data = _prepare_obs_for_model(t['obs'], self.discard)
                    obs_t = torch.as_tensor(
                        obs_data, dtype=torch.float32,
                        device=self.device).unsqueeze(0)
                    mask_t = torch.as_tensor(
                        t['mask'], dtype=torch.float32,
                        device=self.device).unsqueeze(0)

                    logits = self.discard(obs_t)
                    masked = logits + (mask_t - 1.0) * 1e9
                    dist = torch.distributions.Categorical(logits=masked)
                    a_t = torch.tensor([t['action']], dtype=torch.long,
                                        device=self.device)
                    step_logp = dist.log_prob(a_t).sum()
                    if logp_sum is None:
                        logp_sum = step_logp
                    else:
                        logp_sum = logp_sum + step_logp

                if logp_sum is None:
                    continue

                logp_old = sum(t['logp'] for t in transitions)
                log_ratio = logp_sum - logp_old
                ratio = torch.exp(log_ratio)

                total_loss = total_loss - R * ratio * logp_sum
                n_valid += 1

            if n_valid == 0:
                break

            total_loss = total_loss / n_valid
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discard.parameters(), 5.0)
            self._adapt_optim.step()

        self.discard.eval()

    # ------------------------------------------------------------------ #
    def run_all_heads_adapted(self, obs_normal, obs_oracle=None):
        """Forward avec la policy ADAPTEE (theta_a).

        [FIX] Utilise _prepare_obs_for_model pour chaque modele.
        """
        results = {}
        with torch.no_grad():
            for head in ['discard', 'riichi', 'chow', 'pong', 'kong']:
                if head not in self.models:
                    continue
                model = self.discard if head == 'discard' else self.models[head]
                obs_input = _prepare_obs_for_model(obs_normal, model)
                obs_t = torch.as_tensor(obs_input, dtype=torch.float32,
                                         device=self.device).unsqueeze(0)
                logits = model(obs_t)[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                results[head] = probs
        return results

    # ------------------------------------------------------------------ #
    def wait_adaptation(self, timeout=None):
        """Bloque jusqu'a ce que l'adaptation soit terminee (ou timeout)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout or self.timeout_s)