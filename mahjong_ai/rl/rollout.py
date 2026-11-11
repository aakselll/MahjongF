"""Runner d'envs vectorisés : n_envs parties interleavées, forwards batchés.

C'est le remplacement de training/self_play_worker.py. Une seule policy
FROZÉE pendant toute la collection (on-policy strict) :

    runner.collect_batch(n_games, gamma_t)
        -> (episodes, game_stats)   # episodes JETÉS après l'update PPO

Boucle interne : à chaque tick, chaque partie active a UNE requête en
attente. Les requêtes sont groupées par décideur (policy RL / discard SL /
tête riichi / kong / pong / chow / ron) et traitées en UN forward batché
par groupe. L'env ne voit jamais les modèles SL (frozen_sl=None) : il
yield TOUTES ses requêtes, y compris riichi/appels/ron, que ce runner
résout lui-même -> les décisions SL sont batchées au lieu d'être appelées
une à une dans l'env (ancien _sl_decide, forward batch-1).

Différence de comportement assumée vs l'ancien pipeline : les requêtes RON
sont toujours acceptées (RON_YES). L'ancien chemin (frozen_sl dans l'env)
refusait un ron dans un cas étroit de dernière manche (_should_declare_ron
avec le coût exact de la main) ; en pratique on ron quasi toujours, et ne
plus refuser supprime un chemin de furiten temporaire. Les requêtes
riichi/kong/pong/chow utilisent les MÊMES têtes SL sur les MÊMES obs que
l'ancien chemin -> comportement identique.

Récompenses (fidèle Suphx §3.2 + correction terminale) :
    * le reward de la manche k du siège s = delta GRP  r_k = Phi_k - Phi_{k-1}
      (repli : score de manche /1000 si pas de checkpoint GRP), attribué à
      TOUTES les transitions du siège dans la manche (papier : "assign it
      to each state-action pair in round k") ;
    * [NOUVEAU] correction terminale : à la fin de partie, le dernier
      delta est corrigé de (R_vrai - Phi_final) où R_vrai est le reward de
      rang {1:+15, 2:+5, 3:-5, 4:-15}. Ainsi SOMME(r_k) = R_vrai
      exactement : le shaping GRP devient un potential-based shaping
      exact, sans le biais de prédiction du GRP (à petite échelle le GRP
      est imparfait ; le papier n'a pas besoin de cette correction car
      entraîné sur des millions de parties). Désactivable.
    * G_t = retour restant escompté (rho^(j-k)) des rewards de manches
      futures du siège (cible du critic, héritée du pipeline v9 qui avait
      correctement identifié que le delta de manche seul est imprévisible
      -> critic plat).
"""
import numpy as np
import torch

from mahjong_ai.envs.multi_agent import (
    DISCARD, PONG, CHOW, KONG, RIICHI, RON,
)
from mahjong_ai.training.game_env import GameEnv, PredictorHolder, RANK_REWARDS
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM


def load_grp_predictor(path, device='cpu'):
    """Charge le prédicteur GRP (archi lue dans le checkpoint)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state_dict = ck.get('model_state_dict', ck)
    hidden_dim = ck.get('hidden_dim', 64)
    input_dim = ck.get('feature_dim', FEATURE_DIM)
    grp = RewardPredictor(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_layers=2).to(device)
    grp.load_state_dict(state_dict)
    grp.eval()
    return grp


class _Slot:
    """État d'une partie en cours."""
    __slots__ = ('env', 'req', 'agent_seat', 'transitions',
                 'pending_reward', 'round_idx_per_seat', 'round_counter',
                 'grp_delta_sum', 'max_honba', 'stats',
                 'env_stats_before')

    def __init__(self, env, agent_seat):
        self.env = env
        self.agent_seat = agent_seat      # None en self-play
        self.req = None
        self.transitions = []
        self.pending_reward = {s: 0.0 for s in range(4)}
        self.round_idx_per_seat = {s: [] for s in range(4)}
        self.round_counter = 0
        self.grp_delta_sum = {s: 0.0 for s in range(4)}
        self.max_honba = 0
        self.stats = None
        self.env_stats_before = None

    def new_game(self, agent_seat):
        self.agent_seat = agent_seat
        self.req = None
        self.transitions = []
        self.pending_reward = {s: 0.0 for s in range(4)}
        self.round_idx_per_seat = {s: [] for s in range(4)}
        self.round_counter = 0
        self.grp_delta_sum = {s: 0.0 for s in range(4)}
        self.max_honba = 0
        self.stats = {
            'n_rounds': 0, 'n_tsumo': 0, 'n_ron': 0, 'n_exhaustive': 0,
            'n_abortive': 0, 'n_unhandled': 0, 'n_rl_discards': 0,
            'n_danger_discards': 0,
        }
        self.env_stats_before = self.env.get_stats()
        self.req = self.env.reset_round()


def finalize_returns(transitions, rho=0.85):
    """G_t = somme escomptee (rho^(j-r)) des rewards des MANCHES futures
    du siege. Réplique exactement l'algorithme v9 (test_v9_returns.py) :
    groupes (siege, manche) consecutifs, R_g = reward de la DERNIERE
    transition du groupe, correction (reward_i - R_g) pour restaurer les
    penalites individuelles (danger etc.).
    """
    groups = {}
    order = []
    for i, t in enumerate(transitions):
        key = (t['seat'], t.get('round', 0))
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(i)
    suffix = {s: 0.0 for s in set(t['seat'] for t in transitions)}
    G = {}
    for key in reversed(order):
        seat = key[0]
        idxs = groups[key]
        R_g = transitions[idxs[-1]]['reward']
        suffix[seat] = R_g + rho * suffix[seat]
        for i in idxs:
            G[i] = suffix[seat] + (transitions[i]['reward'] - R_g)
    for i, t in enumerate(transitions):
        t['return'] = float(G[i])
    return transitions


class VecRunner:
    """n_envs parties interleavées, décisions batchées par modèle."""

    def __init__(self, agent, n_envs, seed=0, grp_path=None,
                 compute_lookahead=False, danger_labels=False,
                 mode='opponent', device='cpu'):
        self.agent = agent
        self.n_envs = max(1, n_envs)
        self.mode = mode                    # 'opponent' | 'selfplay'
        self.device = device
        self.grp_loaded = grp_path is not None

        grp_model = None
        if grp_path is not None:
            grp_model = load_grp_predictor(grp_path, device)
            print(f"[vec_runner] GRP chargé depuis {grp_path}", flush=True)
        else:
            print("[vec_runner] pas de GRP -> reward = score de manche "
                  "/1000", flush=True)

        self.slots = []
        for i in range(self.n_envs):
            holder = PredictorHolder()
            holder.model = grp_model        # modèle partagé (sans état :
            #                                   les hidden states vivent
            #                                   dans chaque GameEnv)
            env = GameEnv(seed=(seed + i) if seed is not None else None,
                          predictor_holder=holder, game_log_buffer=None,
                          n_rounds=4, game_format='tonpuu',
                          compute_lookahead=compute_lookahead)
            env.round_env.frozen_sl = None  # TOUTES les requêtes nous
            #                                   sont yieldées (batching)
            env.round_env.compute_danger = danger_labels
            self.slots.append(_Slot(env, agent_seat=None))
        self.seat_rng = np.random.RandomState((seed or 0) + 7777)
        la_str = 'ON' if compute_lookahead else (
            'OFF (canaux de toute façon morts, cf. ANALYSE_RL.md)')
        print(f"[vec_runner] {self.n_envs} envs, mode={mode}, "
              f"lookahead={la_str}", flush=True)

    # ------------------------------------------------------------------ #
    def collect_batch(self, n_games, gamma_t, opponent_sample=False,
                      danger_penalty=0.0, collect=True, deterministic=False,
                      gamma_override=None, grp_terminal_correction=True,
                      return_rho=0.85):
        """Joue n_games parties complètes.

        collect=False + deterministic=True + gamma_override=0 : mode ÉVAL
        (déploiement) — aucune transition, argmax partout.

        Retour (episodes, stats) : episodes = liste de listes de
        transitions (seulement les sièges qui apprennent) ; stats = liste
        de dicts de stats par partie.
        """
        active = []
        free = list(self.slots)   # pool de slots inactifs
        n_done = 0
        episodes, all_stats = [], []
        next_slot = 0      # nb de parties déjà lancées

        # Chaque partie en cours occupe un slot pris dans le pool ; un
        # slot libéré par une partie terminée est réutilisé pour la
        # suivante (jamais deux entrées actives sur le même env).
        while n_done < n_games:
            while free and next_slot < n_games:
                slot = free.pop()
                seat = (int(self.seat_rng.randint(4))
                        if self.mode == 'opponent' else None)
                slot.new_game(seat)
                active.append(slot)
                next_slot += 1
            if not active:
                break

            # --- groupage des requêtes par décideur ---
            groups = {}
            for pos, slot in enumerate(active):
                key = self._req_key(slot)
                groups.setdefault(key, []).append(pos)

            decisions = {}
            for key, positions in groups.items():
                if key == 'fallback':
                    # requête inattendue (bug env/runner) : comptée,
                    # le trainer fail-fast si la fenêtre en contient
                    for p in positions:
                        active[p].stats['n_unhandled'] += 1
                reqs = [active[p].req for p in positions]
                actions, extras = self._decide(
                    key, reqs, gamma_t, opponent_sample,
                    deterministic, gamma_override)
                for p, a, ex in zip(positions, actions, extras):
                    decisions[p] = (a, ex)

            # --- application + step ---
            finished = []
            for p in sorted(decisions):
                slot = active[p]
                a, ex = decisions[p]
                self._apply_decision(slot, a, ex, collect, danger_penalty)
                result = slot.env.step(a)
                self._after_step(slot, result, collect,
                                 grp_terminal_correction)
                if result.get('game_done', False):
                    episodes.append(slot.transitions)
                    all_stats.append(self._finish_game(slot))
                    finished.append(p)
            for p in reversed(finished):
                slot = active.pop(p)
                free.append(slot)      # le slot redevient disponible
                n_done += 1

        return episodes, all_stats

    # ------------------------------------------------------------------ #
    def _req_key(self, slot):
        req = slot.req
        t = req['action_type']
        if t == DISCARD:
            if slot.agent_seat is None or req['seat'] == slot.agent_seat:
                return 'rl'
            return 'sl'
        if t == RIICHI:
            return 'riichi'
        if t == KONG:
            return 'kong'
        if t == PONG:
            return 'pong'
        if t == CHOW:
            return 'chow'
        if t == RON:
            return 'ron'
        return 'fallback'      # requête inattendue -> action sûre 0

    def _decide(self, key, reqs, gamma_t, opponent_sample,
                deterministic, gamma_override):
        """Retour (actions, extras) ; extras[i] = dict pour la transition
        (logp/value/oracles) si key == 'rl', sinon None."""
        B = len(reqs)
        if key == 'rl':
            obs_n = np.stack([r['obs'] for r in reqs])
            obs_o = np.stack([r['oracle'] for r in reqs])
            masks = np.stack([r['mask'] for r in reqs])
            g = gamma_t if gamma_override is None else gamma_override
            actions, logps, values, oo_masked = self.agent.act_rl(
                obs_n, obs_o, masks, g, deterministic=deterministic)
            extras = []
            for i, r in enumerate(reqs):
                extras.append({
                    'obs_normal': np.asarray(r['obs'], dtype=np.float16),
                    'obs_oracle': oo_masked[i],
                    'obs_oracle_full': np.asarray(r['oracle'],
                                                  dtype=np.float16),
                    'mask': np.asarray(r['mask'], dtype=np.float16),
                    'danger': (None if r.get('danger') is None
                               else np.asarray(r['danger'],
                                               dtype=np.uint8)),
                    'logp': logps[i],
                    'value': values[i],
                })
            return actions, extras
        if key == 'sl':
            obs_n = np.stack([r['obs'] for r in reqs])
            masks = np.stack([r['mask'] for r in reqs])
            actions = self.agent.act_sl_discard(
                obs_n, masks, sample=opponent_sample)
            return actions, [None] * B
        if key == 'ron':
            return np.ones(B, dtype=np.int64), [None] * B
        if key == 'fallback':
            return np.zeros(B, dtype=np.int64), [None] * B
        # tête SL figée (riichi/kong/pong/chow)
        actions = self.agent.act_head(
            key, [r['obs'] for r in reqs], [r['mask'] for r in reqs])
        return actions, [None] * B

    def _apply_decision(self, slot, action, ex, collect, danger_penalty):
        """Enregistre la transition RL (si collecte) AVANT le step."""
        req = slot.req
        if ex is None:
            return
        if not collect:
            return
        seat = req['seat']
        # reward porté : rewards de manche en attente (siège sans requête
        # dans la manche précédente) + pénalité danger éventuelle
        carried = slot.pending_reward[seat]
        slot.pending_reward[seat] = 0.0
        if ex['danger'] is not None:
            slot.stats['n_rl_discards'] += 1
            if float(ex['danger'][action]) > 0:
                slot.stats['n_danger_discards'] += 1
                carried -= danger_penalty
        tr = {
            'seat': seat,
            'round': slot.round_counter,
            'obs_normal': ex['obs_normal'],
            'obs_oracle': ex['obs_oracle'],
            'obs_oracle_full': ex['obs_oracle_full'],
            'mask': ex['mask'],
            'danger': ex['danger'],
            'action': int(action),
            'logp': float(ex['logp']),
            'value': float(ex['value']),
            'reward': float(carried),
        }
        slot.transitions.append(tr)
        slot.round_idx_per_seat[seat].append(len(slot.transitions) - 1)

    def _after_step(self, slot, result, collect, grp_terminal_correction):
        if not result['round_done']:
            slot.req = result['next_request']
            return

        outcome = result.get('outcome', 'unknown')
        if outcome == 'tsumo':
            slot.stats['n_tsumo'] += 1
        elif outcome in ('ron', 'multi_ron'):
            slot.stats['n_ron'] += 1
        elif outcome == 'exhaustive':
            slot.stats['n_exhaustive'] += 1
        elif outcome.startswith('abortive'):
            slot.stats['n_abortive'] += 1
        slot.stats['n_rounds'] += 1
        slot.round_counter += 1
        slot.max_honba = max(slot.max_honba, slot.env.honba)

        # attribution du reward de manche à TOUTES les transitions du
        # siège dans cette manche (papier §3.2)
        round_rewards = result['final_rewards']
        for s in range(4):
            r_s = float(round_rewards.get(s, 0.0))
            r_s = max(-20.0, min(20.0, r_s))
            if r_s == 0.0:
                continue
            slot.grp_delta_sum[s] += r_s
            idxs = slot.round_idx_per_seat.get(s, [])
            if idxs:
                for idx in idxs:
                    slot.transitions[idx]['reward'] += r_s
            else:
                slot.pending_reward[s] += r_s
        slot.round_idx_per_seat = {s: [] for s in range(4)}

        if result.get('game_done', False):
            # rewards pending -> dernière transition du siège
            for s in range(4):
                if slot.pending_reward[s] != 0.0:
                    last = self._last_transition_of(slot, s)
                    if last is not None:
                        last['reward'] += slot.pending_reward[s]
                    slot.pending_reward[s] = 0.0
            # correction terminale GRP : SOMME(r_k) = reward de rang VRAI
            if self.grp_loaded and grp_terminal_correction:
                scores = slot.env.game_scores
                order = sorted(range(4), key=lambda p: (-scores[p], p))
                for rank, p in enumerate(order, start=1):
                    correction = (RANK_REWARDS[rank]
                                  - slot.grp_delta_sum[p])
                    if abs(correction) < 1e-9:
                        continue
                    last = self._last_transition_of(slot, p)
                    if last is not None:
                        last['reward'] += correction
            if slot.transitions:
                finalize_returns(slot.transitions)
        else:
            # manche terminée mais partie en cours -> manche suivante
            slot.req = slot.env.reset_round()

    @staticmethod
    def _last_transition_of(slot, seat):
        for t in reversed(slot.transitions):
            if t['seat'] == seat:
                return t
        return None

    def _finish_game(self, slot):
        """Stats finales de la partie (clés = ancien format pour réutiliser
        le formatage de logs du trainer)."""
        scores = slot.env.game_scores
        order = sorted(range(4), key=lambda p: (-scores[p], p))
        tracked = 0 if slot.agent_seat is None else slot.agent_seat
        slot.stats['rank_of_seat0'] = order.index(tracked) + 1
        slot.stats['n_honba'] = slot.max_honba
        after = slot.env.get_stats()

        def _d(*keys):
            return sum(int(after.get(k, 0))
                       - int(slot.env_stats_before.get(k, 0))
                       for k in keys)

        slot.stats['n_riichi'] = _d('riichi')
        slot.stats['n_pon'] = _d('pon_calls')
        slot.stats['n_chi'] = _d('chi_calls')
        slot.stats['n_kan'] = _d('ankan', 'shouminkan', 'daiminkan')
        slot.stats['n_ron_refused'] = _d('ron_refused')
        slot.stats['n_sl_errors'] = _d('sl_errors')
        # requêtes inattendues (fallback) -> comptées comme unhandled
        return slot.stats
