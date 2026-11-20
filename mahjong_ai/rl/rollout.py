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
    __slots__ = ('env', 'idx', 'req', 'agent_seat', 'transitions',
                 'pending_reward', 'round_idx_per_seat', 'round_counter',
                 'grp_delta_sum', 'max_honba', 'stats',
                 'env_stats_before', 'diag_all', 'diag_rounds',
                 'diag_round_rows', 'last_discard_danger',
                 'agent_round_seat')

    def __init__(self, env, agent_seat, idx=0):
        self.env = env
        self.idx = idx                    # [ÉTAPE 3] id d'env (appariement)
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
        # [DIAG DEAL-INS] tampons d'instrumentation (inertes sans diag)
        self.diag_all = []
        self.diag_rounds = []
        self.diag_round_rows = []
        # [DIAG v2] la DERNIÈRE défausse RL de la manche courante
        # était-elle dangereuse (label contrefactuel) ? Base du marquage
        # dealt_in et des compteurs "deal-ins payés" (voir _after_step).
        self.last_discard_danger = False
        # [DIAG v4] siège de l'agent DANS le round_env (les sièges du
        # round_env sont tournés : game_seat = (dealer + round_seat)%4,
        # cf. _seat_mapping). Capturé au DÉBUT de chaque manche car
        # dealer_seat est déjà celui de la manche SUIVANTE quand
        # _after_step lit les internes du round_env terminé.
        self.agent_round_seat = None

    def new_game(self, agent_seat):
        self.agent_seat = agent_seat
        self.req = None
        self.transitions = []
        self.pending_reward = {s: 0.0 for s in range(4)}
        self.round_idx_per_seat = {s: [] for s in range(4)}
        self.round_counter = 0
        self.grp_delta_sum = {s: 0.0 for s in range(4)}
        self.max_honba = 0
        self.diag_all = []
        self.diag_rounds = []
        self.diag_round_rows = []
        self.last_discard_danger = False
        self.agent_round_seat = None
        self.stats = {
            'env_idx': self.idx,          # [ÉTAPE 3] clé d'appariement
            'n_rounds': 0, 'n_tsumo': 0, 'n_ron': 0, 'n_exhaustive': 0,
            'n_abortive': 0, 'n_unhandled': 0, 'n_rl_discards': 0,
            'n_danger_discards': 0,
            # [DIAG v2] deal-ins RÉELLEMENT payés (ron sur MA défausse
            # dangereuse) vs paiements non imputables (baton riichi,
            # honba, chankan) -- cf. _after_step
            'n_dealins_paid': 0, 'points_paid': 0.0,
            'n_ron_paid_nodanger': 0,
            # [DIAG v3] chankans payés (ron adverse sur MON shouminkan)
            'n_chankan_paid': 0, 'points_chankan': 0.0,
            # [DIAG v4] arbitrage vrai deal-in (tuile gagnante venue de
            # l'agent, lue dans le round_env) vs fantôme (paiement
            # imputé à tort par la convention ldd : bâton riichi,
            # honba) -- cf. _after_step
            'n_dealins_true': 0, 'points_true': 0.0,
            'n_dealins_ghost': 0, 'points_ghost': 0.0,
            'n_wfa_noldd': 0,
        }
        self.env_stats_before = self.env.get_stats()
        self.req = self.env.reset_round()
        # [DIAG v4] siège round_env de l'agent pour CETTE manche
        self.agent_round_seat = ((agent_seat - self.env.dealer_seat) % 4
                                 if agent_seat is not None else None)


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
            self.slots.append(_Slot(env, agent_seat=None, idx=i))
        self.seat_rng = np.random.RandomState((seed or 0) + 7777)
        # [DIAG DEAL-INS] instrumentation PURE : active uniquement si
        # l'agent expose pop_diag() (mahjong_ai.rl.diag_dealins) ; sinon
        # aucun des hooks ci-dessous ne fait quoi que ce soit (aucun
        # effet sur dynamiques/récompenses/trajectoires).
        self.diag_active = callable(getattr(agent, 'pop_diag', None))
        self.diag_games = []
        la_str = 'ON' if compute_lookahead else (
            "OFF (80 canaux lookahead mis a zero a l'execution : OK "
            "seulement si le SL de depart fut entraine SANS lookahead "
            "-- verifier avec mahjong_ai.check_sl_lookahead)")
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
                                 grp_terminal_correction, return_rho)
                if result.get('game_done', False):
                    episodes.append(slot.transitions)
                    all_stats.append(self._finish_game(slot))
                    # [DIAG DEAL-INS] récupération avant réutilisation
                    # du slot par la partie suivante
                    if self.diag_active:
                        self.diag_games.append({
                            'env_idx': slot.idx,
                            'game': sum(1 for g in self.diag_games
                                        if g['env_idx'] == slot.idx),
                            'rows': slot.diag_all,
                            'rounds': slot.diag_rounds})
                        slot.diag_all = []
                        slot.diag_rounds = []
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
            # [DIAG DEAL-INS] l'agent (diag_dealins) rend une ligne par
            # requête du batch (None si non armé) ; purement passif ici.
            diag_rows = (self.agent.pop_diag()
                         if self.diag_active else None)
            extras = []
            for i, r in enumerate(reqs):
                extras.append({
                    'diag': (diag_rows[i]
                             if diag_rows is not None else None),
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
        # [ÉTAPE 3] comptage dng% AUSSI en mode éval (collect=False) :
        # l'instrumentation de eval_matchup restait muette car ce bloc
        # n'était atteint qu'en collecte. Instrumentation pure (stats
        # déjà existantes), zéro effet sur dynamiques/récompenses.
        if ex['danger'] is not None:
            slot.stats['n_rl_discards'] += 1
            if float(ex['danger'][action]) > 0:
                slot.stats['n_danger_discards'] += 1
            # [DIAG v2] mémorisé pour la fin de manche : le ron payé
            # n'est un deal-in QUE si la dernière défausse était
            # dangereuse (instrumentation pure, zéro dynamique)
            slot.last_discard_danger = float(ex['danger'][action]) > 0
            # [DIAG DEAL-INS] enrichit la ligne rendue par l'agent avec
            # ce que seul le runner connaît (label danger contrefactuel
            # 34 voies, masque, action) ; inert sans diag.
            row = ex.get('diag')
            if row is not None:
                row['action'] = int(action)
                row['danger'] = np.asarray(ex['danger'],
                                            dtype=np.uint8)
                row['mask'] = (np.asarray(ex['mask']) > 0)
                row['round'] = slot.round_counter
                row['env_idx'] = slot.idx
                slot.diag_round_rows.append(row)
        if not collect:
            return
        seat = req['seat']
        # reward porté : rewards de manche en attente (siège sans requête
        # dans la manche précédente) + pénalité danger éventuelle
        carried = slot.pending_reward[seat]
        slot.pending_reward[seat] = 0.0
        if ex['danger'] is not None and float(ex['danger'][action]) > 0:
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

    def _after_step(self, slot, result, collect, grp_terminal_correction,
                    return_rho=0.85):
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

        # [DIAG v2] fin de manche. pay < 0 sur une issue en ron n'est un
        # DEAL-IN de l'agent QUE si sa dernière défausse de manche était
        # dangereuse (danger[action] > 0 -- les requêtes RON adverses
        # sont toujours acceptées -> ron réel ⟹ label positif). Sinon le
        # paiement n'est PAS imputable à une défausse : baton riichi
        # perdu (le -1000 est prélevé dans final_rewards pour TOUTE fin
        # de manche, cf. _final de game_env), honba, ou chankan ->
        # compté à part. (diag v1 sur 96 donnes : 11 des 72 "deal-ins"
        # étaient des CHANKANS -- rons sur le shouminkan de l'agent,
        # mains réelles -2.8/-4.9/-6.8, cf. clé is_chankan [DIAG v3] ;
        # 19 des 80 lbl n'ont jamais été ronnés -- mains yakuless, le
        # label ignore le yaku par construction, cf. _danger_labels.)
        # [DIAG v4] compteurs économiques sur les récompenses BRUTES.
        # ATTENTION : final_rewards est FAÇONNÉ par le GRP quand un
        # prédicteur est chargé (game_env._final : grp_rewards =
        # phi_k - phi_{k-1}), ce ne sont PAS des paiements. Le sweep v2d
        # (--grp chargé) a ainsi compté des deltas GRP (55 payés /
        # -328.4 pts) pendant que le diag v3 (grp_path=None -> brutes)
        # comptait les paiements réels (61 / -254.8) sur les MÊMES
        # donnes : canary rd/dd identiques mais dpid/pts incohérents.
        # La clé additive raw_round_rewards (game_env v4) rend les
        # compteurs corrects dans tous les cas ; repli final_rewards =
        # env sans la clé (déjà brut sans prédicteur).
        raw_rewards = result.get('raw_round_rewards')
        if raw_rewards is None:
            raw_rewards = round_rewards
        pay = (float(raw_rewards.get(slot.agent_seat, 0.0))
               if slot.agent_seat is not None else 0.0)
        ronish = outcome in ('ron', 'multi_ron')
        # [DIAG v3] chankan : ron adverse sur le SHOUMINKAN de l'agent
        # (clé additive posée par l'env). Paiement réel mais NI un
        # deal-in de défausse (la tuile fatale est celle du KAN, pas la
        # dernière défausse -- d'où les fantômes v1 "dealt_in sans
        # label"), NI un baton riichi : compté à part.
        ckan = bool(result.get('is_chankan', False))
        # [DIAG v4] arbitrage lu dans le round_env au moment où la
        # manche se termine :
        #  * win_from_agent : la dernière tuile jetée (défausse OU kakan
        #    -- _last_discarder est posé sur les deux chemins, cf.
        #    multi_agent) vient du siège de l'agent. Sur une issue ron,
        #    la tuile gagnante est donc la sienne -> VRAI deal-in ;
        #  * agent_riichi : l'agent avait déclaré riichi dans la manche
        #    -> un paiement hors win_from_agent est alors son bâton
        #    (-1.0) perdu sur le ron d'un autre.
        re_ = slot.env.round_env
        ars = slot.agent_round_seat
        wfa = bool(ars is not None
                   and getattr(re_, '_last_discarder', None) == ars)
        _rii = getattr(re_, 'riichi', None)
        ari = bool(ars is not None and _rii and _rii[ars])
        paid_ron = (ronish or outcome == 'chankan') and pay < 0.0
        if paid_ron:
            if ckan:
                slot.stats['n_chankan_paid'] += 1
                slot.stats['points_chankan'] += float(pay)
            else:
                # [DIAG v4] vérité terrain (convention ldd conservée à
                # part pour la comparabilité v2/v3)
                if wfa:
                    slot.stats['n_dealins_true'] += 1
                    slot.stats['points_true'] += float(pay)
                    if not slot.last_discard_danger:
                        slot.stats['n_wfa_noldd'] += 1
                else:
                    slot.stats['n_dealins_ghost'] += 1
                    slot.stats['points_ghost'] += float(pay)
                if slot.last_discard_danger:
                    slot.stats['n_dealins_paid'] += 1
                    slot.stats['points_paid'] += float(pay)
                else:
                    slot.stats['n_ron_paid_nodanger'] += 1
        if self.diag_active and slot.diag_round_rows:
            if ronish and pay < 0.0:
                if ckan:
                    slot.diag_round_rows[-1]['chankan_paid'] = True
                    slot.diag_round_rows[-1]['chankan_points'] = pay
                elif slot.last_discard_danger:
                    slot.diag_round_rows[-1]['dealt_in'] = True
                    slot.diag_round_rows[-1]['points_paid'] = pay
                else:
                    # baton riichi perdu sur le ron d'un autre (ou honba)
                    # : paiement réel mais PAS un deal-in de défausse
                    slot.diag_round_rows[-1]['stick_lost'] = True
                    slot.diag_round_rows[-1]['stick_points'] = pay
                # [DIAG v4] champs d'arbitrage sur la ligne marquée
                slot.diag_round_rows[-1]['win_from_agent'] = wfa
                slot.diag_round_rows[-1]['agent_riichi'] = ari
            slot.diag_rounds.append(
                {'round': slot.round_counter, 'outcome': outcome,
                 'agent_pay': pay, 'n_rows': len(slot.diag_round_rows),
                 # [DIAG v4] arbitrage par manche (summary)
                 'win_from_agent': (wfa if ronish else None),
                 'agent_riichi': ari})
            slot.diag_all.extend(slot.diag_round_rows)
            slot.diag_round_rows = []
        slot.last_discard_danger = False

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
                # [V2.3b FIX RETURN-RHO] le flag --return-rho atteint enfin
                # finalize_returns. Avant : collect_batch l'acceptait mais ne
                # le transmettait pas -> rho=0.85 en dur, le flag etait un
                # placebo (advvar ~53 = Var(G_t 0.85) dans tous les logs,
                # au lieu de ~24 = Var(r_k) attendu avec --return-rho 0).
                finalize_returns(slot.transitions, rho=return_rho)
        else:
            # manche terminée mais partie en cours -> manche suivante
            slot.req = slot.env.reset_round()
            # [DIAG v4] siège round_env de l'agent pour la manche
            # suivante (cf. commentaire _Slot.__init__)
            slot.agent_round_seat = ((slot.agent_seat
                                      - slot.env.dealer_seat) % 4
                                     if slot.agent_seat is not None
                                     else None)

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
