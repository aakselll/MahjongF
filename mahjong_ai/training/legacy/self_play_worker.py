"""
Worker self-play : utilise GameEnv (multi-round) au lieu de
MultiAgentMahjongEnv (single round).

[FIX 2025-XX] L'ancien code utilisait MultiAgentMahjongEnv, ce qui faisait
qu'une "partie" etait en realite UNE seule manche. Maintenant : 4 rounds
East 1-4 + repeat dealer + GRP reward.

[NIVEAU 2 / FIX IPC 2026-10-14] Les transitions passent par une mp.Queue
    au lieu d'une liste proxy Manager :
    - l'ancien chemin tuait le serveur Manager sous Windows (~75 min
      observes) ;
    - la Queue est le mecanisme prevu pour le transfert de donnees
      volumineuses (pipes dedies, backpressure via maxsize) ;
    - put(timeout=30) : si le trainer est mort, le worker s'arrete
      proprement au lieu de bloquer indefiniment sur une Queue pleine.

    Signature changee vs Niveau 1 : (buffer_data, buffer_max) ->
    transitions_queue. Le trainer passe la Queue en 4e argument.

[STATS] stats_proxy partage (Manager dict) qui accumule les statistiques
    de jeu (rangs, longueur, outcomes). Volume leger, frequence moderee :
    reste tolerable pour le Manager.
"""
import os
import time

import numpy as np
import torch

from mahjong_ai.training.game_env import GameEnv, PredictorHolder
from mahjong_ai.agents.oracle_agent import OracleDiscardAgent
from mahjong_ai.training.parameter_server import fetch_from_state
from mahjong_ai.envs.multi_agent import DISCARD
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM


def worker_loop(worker_id, state, lock, transitions_queue,
                stats_proxy,
                sl_discard_path, sl_frozen_dir,
                grp_path=None,
                n_games_per_cycle=10, device='cpu',
                opponent_sl=False, opponent_mode='argmax',
                danger_penalty=0.0):
    """
    [NIVEAU 2] Les transitions passent par une mp.Queue au lieu d'une
    liste proxy Manager.

    Args :
        worker_id          : id du worker (pour logs + seed GameEnv)
        state, lock        : proxies Manager du parameter server
        transitions_queue  : mp.Queue partagee avec le trainer
        stats_proxy        : proxy Manager dict pour les stats de jeu
        sl_discard_path    : chemin du checkpoint SL discard (warm-start)
        sl_frozen_dir      : dossier des tetes SL gelees
        grp_path           : chemin du checkpoint GRP (optionnel)
        n_games_per_cycle  : parties jouees avant re-sync des poids
        device             : cpu
        opponent_sl        : True -> 1 siege RL (tire au hasard par partie)
                             + 3 sieges SL figes ; False -> self-play
        opponent_mode      : 'argmax' | 'sample' pour les SL adversaires
        danger_penalty     : > 0 -> penalite sur chaque decision du siege
                             RL qui defausse une tuile 'dangereuse'
                             (ron immediat pour un adversaire tenpai,
                             calcule depuis les mains cachees). En
                             phase 1 l'oracle rend ce label LISIBLE ->
                             la policy apprend a s'en servir.
    """
    # [FIX] Limite les threads PyTorch par worker pour eviter
    # l'oversubscription.
    n_workers_total = int(os.environ.get("RL_N_WORKERS", "1"))
    threads_per_worker = max(1, torch.get_num_threads() // max(n_workers_total, 1))
    torch.set_num_threads(threads_per_worker)
    print(f"[worker {worker_id}] torch threads = {threads_per_worker}",
          flush=True)

    agent = OracleDiscardAgent(sl_discard_path, sl_frozen_dir, device=device)

    # --- GRP ---
    predictor_holder = PredictorHolder()
    if grp_path is not None:
        try:
            ck = torch.load(grp_path, map_location=device, weights_only=False)
            # [FIX] Le checkpoint est un dict avec metadonnees -- extraire
            # le state_dict et lire l'archi dedans (hidden_dim=64 sauvegarde,
            # ne pas hardcoder 128).
            state_dict = ck.get("model_state_dict", ck)
            hidden_dim = ck.get("hidden_dim", 64)
            input_dim = ck.get("feature_dim", FEATURE_DIM)
            grp = RewardPredictor(input_dim=input_dim, hidden_dim=hidden_dim,
                                  num_layers=2).to(device)
            grp.load_state_dict(state_dict)
            grp.eval()
            predictor_holder.model = grp
            print(f"[worker {worker_id}] GRP charge (hidden={hidden_dim}, "
                  f"val_loss={ck.get('val_loss', '?')})", flush=True)
        except Exception as e:
            print(f"[worker {worker_id}] GRP load error : {e}", flush=True)

    # --- GameEnv 4-round East ---
    env = GameEnv(seed=worker_id, predictor_holder=predictor_holder,
                  game_log_buffer=None, n_rounds=4,
                  game_format="tonpuu")
    env.round_env.frozen_sl = agent.frozen_sl
    # [DANGER] labels de danger dans les requetes DISCARD si penalite
    # [DANGER LOSS] MJ_DANGER_LABELS=1 (pose par train_rl si
    # --danger-loss-coef > 0) active aussi les labels sans penalite.
    env.round_env.compute_danger = bool(
        danger_penalty > 0 or os.environ.get('MJ_DANGER_LABELS') == '1')
    print(f"[worker {worker_id}] demarre (n_rounds=4, "
          f"opponent_sl={opponent_sl}, danger_penalty={danger_penalty})",
          flush=True)
    seat_rng = np.random.RandomState(1000 + worker_id)

    while True:
        policy_state, critic_state, step, alpha, gamma_t = \
            fetch_from_state(state, lock)
        if policy_state is not None:
            try:
                agent.policy.load_state_dict(policy_state)
                agent.critic.load_state_dict(critic_state)
            except Exception as e:
                print(f"[worker {worker_id}] load error : {e}", flush=True)
        # [FIX BATCHNORM] rollouts toujours en eval (stats BN du SL).
        agent.set_eval_mode()

        for _ in range(n_games_per_cycle):
            # [ADVERSAIRES SL] siege RL aleatoire : GameEnv demarre toujours
            # avec le joueur 0 comme donneur -> tirer le siege RL randomise
            # aussi le vent de depart / l'ordre de jeu.
            agent_player = int(seat_rng.randint(4)) if opponent_sl else None
            transitions, game_stats = _play_one_game(
                env, agent, gamma_t, agent_player=agent_player,
                opponent_argmax=(opponent_mode == 'argmax'),
                danger_penalty=danger_penalty)
            if transitions:
                # [NIVEAU 2] put au lieu de buffer_add.
                # timeout : ne pas bloquer indefiniment si le trainer est
                # mort (Queue pleine -> exception -> arret propre).
                # [FIX MEMOIRE] La Queue est volontairement PETITE
                # (backpressure) : une file momentanement pleine est normale
                # quand le trainer est occupe. On reessaie (6 x 30 s) avant
                # de conclure que le trainer est mort.
                put_ok = False
                for _attempt in range(6):
                    try:
                        transitions_queue.put(transitions, timeout=30.0)
                        put_ok = True
                        break
                    except Exception:
                        print(f"[worker {worker_id}] Queue pleine/indisponible "
                              f"(essai {_attempt + 1}/6)", flush=True)
                if not put_ok:
                    print(f"[worker {worker_id}] Queue indisponible -- "
                          f"arret du worker", flush=True)
                    return
                transitions = None      # libere la partie des que envoyee
            # [STATS] mise a jour du proxy partage
            if game_stats is not None:
                _update_stats(stats_proxy, lock, game_stats)


def _update_stats(stats_proxy, lock, game_stats):
    """Accumule les stats d'une partie dans le proxy partage.

    [FIX 2026-10-14] Aligne sur le format REEL de _play_one_game :
      - rang          : cle 'rank_of_seat0' (pas 'rank')
      - outcomes      : compteurs CUMULES par partie (n_tsumo/n_ron/
                        n_exhaustive/n_abortive somment les manches),
                        pas une cle 'outcome' unique
      - n_rounds/n_honba : cles deja correctes
    """
    if game_stats is None:
        return
    with lock:
        stats_proxy['n_games'] = stats_proxy.get('n_games', 0) + 1
        rank = game_stats.get('rank_of_seat0', 0)
        if rank == 1:
            stats_proxy['n_1st'] = stats_proxy.get('n_1st', 0) + 1
        elif rank == 2:
            stats_proxy['n_2nd'] = stats_proxy.get('n_2nd', 0) + 1
        elif rank == 3:
            stats_proxy['n_3rd'] = stats_proxy.get('n_3rd', 0) + 1
        elif rank == 4:
            stats_proxy['n_4th'] = stats_proxy.get('n_4th', 0) + 1
        stats_proxy['sum_rank'] = stats_proxy.get('sum_rank', 0.0) + rank
        stats_proxy['n_rounds_total'] = stats_proxy.get('n_rounds_total', 0) \
            + game_stats.get('n_rounds', 0)
        stats_proxy['n_honba_total'] = stats_proxy.get('n_honba_total', 0) \
            + game_stats.get('n_honba', 0)
        # [FIX] Outcomes : compteurs cumules par manche dans la partie --
        # on ajoute directement les compteurs de la partie.
        stats_proxy['n_tsumo'] = stats_proxy.get('n_tsumo', 0) \
            + game_stats.get('n_tsumo', 0)
        stats_proxy['n_ron'] = stats_proxy.get('n_ron', 0) \
            + game_stats.get('n_ron', 0)
        stats_proxy['n_exhaustive'] = stats_proxy.get('n_exhaustive', 0) \
            + game_stats.get('n_exhaustive', 0)
        stats_proxy['n_abortive'] = stats_proxy.get('n_abortive', 0) \
            + game_stats.get('n_abortive', 0)
        # [STATS] riichi / appels / refus de ron / erreurs SL / requetes
        # non gerees (doivent rester a 0 : sinon une tete SL figee est
        # cassee).
        for k in ('n_riichi', 'n_pon', 'n_chi', 'n_kan', 'n_ron_refused',
                  'n_sl_errors', 'n_unhandled', 'n_rl_discards',
                  'n_danger_discards'):
            stats_proxy[k] = stats_proxy.get(k, 0) + game_stats.get(k, 0)

def _finalize_episode_returns(transitions, rho=0.85):
    """[V9 CRITIC] Retour restant de partie par decision : G_t = somme
    escomptee (rho^(j-r)) des rewards des manches FUTURES du siege.

    Pourquoi : la cible du critic v8 etait le delta GRP de la MANCHE
    courante -- une return Monte Carlo determinee par l'issue de la
    manche (le futur). Probe sur le checkpoint 200k : EV=0.0000,
    val_std=0.0001, ridge R2=-0.87, critic frais R2_frais=-1.28 ->
    E[R|s] est plat, le critic degenerait en constante. Le RETOUR
    RESTANT de partie, lui, est previsible depuis l'etat (own/opp
    scores en buckets, manche, honba + oracle sont dans les 283
    canaux ; le GRP lui-meme le prouve sur 12 features grossieres).
    A = G_t - V(s_t) reste sans biais (V ne depend que de s).

    Groupe les transitions par (siege, manche) consecutives ; R_g =
    reward de la DERNIERE transition du groupe (le backfill touche tout
    le groupe a l'identique). Les corrections (carried/pending, qui
    n'affectent que la 1re/derniere transition) sont restaurees par
    le terme (t['reward'] - R_g) : chaque transition garde exactement
    les rewards qui lui ont ete attribues.
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


def _play_one_game(env, agent, gamma_t, agent_player=None,
                   opponent_argmax=True, return_rho=0.85,
                   danger_penalty=0.0):
    """Joue une partie COMPLETE (4 rounds + repeats).

    [FIX MULTI-SEAT REWARD] Le reward de la manche k est attribue a
    TOUTES les transitions du siege dans cette manche (papier : "assign
    it to each state-action pair in round k"). L'ancien code ne
    l'assignait qu'a la DERNIERE transition -- 98 pourcent des
    transitions avaient reward=0, diluant le signal d'apprentissage par
    ~50x.

    [V9 CRITIC] Chaque transition porte en plus 'round' (index de
    manche) et, en fin de partie, 'return' = G_t (retour restant
    escompte, cible du critic -- voir _finalize_episode_returns).

    [DANGER] si danger_penalty > 0, chaque decision du siege RL qui
    defausse une tuile dangereuse (ron immediat pour un tenpai) prend
    la penalite sur SA reward -- et uniquement elle : dans
    _finalize_episode_returns, le terme de correction (reward_i - R_g)
    garde la penalite sur la transition coupable et l'annule pour les
    autres transitions de la manche (credit par-decision exact).

    [ADVERSAIRES SL] agent_player=None : self-play (4 sieges RL).
    agent_player=k : seul le joueur k joue avec la policy RL et produit des
    transitions ; les 3 autres jouent la policy SL figee (sans transition).
    """
    transitions = []
    pending_reward = {s: 0.0 for s in range(4)}
    # [FIX] Tous les indices par siege pour la manche courante
    round_indices_per_seat = {s: [] for s in range(4)}
    # [V9] Index de manche pose sur chaque transition (pour G_t)
    round_counter = 0
    max_honba = 0

    stats = {
        'n_rounds': 0,
        'n_tsumo': 0,
        'n_ron': 0,
        'n_exhaustive': 0,
        'n_abortive': 0,
        'n_unhandled': 0,
        'n_rl_discards': 0,
        'n_danger_discards': 0,
    }
    # [STATS] snapshot des compteurs cumules de l'env (par process) pour
    # en deduire les deltas de CETTE partie : riichi, appels, ron refuses.
    env_stats_before = env.get_stats()

    req = env.reset_round()
    while True:
        seat = req['seat']
        action = None
        env_action = None
        logp = value = 0.0
        obs_n = obs_o = mask = None
        danger = None

        if req['action_type'] == DISCARD and agent_player is not None \
                and seat != agent_player:
            # Adversaire SL fige : pas de transition enregistree.
            env_action = agent.select_action_sl(
                req['obs'], req['mask'], argmax=opponent_argmax)
        elif req['action_type'] == DISCARD:
            obs_n = req['obs']
            obs_o = req['oracle']
            mask = req['mask']
            danger = req.get('danger')
            action, logp, value, masked_obs_o = agent.select_action(
                obs_n, obs_o, mask, gamma_t)
            env_action = action
        else:
            # [FIX] Avec les tetes SL figees, l'env ne doit yield QUE des
            # DISCARD. Une autre requete (tete manquante / erreur SL) etait
            # renvoyee avec action=None : pour CHOW, None != CHOW_PASS ->
            # chow "upper" applique meme s'il est illegal. On repond par
            # l'action sure (0 = passer / non) et on compte l'evenement.
            stats['n_unhandled'] += 1
            env_action = 0

        result = env.step(env_action)

        if action is not None:
            # Un siege peut ne recevoir aucune requete DISCARD dans une
            # manche terminee immediatement. Conserver son reward jusqu'a
            # sa prochaine transition plutot que le perdre.
            carried_reward = pending_reward[seat]
            pending_reward[seat] = 0.0
            # [DANGER] penalite par-decision : tuile qui ferait ron.
            # Entre dans t['reward'] AVANT le backfill du delta de la
            # manche -> le credit est exact (voir docstring).
            if danger is not None:
                stats['n_rl_discards'] += 1
                if float(danger[action]) > 0:
                    stats['n_danger_discards'] += 1
                    carried_reward -= danger_penalty
            transitions.append({
                'seat': seat,
                # [FIX MEMOIRE] float16 : ~18 Ko au lieu de ~36 Ko par coup.
                # Les canaux sont binaires (hands >= n) ou multiples de 1/4
                # (unseen/4) : exacts en float16. pg_is_loss reconvertit en
                # float32 (torch.as_tensor(..., dtype=float32)).
                'obs_normal': np.asarray(obs_n, dtype=np.float16),
                'obs_oracle': masked_obs_o,
                # [CRITIC] oracle COMPLET (petit : 4x34 floats) pour le
                # critic asymetrique / la baseline V(s).
                'obs_oracle_full': np.asarray(obs_o, dtype=np.float32),
                'mask': mask,
                # [DANGER LOSS] label exact (34,) : 1 = ferait ron
                'danger': (None if danger is None
                           else np.asarray(danger, dtype=np.uint8)),
                'action': action,
                'logp': logp,
                'value': value,
                'reward': carried_reward,
                'round': round_counter,
                'done': False,
            })
            # [FIX] Enregistrer TOUTES les transitions du siege
            round_indices_per_seat[seat].append(len(transitions) - 1)

        if result['round_done']:
            outcome = result.get('outcome', 'unknown')
            if outcome == 'tsumo':
                stats['n_tsumo'] += 1
            elif outcome in ('ron', 'multi_ron'):
                stats['n_ron'] += 1
            elif outcome == 'exhaustive':
                stats['n_exhaustive'] += 1
            elif outcome.startswith('abortive'):
                stats['n_abortive'] += 1
            stats['n_rounds'] += 1
            # [V9] La manche est terminee : les transitions suivantes
            # appartiennent a la manche suivante.
            round_counter += 1

            max_honba = max(max_honba, env.honba)

            # [FIX] Attribuer le reward de la manche a TOUTES les
            # transitions de chaque siege dans cette manche
            round_rewards = result['final_rewards']
            for s in range(4):
                reward_s = round_rewards.get(s, 0.0)
                reward_s = max(-20.0, min(20.0, reward_s))
                if reward_s == 0.0:
                    continue
                # [FIX] Boucle sur TOUS les indices du siege,
                # pas seulement le dernier
                for idx in round_indices_per_seat[s]:
                    transitions[idx]['reward'] += reward_s
                if not round_indices_per_seat[s]:
                    pending_reward[s] += reward_s

            # [FIX] Reset pour la prochaine manche
            round_indices_per_seat = {s: [] for s in range(4)}

            if result.get('game_done', False):
                # Reward pending (siege sans action dans la derniere manche)
                for s in range(4):
                    if pending_reward[s] != 0.0:
                        for t in reversed(transitions):
                            if t['seat'] == s:
                                t['reward'] += pending_reward[s]
                                break
                        pending_reward[s] = 0.0
                break
            req = env.reset_round()
        else:
            req = result['next_request']

    if transitions:
        transitions[-1]['done'] = True
        # [V9 CRITIC] Retour restant de partie (cible du critic)
        _finalize_episode_returns(transitions, rho=return_rho)

    scores = env.game_scores
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    # [ADVERSAIRES SL] rang du siege RL (cle conservee : 'rank_of_seat0')
    tracked = 0 if agent_player is None else agent_player
    rank_of_seat0 = order.index(tracked) + 1

    stats['rank_of_seat0'] = rank_of_seat0
    stats['n_honba'] = max_honba

    # [STATS] deltas env sur cette partie (compteurs cumules par process)
    env_stats_after = env.get_stats()

    def _delta(*keys):
        return sum(int(env_stats_after.get(k, 0)) - int(env_stats_before.get(k, 0))
                   for k in keys)

    stats['n_riichi'] = _delta('riichi')
    stats['n_pon'] = _delta('pon_calls')
    stats['n_chi'] = _delta('chi_calls')
    stats['n_kan'] = _delta('ankan', 'shouminkan', 'daiminkan')
    stats['n_ron_refused'] = _delta('ron_refused')
    stats['n_sl_errors'] = _delta('sl_errors')

    return transitions, stats