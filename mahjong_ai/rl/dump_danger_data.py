"""[PLAN DANGER - ÉTAPE 1] Dump du dataset supervisé pour la tête danger.

Contexte (fermeture RL-sur-pen) : D-2 event_pen terminé sans bouger le
dng% (pl plate de bout en bout, ~1500 événements pénalisants sur 162k
steps) et mr final 2.643 ± 0.107 contre baseline 2.484 ± 0.113 (éval
appariée 384 parties, seed 777) ; deux mécanismes d'échec indépendants
en deux runs (variance rl_defense2 ; rareté + fuite par tronc partagé
D-2). Pivot validé : prédiction de danger SUPERVISÉE et DENSE, sur les
labels contrefactuels `_danger_labels()` de envs/multi_agent.py —
exposés par le flag `danger_labels=True` du VecRunner : à CHAQUE
défausse du siège agent, un vecteur 34 « cette tuile donne-t-elle ron
à un adversaire tenpai non-furiten », calculé exactement depuis les
mains cachées. Là où le D-2 n'a vu que ~1500 événements bruités par la
chance adverse, ce dump produit ~80 décisions × 34 labels PAR PARTIE,
sans bruit d'échantillonnage (label contrefactuel exact, pas un tirage).

Le module ne change RIEN au champ de jeu : même runner, adversaires
3 × SL figé argmax, GRP optionnel, siège agent = policy à l'init SL
(= le SL lui-même), échantillonné par défaut pour la couverture des
états visités (--argmax pour l'argmax). Il consomme les transitions
déjà collectées par collect_batch(collect=True) et écrit UN shard .npz
par chunk (reprise/extension facile via --shard-offset) :

    obs_normal (N, 279, 34) float16  entrée EXACTE du tronc SL figé
                                     (comprend les 80 canaux lookahead
                                     si --compute-lookahead, défaut ON
                                     = conditions de l'éval de réf.)
    mask       (N, 34)        uint8  défausses légales
    danger     (N, 34)        uint8  labels contrefactuels 0/1
    action     (N,)           int16  tuile jouée
    seat       (N,)           uint8  siège de l'agent (0-3)
    round      (N,)           int16  index de manche dans la partie
    game       (N,)           int32  id GLOBAL de partie -> le split
                                    train/val de l'étape 2 se fait PAR
                                    PARTIE (jamais par transition)

Portes de sanity (cumulées à chaque chunk) :
  * plomberie : 0 transition avec label au premier chunk = flag
    danger_labels cassé -> SystemExit ;
  * cohérence : comptage du dump vs compteurs du runner
    (n_rl_discards / n_danger_discards) doivent coïncider ;
  * taux « joué » : % de tuiles JOUÉES dangereuses, à comparer au dng%
    connu du champ (~0.9-1.0 % en échantillonné) ;
  * taux « légal » : % de tuiles LÉGALES dangereuses = prior p de la
    BCE pondérée de l'étape 2 (pos_weight = (1-p)/p).

Usage (racine du repo ; ~5-6 Go disque et ~5-6 h pour 4000 parties
avec lookahead ON ; ~4x plus rapide avec --no-compute-lookahead) :
    py -3.14 -m mahjong_ai.rl.dump_danger_data ^
      --sl-discard checkpoints/sl_tonpusen_la/discard.pt ^
      --sl-frozen checkpoints/sl_tonpusen_la ^
      --grp checkpoints/grp_tonpusen.pt ^
      --out checkpoints/danger_data ^
      --games 4000 --chunk-size 32 --seed 777

Étape 2 (patch séparé) : tronc SL FIGÉ + tête danger 34 voies entraînée
sur ces shards (BCE pondérée, AUC/calibration, split par partie).
"""
import argparse
import os
import time

import numpy as np
import torch

from mahjong_ai.agents.suphx_cnn import N_CHANNELS
from mahjong_ai.envs.suphx_features import N_TILES
from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner

_KB_PER_TR = N_CHANNELS * N_TILES * 2 / 1024.0    # obs_normal float16


def stack_episodes(episodes, game_offset):
    """Empile les transitions de DÉFAUSSE du siège agent (les seules
    collectées par le runner : key 'rl' = requêtes DISCARD du siège
    agent, toutes porteuses d'un vecteur danger quand danger_labels est
    actif). Retour (dict d'arrays, n_rejetées) ; n_rejetées > 0
    signalerait une décision collectée sans label (plomberie)."""
    obs, masks, dangers, actions, seats, rounds_, games = \
        [], [], [], [], [], [], []
    n_rej = 0
    for gi, trs in enumerate(episodes):
        for t in trs:
            d = t.get('danger')
            if d is None:
                n_rej += 1
                continue
            obs.append(np.asarray(t['obs_normal'], dtype=np.float16))
            masks.append(np.asarray(t['mask'], dtype=np.uint8))
            dangers.append(np.asarray(d, dtype=np.uint8))
            actions.append(int(t['action']))
            seats.append(int(t['seat']))
            rounds_.append(int(t['round']))
            games.append(game_offset + gi)
    n = len(obs)
    return {
        'obs_normal': (np.stack(obs) if n else
                       np.zeros((0, N_CHANNELS, N_TILES), np.float16)),
        'mask': (np.stack(masks) if n else
                 np.zeros((0, N_TILES), np.uint8)),
        'danger': (np.stack(dangers) if n else
                   np.zeros((0, N_TILES), np.uint8)),
        'action': np.asarray(actions, dtype=np.int16),
        'seat': np.asarray(seats, dtype=np.uint8),
        'round': np.asarray(rounds_, dtype=np.int16),
        'game': np.asarray(games, dtype=np.int32),
    }, n_rej


def main():
    p = argparse.ArgumentParser(
        description='[PLAN DANGER - ÉTAPE 1] dump (features, labels '
                    'danger contrefactuels) pour la tête danger '
                    'supervisée (étape 2)')
    p.add_argument('--sl-discard', required=True,
                   help='discard.pt SL (le siège agent démarre à '
                        'l\'init SL = le SL lui-même)')
    p.add_argument('--sl-frozen', required=True,
                   help='dossier des têtes SL figées (riichi/chow/pong/'
                        'kong, adversaires)')
    p.add_argument('--grp', default=None,
                   help='checkpoint GRP du runner (même champ que le '
                        'training ; optionnel)')
    p.add_argument('--out', default='checkpoints/danger_data',
                   help='dossier de sortie (un shard .npz par chunk)')
    p.add_argument('--games', type=int, default=4000)
    p.add_argument('--chunk-size', type=int, default=32,
                   help='parties jouées en parallèle = un shard npz')
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--shard-offset', type=int, default=0,
                   help='premier index de shard (reprise/extension d\'un '
                        'dump existant SANS écraser)')
    p.add_argument('--argmax', action='store_true',
                   help='siège agent en argmax au lieu d\'échantillonné '
                        '(défaut : échantillonné, pour la couverture '
                        'des états visités)')
    p.add_argument('--compute-lookahead',
                   action=argparse.BooleanOptionalAction, default=True,
                   help='calcule les 80 canaux lookahead. DÉFAUT ON = '
                        'conditions de l\'éval de référence (2.484, '
                        '--compute-lookahead) : le dump doit matcher le '
                        'déploiement. --no-compute-lookahead = champ du '
                        'trainer D-2 (canaux à zéro, ~4x plus rapide).')
    p.add_argument('--compress', action='store_true',
                   help='np.savez_compressed au lieu de np.savez '
                        '(~2-3x plus petit, un peu plus lent)')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = PPOAgent(sl_discard_path=args.sl_discard,
                     sl_frozen_dir=args.sl_frozen, device=device)
    runner = VecRunner(agent, n_envs=args.chunk_size, seed=args.seed,
                       grp_path=args.grp,
                       compute_lookahead=args.compute_lookahead,
                       danger_labels=True, mode='opponent', device=device)

    os.makedirs(args.out, exist_ok=True)
    chunk = min(args.chunk_size, args.games)
    n_chunks = (args.games + chunk - 1) // chunk
    saver = np.savez_compressed if args.compress else np.savez

    tot = {'games': 0, 'n': 0, 'legal': 0, 'legal_dng': 0,
           'played_dng': 0, 'rej': 0}
    t0 = time.time()
    print(f"[dump] {args.games} parties par chunks de {chunk} | siège "
          f"agent {'argmax' if args.argmax else 'échantillonné'} | "
          f"lookahead={'ON' if args.compute_lookahead else 'OFF'} | "
          f"shards -> {args.out} (offset {args.shard_offset})", flush=True)
    print(f"[dump] disque estimé : "
          f"~{args.games * 80 * _KB_PER_TR / (1024 * 1024):.1f} Go non "
          f"compressé (~80 tr/partie, à recaler sur le réel)", flush=True)

    for c in range(n_chunks):
        n_here = min(chunk, args.games - c * chunk)
        episodes, all_stats = runner.collect_batch(
            n_here, gamma_t=0.0, collect=True,
            deterministic=args.argmax, gamma_override=0.0)
        arrays, n_rej = stack_episodes(episodes, tot['games'])
        saver(os.path.join(
            args.out,
            f'danger_shard_{args.shard_offset + c:05d}.npz'), **arrays)

        n = int(arrays['danger'].shape[0])
        tot['games'] += len(episodes)
        tot['n'] += n
        tot['rej'] += n_rej
        chunk_pd = 0
        if n:
            legal = arrays['mask'] > 0
            dng = arrays['danger'] > 0
            played = np.zeros((n, N_TILES), dtype=bool)
            played[np.arange(n), arrays['action']] = True
            chunk_pd = int((played & dng).sum())
            tot['legal'] += int(legal.sum())
            tot['legal_dng'] += int((dng & legal).sum())
            tot['played_dng'] += chunk_pd
        # cohérence dump vs compteurs du runner (même définition que
        # le dng% du trainer) : divergence = plomberie à investiguer
        rd = sum(int(s.get('n_rl_discards', 0)) for s in all_stats)
        dd = sum(int(s.get('n_danger_discards', 0)) for s in all_stats)
        if rd != n or dd != chunk_pd:
            print(f"[dump] ATTENTION : comptage dump (n={n}, dng="
                  f"{chunk_pd}) != runner (n={rd}, dng={dd})",
                  flush=True)
        if c == 0 and tot['n'] == 0:
            raise SystemExit(
                "[dump] ÉCHEC plomberie : 0 transition avec label danger "
                "au premier chunk -- danger_labels=True n'est pas arrivé "
                "jusqu'à l'env (vérifier env.round_env.compute_danger "
                "dans rl/rollout.py).")
        p_leg = tot['legal_dng'] / max(tot['legal'], 1)
        p_play = tot['played_dng'] / max(tot['n'], 1)
        done = min((c + 1) * chunk, args.games)
        eta = (time.time() - t0) / max(done, 1) * (args.games - done)
        print(f"[dump] {done}/{args.games} parties "
              f"({time.time() - t0:.0f}s, ETA {eta / 60:.0f} min) | "
              f"tr={tot['n']} (rejets {tot['rej']}) | légal dng "
              f"{100 * p_leg:.2f}% | joué dng {100 * p_play:.2f}% "
              f"(champ trainer ~0.9-1.0 en échantillonné)", flush=True)

    p_leg = tot['legal_dng'] / max(tot['legal'], 1)
    print()
    print(f"[dump] TERMINÉ : {tot['games']} parties, {tot['n']} "
          f"transitions de défausse, {n_chunks} shards dans {args.out}",
          flush=True)
    print(f"[dump] prior danger (tuiles légales) p={100 * p_leg:.3f}% -> "
          f"pos_weight BCE étape 2 = {(1 - p_leg) / max(p_leg, 1e-9):.1f}",
          flush=True)
    print("[dump] étape 2 (patch séparé) : tronc SL FIGÉ + tête danger "
          "34 voies, split train/val PAR PARTIE (colonne 'game').",
          flush=True)


if __name__ == '__main__':
    main()
