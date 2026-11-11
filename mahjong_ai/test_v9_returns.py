"""Tests du patch v9 : retour restant de partie G_t (cible du critic).

Verifie :
  1. _finalize_episode_returns : suffixes escomptes par siege, corrections
     carried/pending, chainage independant par siege, rho=1 telescoping,
     rho=0 = delta de manche.
  2. compute_advantages : l'avantage suit 'return' (G_t), pas 'reward' ;
     repli sur 'reward' pour les episodes d'avant le patch.

Usage : py -3.14 test_v9_returns.py (depuis mahjong_ai/ ou son parent)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
from mahjong_ai.training.self_play_worker import _finalize_episode_returns

ok = True
def check(c, m):
    global ok
    print(("  [OK  ] " if c else "  [FAIL] ") + m)
    ok &= bool(c)


def T(seat, round_, reward):
    return {'seat': seat, 'round': round_, 'reward': float(reward)}


# --- 1. un siege, 3 manches, pending (+3) sur la derniere transition ---
tr = [T(0, 0, 1.0), T(0, 0, 1.0),      # manche 0 : R=1 (backfill uniforme)
      T(0, 1, 0.0),                     # manche 1 : R=0
      T(0, 2, 2.0), T(0, 2, 5.0)]       # manche 2 : R=2, +3 pending
_finalize_episode_returns(tr, rho=0.85)
check(abs(tr[0]['return'] - (1.0 + 0.85 * (0.0 + 0.85 * 5.0))) < 1e-9,
      f"G manche0 = {tr[0]['return']:.4f} (attendu 4.6125)")
check(abs(tr[2]['return'] - (0.0 + 0.85 * 5.0)) < 1e-9,
      f"G manche1 = {tr[2]['return']:.4f} (attendu 4.25)")
check(abs(tr[3]['return'] - 2.0) < 1e-9,
      f"G manche2 sans pending = {tr[3]['return']:.4f} (attendu 2.0)")
check(abs(tr[4]['return'] - 5.0) < 1e-9,
      f"G manche2 avec pending = {tr[4]['return']:.4f} (attendu 5.0)")

# --- 2. carried : reward reporte sur la 1re transition de la manche +1 ---
tr = [T(0, 0, 1.0),           # manche 0 : R=1
      T(0, 1, 2.7)]           # manche 1 : R=2 + carried 0.7
_finalize_episode_returns(tr, rho=0.85)
check(abs(tr[1]['return'] - 2.7) < 1e-9,
      f"carried restaure : G = {tr[1]['return']:.4f} (attendu 2.7)")
check(abs(tr[0]['return'] - (1.0 + 0.85 * 2.7)) < 1e-9,
      f"G amont = {tr[0]['return']:.4f} (attendu 3.295)")

# --- 3. deux sieges entrelaces : chainage par siege, pas entre sieges ---
tr = [T(0, 0, 1.0), T(1, 0, -1.0), T(0, 1, 2.0), T(1, 1, 0.0)]
_finalize_episode_returns(tr, rho=0.85)
check(abs(tr[0]['return'] - (1.0 + 0.85 * 2.0)) < 1e-9,
      f"siege0 manche0 = {tr[0]['return']:.4f} (attendu 2.7)")
check(abs(tr[1]['return'] - (-1.0 + 0.85 * 0.0)) < 1e-9,
      f"siege1 manche0 = {tr[1]['return']:.4f} (attendu -1.0)")
check(abs(tr[3]['return'] - 0.0) < 1e-9,
      f"siege1 manche1 = {tr[3]['return']:.4f} (attendu 0.0)")

# --- 4. rho=1 : telescopage -> G(1re decision) = somme des deltas ---
tr = [T(0, 0, 1.0), T(0, 1, 2.0), T(0, 2, 3.0)]
_finalize_episode_returns(tr, rho=1.0)
check(abs(tr[0]['return'] - 6.0) < 1e-9,
      f"rho=1 telescoping : {tr[0]['return']:.4f} (attendu 6.0)")

# --- 5. rho=0 : G = delta de manche (+ corrections) ---
tr = [T(0, 0, 1.0), T(0, 1, 2.0)]
_finalize_episode_returns(tr, rho=0.0)
check(abs(tr[0]['return'] - 1.0) < 1e-9 and abs(tr[1]['return'] - 2.0) < 1e-9,
      "rho=0 : G = reward de manche")

# --- 6. compute_advantages suit 'return', pas 'reward' (leurre) ---
from mahjong_ai.training import train_rl_suphx as TR

rng = np.random.RandomState(0)
G = rng.randn(40).astype(np.float32)
trans = [{'seat': 0, 'round': i // 10, 'reward': 100.0,   # leurre ignore
          'return': float(g)} for i, g in enumerate(G)]
advs = TR.compute_advantages(trans, adv_mode='batch_mean')
ref = (G - G.mean()) / (G.std() + 1e-8)
check(np.allclose(advs, ref, atol=1e-5),
      "A suit G_t ('return'), leurre reward=100 ignore")
check(all(abs(t['return'] - g) < 1e-6 for t, g in zip(trans, G)),
      "'return' preservé par compute_advantages")

# --- 7. repli : episodes sans 'return' (avant le patch) -> reward ---
old = [{'seat': 0, 'round': 0, 'reward': float(g)} for g in G]
advs2 = TR.compute_advantages(old, adv_mode='batch_mean')
check(np.allclose(advs2, ref, atol=1e-5),
      "repli compat : episodes sans 'return' -> reward")

print("OK" if ok else "ECHEC")
sys.exit(0 if ok else 1)
