"""Tests de la penalite de danger (worker, sans torch reel).

Stube torch + les modules lourds, importe le VRAI self_play_worker, puis
verifie sur un FakeEnv a 2 manches :
  1. la penalite porte sur la reward de la decision coupable uniquement
  2. dans G_t (_finalize_episode_returns), la penalite reste localisee
     a la decision coupable (le terme de correction l'annule pour les
     autres transitions de la manche)
  3. compteurs n_rl_discards / n_danger_discards
  4. sans cle 'danger' ni penalite : comportement inchange

Usage : py -3.14 test_danger_v10.py (depuis mahjong_ai/)
"""
import sys
import types

import numpy as np

# --- stubs torch + modules lourds (le worker n'utilise torch que dans
# worker_loop, pas dans _play_one_game / _finalize_episode_returns) ---
sys.modules.setdefault('torch', types.ModuleType('torch'))
for name, kw in [
    ('mahjong_ai.training.game_env', dict(GameEnv=object, PredictorHolder=object)),
    ('mahjong_ai.agents.oracle_agent', dict(OracleDiscardAgent=object)),
    ('mahjong_ai.training.parameter_server', dict(fetch_from_state=None,
                                                  publish_to_state=None)),
    ('mahjong_ai.agents.reward_predictor', dict(RewardPredictor=object,
                                                FEATURE_DIM=12)),
]:
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m

import test_stubs  # noqa: F401  (lib mahjong pour envs.multi_agent)
from mahjong_ai.training import self_play_worker as W

ok = True
def check(c, m):
    global ok
    print(("  [OK  ] " if c else "  [FAIL] ") + m)
    ok &= bool(c)


class FakeEnv:
    """2 manches, siege RL = 0. Round 1 : jusqu'a 3 defausses (deal-in si
    la tuile dangereuse est jetee). Round 2 : defausses sures, delta -1.
    Delta round 1 : +2."""
    def __init__(self, with_danger, dangerous_tile=1, deal_in_at=3):
        self.game_scores = [0.0] * 4
        self.honba = 0
        self.wd = with_danger
        self.dtile = dangerous_tile
        self.deal_at = deal_in_at
        self.deltas = [2.0, -1.0]
        self.rd = 0
        self.n = 0

    def get_stats(self):
        return {}

    def _req(self, seat):
        r = dict(seat=seat, action_type=W.DISCARD, obs=np.zeros(3),
                 oracle=np.zeros(2), mask=np.ones(34))
        if self.wd:
            d = np.zeros(34, dtype=np.float32)
            d[self.dtile] = 1.0
            r['danger'] = d
        return r

    def reset_round(self):
        self.n = 0
        self.rd += 1
        return self._req(0)

    def step(self, a):
        self.n += 1
        deal_in = (self.wd and a == self.dtile and self.n >= self.deal_at)
        if deal_in or self.n >= 3:
            fr = {s: (self.deltas[self.rd - 1] if s == 0 else 0.0)
                  for s in range(4)}
            self.n = 0
            return dict(round_done=True,
                        outcome=('ron' if deal_in else 'exhaustive'),
                        final_rewards=fr, game_done=self.rd >= 2)
        return dict(round_done=False, next_request=self._req(0))


class FakeAgent:
    def __init__(self, actions):
        self.actions = actions
        self.i = 0
    def select_action(self, o, oo, m, g):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a, 0.0, 0.0, oo


RHO = 0.85
PEN = 0.7

# --- 1. deal-in en FIN de manche 1 (cas reel : le ron termine la manche) ---
tr, st = W._play_one_game(
    FakeEnv(True), FakeAgent([2, 2, 1, 2, 2, 2]), 1.0,
    agent_player=0, danger_penalty=PEN)
check(len(tr) == 6, f"6 transitions (3+3), obtenu {len(tr)}")
check(abs(tr[2]['reward'] - (2.0 - PEN)) < 1e-9,
      f"reward decision coupable = 2 - 0.7 = {tr[2]['reward']:.3f}")
check(all(abs(tr[i]['reward'] - 2.0) < 1e-9 for i in (0, 1)),
      "rewards des autres decisions de la manche 1 : +2 (pas de penalite)")
check(all(abs(tr[i]['reward'] - (-1.0)) < 1e-9 for i in (3, 4, 5)),
      "rewards manche 2 : -1 (delta, pas de penalite)")
# G_t : S2 = -1 ; S1 = (2-0.7) + 0.85*(-1) = 0.45
g_guilty = (2.0 - PEN) + RHO * (-1.0)                     # 0.45
g_mates = g_guilty + (2.0 - (2.0 - PEN))                  # 1.15
check(abs(tr[2]['return'] - g_guilty) < 1e-9,
      f"G_t decision coupable = {tr[2]['return']:.4f} (attendu {g_guilty:.4f})")
check(all(abs(tr[i]['return'] - g_mates) < 1e-9 for i in (0, 1)),
      f"G_t decisions sures de la meme manche = {g_mates:.4f} "
      f"(penalite NON partagee)")
check(abs((tr[0]['return'] - tr[2]['return']) - PEN) < 1e-9,
      "ecart G_t coupable vs sures = exactement la penalite")
check(st['n_rl_discards'] == 6 and st['n_danger_discards'] == 1,
      f"compteurs dng : {st['n_danger_discards']}/{st['n_rl_discards']}")

# --- 2. deal-in en MILIEU de manche (robustesse du credit) ---
tr2, st2 = W._play_one_game(
    FakeEnv(True, deal_in_at=2), FakeAgent([2, 1, 2, 2, 2, 2]), 1.0,
    agent_player=0, danger_penalty=PEN)
# manche 1 : t0 (sur, +2), t1 (coupable, 2-0.7) ; manche 2 : 3 x -1
check(abs(tr2[1]['reward'] - (2.0 - PEN)) < 1e-9,
      "deal-in milieu de manche : penalite sur la decision coupable")
g_g = (2.0 - PEN) + RHO * (-1.0)
check(abs(tr2[0]['return'] - (g_g + PEN)) < 1e-9 and
      abs(tr2[1]['return'] - g_g) < 1e-9,
      "G_t : penalite localisee meme en milieu de manche")
check(st2['n_danger_discards'] == 1, "compteur dng = 1")

# --- 3. sans cle 'danger' : comportement inchange ---
tr3, st3 = W._play_one_game(
    FakeEnv(False), FakeAgent([2, 2, 1, 2, 2, 2]), 1.0,
    agent_player=0, danger_penalty=PEN)
check(all(abs(t['reward'] - (2.0 if i < 3 else -1.0)) < 1e-9
          for i, t in enumerate(tr3)),
      "sans danger : rewards = deltas purs")
check(st3['n_rl_discards'] == 0 and st3['n_danger_discards'] == 0,
      "sans danger : compteurs a 0")
g3 = 2.0 + RHO * (-1.0)
check(all(abs(tr3[i]['return'] - g3) < 1e-9 for i in range(3)),
      f"sans danger : G_t manche 1 uniforme = {g3:.4f}")

# --- 4. penalite 0 avec cle danger : compteurs on, penalties off ---
tr4, st4 = W._play_one_game(
    FakeEnv(True), FakeAgent([2, 2, 1, 2, 2, 2]), 1.0,
    agent_player=0, danger_penalty=0.0)
check(abs(tr4[2]['reward'] - 2.0) < 1e-9, "penalite 0 : reward inchangee")
check(st4['n_rl_discards'] == 6 and st4['n_danger_discards'] == 1,
      "penalite 0 : compteurs quand meme comptes (diagnostic)")

print("OK" if ok else "ECHEC")
sys.exit(0 if ok else 1)
