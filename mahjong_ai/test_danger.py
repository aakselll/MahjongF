"""Tests des labels de danger (env, stubs mahjong, sans torch)."""
import sys
import numpy as np
import test_stubs  # noqa: F401
from mahjong_ai.envs.multi_agent import MultiAgentMahjongEnv

ok = True
def check(c, m):
    global ok
    print(("  [OK  ] " if c else "  [FAIL] ") + m); ok &= bool(c)

# main tenpai attente tanki 2z (28) : 123m 456p 789s EEE + S (tanki)
tenpai = [0, 1, 2, 12, 13, 14, 24, 25, 26, 27, 27, 27, 28]
junk = [0, 3, 6, 10, 13, 16, 19, 22, 25, 29, 30, 31, 32]   # loin du tenpai


def build(hands, river=None, flag=True):
    env = MultiAgentMahjongEnv(seed=4, compute_lookahead=False)
    env.compute_danger = flag
    env.reset_round()
    for s in range(4):
        h = np.zeros(34, dtype=np.int32)
        for t in hands[s]:
            h[t] += 1
        env.hands[s] = h
        env.melds[s] = []
    env._clear_reds()
    if river:
        for s, t in river:
            env.discards[s][t] += 1
    return env


env = build([junk + [33], tenpai, junk, junk])          # siege 0 doit defausser
d = env._danger_labels(0)
check(d[28] == 1.0 and d.sum() == 1.0,
      "tenpai tanki 2z : danger uniquement sur la tuile 28")
# penchan 1-2m : attente 3m (tuile 2)
ry = [0, 1, 12, 13, 14, 24, 25, 26, 27, 27, 27, 29, 29]
env2 = build([junk + [33], ry, junk, junk])
d2 = env2._danger_labels(0)
check(d2[2] == 1.0, "penchan 1-2m : attente 3m (tuile 2) marquee dangereuse")
# furiten : l'adversaire a deja defausse 28 -> pas de danger
env3 = build([junk + [33], tenpai, junk, junk], river=[(1, 28)])
check(env3._danger_labels(0).sum() == 0.0,
      "adversaire furiten : aucun danger")
# pas tenpai
env4 = build([junk + [33], junk, junk, junk])
check(env4._danger_labels(0).sum() == 0.0,
      "aucun adversaire tenpai : aucun danger")
# l'acteur lui-meme est exclu
env5 = build([tenpai + [33], junk, junk, junk])
check(env5._danger_labels(0).sum() == 0.0,
      "la main du siege acteur n'est pas comptee")
# flag : cle presente dans la requete seulement si actif
e1 = MultiAgentMahjongEnv(seed=2, compute_lookahead=False)
e1.compute_danger = True
r1 = e1.reset_round()
e0 = MultiAgentMahjongEnv(seed=2, compute_lookahead=False)
r0 = e0.reset_round()
check('danger' in r1 and r1['danger'].shape == (34,),
      "flag actif : requete DISCARD avec 'danger' (34,)")
check('danger' not in r0,
      "flag inactif : pas de cle 'danger' (comportement inchange)")
# parties aleatoires : jamais d'exception, valeurs binaires
n = 0
for sd in range(30):
    e = MultiAgentMahjongEnv(seed=100 + sd, compute_lookahead=False)
    e.compute_danger = True
    req = e.reset_round()
    steps = 0
    while steps < 400:
        if req['action_type'] == 0:
            dd = req['danger']
            assert set(np.unique(dd)) <= {0.0, 1.0}
            n += int(dd.sum() > 0)
        a = int(np.flatnonzero(req['mask'])[0]) \
            if req['action_type'] == 0 else 0
        res = e.step(a)
        steps += 1
        if res['round_done']:
            break
        req = res['next_request']
check(True, f"30 manches aleatoires sans erreur (requetes avec danger>0 : {n})")
print("OK" if ok else "ECHEC"); sys.exit(0 if ok else 1)
