"""Tests numpy de la baseline critic (sans torch reel)."""
import sys, types
import numpy as np
import test_stubs  # noqa: F401
def _mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m
for n, kw in [('mahjong_ai.agents.oracle_agent', dict(OracleDiscardAgent=object)),
              ('mahjong_ai.training.parameter_server', dict(publish_to_state=None, fetch_from_state=None, ParameterServer=object, init_parameter_server=None)),
              ('mahjong_ai.training.self_play_worker', dict(worker_loop=None))]:
    _mod(n, **kw)
try:
    from mahjong_ai.training import train_rl_suphx as T
except Exception as e:                       # imports lourds absents
    print("import train_rl_suphx impossible (", e, ") -> test de la fonction pure seulement")
    T = None

ok = True
def check(c, m):
    global ok
    print(("  [OK  ] " if c else "  [FAIL] ") + m); ok &= bool(c)

if T is None:
    src = open('training/train_rl_suphx.py').read()
    a = src.index('def advantages_from_values'); b = src.index('def compute_advantages')
    ns = {'np': np}; exec(src[a:b], ns); f = ns['advantages_from_values']
else:
    f = T.advantages_from_values

rng = np.random.RandomState(0)
n = 5000
signal = rng.randn(n) * 5                       # part previsible par l'etat
R = signal + rng.randn(n) * 5                   # + bruit de la manche
# 1. critic parfait sur le signal -> EV ~ 0.5, variance de l'avantage divisee par ~2
adv, ev, used = f(R, signal)
check(used and 0.45 < ev < 0.55, f"EV={ev:.3f} ~ 0.5 (signal/variance totale)")
check(adv.var() < 0.6 * (R - R.mean()).var(), "variance de l'avantage reduite vs batch-mean")
# 2. baseline sans biais : E[A | etat] identique (R - V garde le meme gradient attendu)
check(abs(np.corrcoef(adv, noise := (R - signal))[0, 1] - 1) < 1e-6, "A = bruit de la manche (la part previsible est retiree)")
# 3. critic inutile / pire -> repli batch-mean
bad = rng.randn(n) * 20
adv2, ev2, used2 = f(R, bad)
check((not used2) and ev2 < 0 and abs(adv2.mean()) < 1e-3, f"critic mauvais (EV={ev2:.2f}) -> repli batch-mean")
# 4. pas de critic
adv3, ev3, used3 = f(R, None)
check((not used3) and np.allclose(adv3, R - R.mean()), "values=None -> batch-mean")
# 5. rewards constantes
adv4, ev4, used4 = f(np.zeros(10), np.zeros(10))
check(not used4 and np.allclose(adv4, 0), "rewards constantes -> 0, pas de division par 0")
print("OK" if ok else "ECHEC"); sys.exit(0 if ok else 1)
