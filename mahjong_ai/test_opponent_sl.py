"""Test du mode --opponent-sl (stubs, sans torch reel necessaire)."""
import os, sys, types
import numpy as np
import test_stubs  # noqa: F401  (installe les stubs avant mahjong_ai)
def _mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m
_mod('mahjong_ai.training.game_env', GameEnv=object, PredictorHolder=object)
_mod('mahjong_ai.agents.oracle_agent', OracleDiscardAgent=object)
_mod('mahjong_ai.training.parameter_server', fetch_from_state=None)
_mod('mahjong_ai.agents.reward_predictor', RewardPredictor=object, FEATURE_DIM=12)
from mahjong_ai.training import self_play_worker as W

ok = True
def check(c, m):
    global ok
    print(("  [OK  ] " if c else "  [FAIL] ") + m); ok &= bool(c)

class FakeEnv:
    """4 sieges, 2 manches de 8 requetes DISCARD tournantes."""
    def __init__(self): self.game_scores=[0.,0.,0.,0.]; self.honba=0
    def get_stats(self): return {}
    def reset_round(self):
        self.n=0; self.rd=getattr(self,'rd',0)+1
        return self._req()
    def _req(self):
        return dict(seat=self.n%4, action_type=W.DISCARD, obs=np.zeros(3),
                    oracle=np.zeros(2), mask=np.ones(34))
    def step(self, a):
        self.n+=1
        if self.n>=8:
            fr={0:5.,1:-5.,2:0.,3:0.}
            for k,v in fr.items(): self.game_scores[k]+=v
            return dict(round_done=True, outcome='tsumo', final_rewards=fr,
                        game_done=self.rd>=2)
        return dict(round_done=False, next_request=self._req())

class FakeAgent:
    def __init__(self): self.rl=0; self.sl=0
    def select_action(self,o,oo,m,g): self.rl+=1; return 1,0.,0.,oo
    def select_action_sl(self,o,m,argmax=True): self.sl+=1; return 2

seats=[]
for ap in [None,0,1,2,3]:
    env=FakeEnv(); ag=FakeAgent()
    tr,st=W._play_one_game(env,ag,1.0,agent_player=ap)
    tseats={t['seat'] for t in tr}
    if ap is None:
        check(tseats=={0,1,2,3} and ag.sl==0, "self-play inchange : 4 sieges, 0 SL")
        check(st['rank_of_seat0']==1, "rang = siege 0 en self-play")
    else:
        check(tseats=={ap} and ag.sl==12 and ag.rl==4, f"agent={ap}: transitions seulement du siege RL ({ag.rl} RL / {ag.sl} SL)")
        exp={0:1,1:4}.get(ap,2 if ap==2 else 3)
        check(st['rank_of_seat0']==exp, f"rang suivi = siege RL ({ap}) -> {st['rank_of_seat0']}")
rng=np.random.RandomState(1000)
check(len({int(rng.randint(4)) for _ in range(50)})==4, "tirage du siege couvre les 4 sieges")
print("OK" if ok else "ECHEC"); sys.exit(0 if ok else 1)
