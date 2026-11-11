# test_pmcpa.py
from mahjong_ai.agents.pmcpa import PMCPA
from mahjong_ai.agents.suphx_cnn import build_model
import numpy as np
import torch

# Charge les modeles
models = {}
for h in ['discard', 'riichi', 'chow', 'pong', 'kong']:
    m = build_model(h)
    ck = torch.load(f'checkpoints/sl_2015_run1/{h}.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    models[h] = m

# Instancie pMCPA avec K=3 pour le test
pmcpa = PMCPA(models, {}, device='cpu', k=3, lr=1e-4, grad_steps=1,
              timeout_s=60.0, verbose=True)

# Main simple
our_hand = np.zeros(34, dtype=np.int32)
for t in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    our_hand[t] = 1

# Simule une fausse "manche" (juste pour tester)
class FakeState:
    hand = list(range(13))
    my_seat = 0
    melds = []
    dora_indicators = [0]

pmcpa.reset()
pmcpa._simulate_one_round(our_hand, FakeState(), np.random.RandomState(0), None)
# Doit imprimer les infos de la sim