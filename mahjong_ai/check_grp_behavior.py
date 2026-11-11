# check_grp_deltas.py
import torch
import numpy as np
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM

grp = RewardPredictor(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2)
ck = torch.load("checkpoints/grp_tonpusen.pt", map_location="cpu", weights_only=False)
grp.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
grp.eval()

def phi_prefix(seq):
    """Phi au DERNIER pas de la sequence (prediction du rang final depuis cet etat)."""
    x = torch.as_tensor(np.array(seq, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        return grp(x)[0, -1].item()

def feat(delta=0, score=20000, rank=4, rr=3, gap_leader=5000, gap_last=0):
    return [delta/1000, score/1000, 0.0, rr/8.0, 0.0, rank/4.0,
            gap_leader/1000, gap_last/1000, 1.0 if rr == 0 else 0.0,
            0.0, 0.0, 0.0]

print("=== Test 2' : saut DeltaPhi d'un +8000 (rank 4 -> 2) ===")
# Gain pendant E2 : saut entre post-E1 et post-E2
avant_tard = feat(rr=3)                       # 20k, rank4
apres      = dict(score=28000, rank=2, gap_leader=2000)
seq_t = [feat(rr=3), feat(delta=8000, rr=2, **apres)]
saut_tot  = phi_prefix(seq_t) - phi_prefix([feat(rr=3)])
# Gain pendant E4 : saut entre post-E3 et post-E4
seq_l = [feat(rr=3), feat(rr=2), feat(rr=1), feat(delta=8000, rr=0, **apres)]
saut_tard = phi_prefix(seq_l) - phi_prefix([feat(rr=3), feat(rr=2), feat(rr=1)])
print(f"DeltaPhi(+8000 en E2) : {saut_tot:+.2f}")
print(f"DeltaPhi(+8000 en E4) : {saut_tard:+.2f}")
print(f"OK : {saut_tard > saut_tot}  (attendu True : gain certain > gain incertain)")

print("\n=== Test 3' : saut DeltaPhi d'un deal-in -12000 (leader -> rank 3) ===")
lead32 = dict(score=32000, rank=1, gap_leader=0, gap_last=7000)
apres3 = dict(score=20000, rank=3, gap_leader=10000)
# Deal-in pendant E2
seq_de = [feat(**lead32, rr=3), feat(delta=-12000, rr=2, **apres3)]
saut_de = phi_prefix(seq_de) - phi_prefix([feat(**lead32, rr=3)])
# Deal-in pendant E4
seq_dl = [feat(**lead32, rr=3), feat(**lead32, rr=2),
          feat(**lead32, rr=1), feat(delta=-12000, rr=0, **apres3)]
saut_dl = phi_prefix(seq_dl) - phi_prefix([feat(**lead32, rr=3), feat(**lead32, rr=2), feat(**lead32, rr=1)])
print(f"DeltaPhi(-12000 en E2) : {saut_de:+.2f}")
print(f"DeltaPhi(-12000 en E4) : {saut_dl:+.2f}")
print(f"OK : {saut_dl < saut_de}  (attendu True : deal-in d'orasu coute plus cher)")