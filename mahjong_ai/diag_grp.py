# diag_grp_v2.py
import numpy as np
import torch
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM

data = np.load('grp_dataset_cache.npy', allow_pickle=True).item()
samples = data['samples']
print(f"{len(samples)} sequences")

model = RewardPredictor(input_dim=FEATURE_DIM, hidden_dim=128, num_layers=2)
model.load_state_dict(torch.load('checkpoints/grp_sl_v4.pt', map_location='cpu'))
model.eval()

preds_final = []
true_final = []
with torch.no_grad():
    for seq, rew in samples:
        if seq.shape[0] < 2:
            continue
        x = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0)
        phi = model(x)[0].numpy()
        preds_final.append(phi[-1])
        true_final.append(rew)

preds_final = np.array(preds_final)
true_final = np.array(true_final)

print(f"\n=== Correlation (mais sur le train, gonflee) ===")
print(f"Pearson r = {np.corrcoef(preds_final, true_final)[0,1]:.4f}")

print(f"\n=== Calibration par classe ===")
for target in [-15, -5, 5, 15]:
    mask = (true_final == target)
    if mask.sum() == 0: continue
    print(f"  reward={target:+3d}  n={mask.sum():5d}  "
          f"Phi moy={preds_final[mask].mean():+.2f}  "
          f"erreur={preds_final[mask].mean() - target:+.2f}  "
          f"std={preds_final[mask].std():.2f}")

# Comparaison avec la distribution reelle
print(f"\n=== Distribution ===")
print(f"  Phi  : min={preds_final.min():+.2f}  max={preds_final.max():+.2f}  "
      f"mean={preds_final.mean():+.2f}  std={preds_final.std():.2f}")
print(f"  vrai : min={true_final.min():+.2f}  max={true_final.max():+.2f}  "
      f"mean={true_final.mean():+.2f}  std={true_final.std():.2f}")