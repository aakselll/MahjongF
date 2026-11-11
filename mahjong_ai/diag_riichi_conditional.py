# Script diag_riichi_conditional.py
import torch, numpy as np
from collections import Counter
from mahjong_ai.sl.dataset import PolicyDataset, discover_files
from mahjong_ai.agents.suphx_cnn import build_model

files = discover_files('data/tenhou/2015', max_files=2900)
ds = PolicyDataset(files, 'riichi', cache_path='cache/sl_2015/samples_all.pkl',
                    verbose=False)
_, val_ds = ds.make_subsets(val_fraction=0.1)

ckpt = torch.load('checkpoints/sl_2015_run1/riichi.pt', map_location='cpu')
model = build_model('riichi').eval()
model.load_state_dict(ckpt['model_state_dict'])

# Filtre : ne garder que les états où riichi est légal (proxy = shanten == 0)
from mahjong.shanten import Shanten
_shanten = Shanten()

correct_legal, n_legal = 0, 0
correct_pos, n_pos = 0, 0    # parmi les vraies riichi
correct_neg, n_neg = 0, 0    # parmi les vrais no-riichi

with torch.no_grad():
    for i in val_ds.indices:
        obs_u8, label, _ = ds.samples[i]
        obs = obs_u8.astype(np.float32) / 255.0
        logits = model(torch.from_numpy(obs).unsqueeze(0))
        pred = int(logits.argmax(1).item())

        # Calcul shanten depuis les canaux own_hand (offset 0-3)
        own_hand = obs_u8[0:4]   # 4 canaux de main (uint8 0/255)
        counts = (own_hand > 127).sum(axis=0).astype(np.int32)
        if counts.sum() < 13 or counts.sum() > 14:
            continue
        sh = _shanten.calculate_shanten(counts.tolist(),
                                         use_chiitoitsu=True, use_kokushi=False)
        if sh != 0:
            continue
        n_legal += 1
        if pred == label:
            correct_legal += 1
        if label == 1:
            n_pos += 1
            if pred == 1: correct_pos += 1
        else:
            n_neg += 1
            if pred == 0: correct_neg += 1

print(f"Val_acc conditionnel (tenpai) : {correct_legal/max(n_legal,1):.4f}  "
      f"(n={n_legal})")
print(f"  Positifs (riichi) : {correct_pos}/{n_pos} = {correct_pos/max(n_pos,1):.4f}")
print(f"  Negatifs (passer) : {correct_neg}/{n_neg} = {correct_neg/max(n_neg,1):.4f}")