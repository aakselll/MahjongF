# make_cal_head.py  —  py -3.14 make_cal_head.py
import torch, torch.nn.functional as F
from pathlib import Path

CKPT = Path(r"checkpoints\danger_head\danger_head.pt")
OUT  = Path(r"checkpoints\danger_head\danger_head_cal.pt")
DATA = Path(r"checkpoints\danger_data")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

# ---- ADAPTE : reprends EXACTEMENT le chargement de ton audit en mode --data
# X, y = ...        # 20000 etats, labels 0/1 par tuile legale (205832 au total)
# head = ...        # construction identique a l'audit
# with torch.no_grad():
#     logits = head(X).flatten()
# y = y.flatten()

# ---- Platt : p_cal = sigmoid(a * logit + b) ----
a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=200)
def closure():
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(a * logits + b, y)
    loss.backward()
    return loss
opt.step(closure)

with torch.no_grad():
    p_raw, p_cal = torch.sigmoid(logits), torch.sigmoid(a * logits + b)
print(f"Platt: a={a.item():.4f} b={b.item():.4f}")
print(f"mediane p_raw={p_raw.median():.4f}  p_cal={p_cal.median():.4f} (attendu ~0.008)")

def ece(p, n_bins=15):
    bins = torch.clamp((p * n_bins).long(), 0, n_bins - 1); e = 0.0
    for i in range(n_bins):
        m = bins == i
        if m.any(): e += m.float().mean().item() * abs(p[m].mean().item() - y[m].float().mean().item())
    return e
print(f"ECE brut={ece(p_raw):.4f}  calibre={ece(p_cal):.4f}")

ckpt["calib"] = {"type": "platt", "a": round(a.item(), 6), "b": round(b.item(), 6)}
torch.save(ckpt, OUT)
print(f"[ok] {OUT} ecrit avec calib={ckpt['calib']}")