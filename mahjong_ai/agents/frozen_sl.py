"""Wrapper des 4 tetes SL figees (riichi/chow/pong/kong)."""
import os
import torch
from mahjong_ai.agents.suphx_cnn import build_model


FROZEN_HEADS = ['riichi', 'chow', 'pong', 'kong']


def load_frozen_sl(checkpoint_dir, device='cpu'):
    frozen = {}
    for head in FROZEN_HEADS:
        path = os.path.join(checkpoint_dir, f'{head}.pt')
        if not os.path.isfile(path):
            print(f"[frozen_sl] MANQUANT : {path}")
            continue
        m = build_model(head).to(device)
        ck = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        frozen[head] = m
        print(f"[frozen_sl] {head} charge (val_acc={ck.get('val_acc','?')})")
    return frozen