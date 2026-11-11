"""
export_rl_discard.py -- Exporte la policy RL au format "tete SL" attendu par
le sniffer / le bot (un dossier avec discard.pt, riichi.pt, chow.pt, pong.pt,
kong.pt, chacun {'model_state_dict', 'config', ...}).

Pourquoi : l'entrainement RL ecrit UN fichier {'policy','critic','step',
'alpha'} dont la policy a 254 canaux d'entree (250 normaux + 4 oracle). Le
sniffer cherche <dossier>/discard.pt au format SL (250 canaux, cle
'model_state_dict', cle 'config').

Equivalence EXACTE : a l'inference (gamma=0) les 4 canaux oracle valent 0,
leur contribution a la 1re convolution est donc nulle. Retirer ces 4 canaux
des poids d'entree ne change aucun logit.

Usage :
    py -3.14 -m mahjong_ai.training.export_rl_discard ^
        --rl-checkpoint checkpoints\\rl_tonpusen_v2\\rl_tonpusen_v2_latest.pt ^
        --sl-discard    <ancien discard.pt SL (pour la config d'archi)> ^
        --sl-frozen     <dossier des tetes SL riichi/chow/pong/kong> ^
        --out-dir       rl_final --check
"""
import argparse
import os
import shutil

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rl-checkpoint', required=True)
    p.add_argument('--sl-discard', required=True,
                   help="checkpoint SL discard : fournit 'config' (archi)")
    p.add_argument('--sl-frozen', required=True,
                   help='dossier des tetes SL a copier (riichi/chow/pong/kong)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--check', action='store_true',
                   help='verifie logits RL(oracle=0) == logits exportes')
    args = p.parse_args()

    from mahjong_ai.agents.suphx_cnn import (
        build_model, N_CHANNELS, IN_CHANNELS_ORACLE, adapt_in_channels)

    rl = torch.load(args.rl_checkpoint, map_location='cpu',
                    weights_only=False)
    if 'policy' not in rl:
        raise SystemExit(f"checkpoint RL inattendu, cles = {list(rl)}")
    sd = dict(rl['policy'])

    sl = torch.load(args.sl_discard, map_location='cpu', weights_only=False)
    cfg = dict(sl.get('config', {}))
    channels = cfg.get('channels', 256)
    n_blocks = cfg.get('n_blocks', 16)
    binary_hidden = cfg.get('binary_hidden', 256)

    key = 'trunk.input_conv.weight'
    w = sd[key]
    print(f"[export] {key} : {tuple(w.shape)}")
    if w.shape[1] != N_CHANNELS:
        # [ROUGES] retire les canaux oracle ; ajoute des poids nuls pour
        # les canaux rouges si le checkpoint RL est d'avant les rouges.
        sd[key] = adapt_in_channels(w, N_CHANNELS)
        if sd[key].shape[1] != N_CHANNELS:
            raise SystemExit(f"canaux d'entree inattendus : {w.shape[1]} "
                             f"(attendu {IN_CHANNELS_ORACLE} ou {N_CHANNELS})")
        print(f"[export] canaux adaptes -> {tuple(sd[key].shape)}")

    model = build_model('discard', channels=channels, n_blocks=n_blocks,
                        binary_hidden=binary_hidden, in_channels=N_CHANNELS)
    model.load_state_dict(sd, strict=True)      # echoue si archi differente
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'discard.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'head_type': 'discard',
        'config': {'channels': channels, 'n_blocks': n_blocks,
                   'binary_hidden': binary_hidden,
                   'in_channels': N_CHANNELS},
        'val_acc': None,
        'rl_step': rl.get('step'),
        'rl_alpha': rl.get('alpha'),
        'source': 'rl_export',
    }, out)
    print(f"[export] ecrit {out} (step RL={rl.get('step')})")

    for h in ('riichi', 'chow', 'pong', 'kong'):
        src = os.path.join(args.sl_frozen, f'{h}.pt')
        if not os.path.isfile(src):
            print(f"[export] !! tete manquante : {src}")
            continue
        shutil.copy2(src, os.path.join(args.out_dir, f'{h}.pt'))
        print(f"[export] copie {h}.pt")

    if args.check:
        rl_model = build_model('discard', use_oracle=True, channels=channels,
                               n_blocks=n_blocks, binary_hidden=binary_hidden)
        rl_model.load_state_dict(rl['policy'], strict=True)
        rl_model.eval()
        x = torch.rand(8, N_CHANNELS, 34)
        z = torch.zeros(8, IN_CHANNELS_ORACLE - N_CHANNELS, 34)
        with torch.no_grad():
            a = rl_model(torch.cat([x, z], dim=1))
            b = model(x)
        err = (a - b).abs().max().item()
        print(f"[check] ecart max des logits = {err:.2e}")
        assert err < 1e-4, "export NON equivalent"
        print("[check] OK : export equivalent a la policy RL (oracle=0)")


if __name__ == '__main__':
    main()
