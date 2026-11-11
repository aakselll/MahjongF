"""
Diagnostic : matrice de confusion + recall/precision/F1 par classe.

Fonctionne pour les 5 tetes Suphx SL :
    discard (34 classes) / riichi (2) / chow (4) / pong (2) / kong (2)

Usage :
    py -3.14 -m mahjong_ai.diag_confusion --head chow
    py -3.14 -m mahjong_ai.diag_confusion --head riichi
    py -3.14 -m mahjong_ai.diag_confusion --head discard

Options :
    --head            tete a diagnostiquer (obligatoire)
    --checkpoint-dir  dossier des .pt (defaut checkpoints/sl_2015_run1)
    --cache-dir       dossier du cache (defaut cache/sl_2015)
    --data-dir        dossier des .zst (defaut data/tenhou/2015)
    --max-files       nombre de fichiers (defaut 2900)
    --batch-size      batch size (defaut 256)
    --device          cpu / cuda (defaut auto)
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from mahjong_ai.sl.dataset import PolicyDataset, discover_files
from mahjong_ai.agents.suphx_cnn import build_model, N_CHANNELS
from mahjong_ai.envs.multi_agent import DISCARD, PONG, CHOW, KONG, RIICHI


HEAD_TO_N_CLASSES = {
    'discard': 34,
    'riichi': 2,
    'chow': 4,
    'pong': 2,
    'kong': 2,
}

HEAD_TO_LABELS = {
    'discard': [f"{i}" for i in range(34)],  # sera remplace par tuile_str
    'riichi': ['no', 'riichi'],
    'chow': ['PASS', 'LOWER', 'MID', 'UPPER'],
    'pong': ['no', 'pong'],
    'kong': ['no', 'kong'],
}


def tile_str(idx):
    if idx < 0 or idx >= 34:
        return "?"
    if idx < 9:   return f"{idx+1}m"
    if idx < 18:  return f"{idx-9+1}p"
    if idx < 27:  return f"{idx-18+1}s"
    return ["E", "S", "W", "N", "P", "F", "C"][idx - 27]


def get_labels(head):
    if head == 'discard':
        return [tile_str(i) for i in range(34)]
    return HEAD_TO_LABELS[head]


@torch.no_grad()
def run_diagnostic(model, loader, n_classes, device):
    """Retourne confusion matrix (n_classes, n_classes) + total."""
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    total_loss, total_n = 0.0, 0
    for chs, labels in loader:
        chs = chs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(chs)
        preds = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        for lbl, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            conf[lbl, pred] += 1
    return conf, total_loss / max(total_n, 1)


def print_confusion(conf, labels, max_display=34):
    """Affiche la matrice. Si trop de classes, on compacte l'affichage."""
    n = conf.shape[0]
    if n > 12:
        # Affichage compact : on ne montre que les classes non nulles
        active = [i for i in range(n) if conf[i].sum() > 0]
        print(f"\n=== Matrice de confusion (classes actives uniquement) ===")
        print(f"Total classes : {n}, actives : {len(active)}")
        # Pour discard : on affiche les stats par classe au lieu de la matrice
        return
    # Affichage complet pour <= 12 classes
    col_width = max(8, max(len(l) for l in labels) + 2)
    header = " " * (col_width + 2) + " ".join(
        f"{l:>{col_width}}" for l in labels)
    print(f"\n=== Matrice de confusion (lignes=reel, cols=pred) ===")
    print(header)
    for i, lbl in enumerate(labels):
        row = " ".join(f"{conf[i, j]:>{col_width}}" for j in range(n))
        print(f"  {lbl:>{col_width}}  {row}")


def print_per_class(conf, labels):
    n = conf.shape[0]
    print(f"\n=== Recall / Precision / F1 par classe ===")
    print(f"  {'classe':>10s}  {'support':>8s}  {'recall':>8s}  "
          f"{'prec':>8s}  {'f1':>8s}")
    f1s = []
    supports = []
    for i in range(n):
        support = conf[i].sum()
        tp = conf[i, i]
        if support == 0:
            f1s.append(0.0)
            supports.append(0)
            continue
        recall = tp / max(conf[i].sum(), 1)
        precision = tp / max(conf[:, i].sum(), 1)
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0
        f1s.append(f1)
        supports.append(support)
        print(f"  {labels[i]:>10s}  {support:>8d}  {recall:>8.4f}  "
              f"{precision:>8.4f}  {f1:>8.4f}")
    # F1 macro : moyenne non ponderee par classe (plus utile que micro)
    f1_macro = np.mean(f1s)
    print(f"\n  F1 macro = {f1_macro:.4f}")

    # F1 micro (= accuracy globale)
    total = conf.sum()
    acc = np.trace(conf) / max(total, 1)
    print(f"  Accuracy globale = {acc:.4f}")
    return f1_macro, acc


def print_conditional(conf, labels, head):
    """Statistiques conditionnelles : sur les etats ou l'action est LEGALE.

    Pour head == 'riichi', on ne peut pas le calculer sans la matrice
    complete du cache (shanten), donc on affiche uniquement les stats
    brutes.
    """
    n = conf.shape[0]
    if n == 2:
        # Binaire : rappel de la classe positive + precision
        support_pos = conf[1].sum()
        support_neg = conf[0].sum()
        tp = conf[1, 1]
        fp = conf[0, 1]
        fn = conf[1, 0]
        tn = conf[0, 0]
        if support_pos > 0:
            print(f"\n  Class 1 (positive) : n={support_pos}")
            print(f"    TP={tp}  FN={fn}  -> recall   = {tp/support_pos:.4f}")
            print(f"    FP={fp}  TN={tn}")
            if tp + fp > 0:
                print(f"                  precision = {tp/(tp+fp):.4f}")
        if support_neg > 0:
            print(f"  Class 0 (negative) : n={support_neg}")
            print(f"    recall = {tn/support_neg:.4f}")
    elif n == 4:
        # Chow : 4 classes
        for i in range(4):
            support = conf[i].sum()
            if support == 0:
                continue
            tp = conf[i, i]
            print(f"  {labels[i]:>6s} : n={support:6d}  "
                  f"recall={tp/support:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--head', required=True,
                   choices=['discard', 'riichi', 'chow', 'pong', 'kong'])
    p.add_argument('--checkpoint-dir', default='checkpoints/sl_2015_run1')
    p.add_argument('--cache-dir', default='cache/sl_2015')
    p.add_argument('--data-dir', default='data/tenhou/2015')
    p.add_argument('--max-files', type=int, default=2900)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[diag/{args.head}] device={device}", flush=True)

    # --- Dataset (charge le cache partage) ---
    files = discover_files(args.data_dir, max_files=args.max_files)
    cache = os.path.join(args.cache_dir, 'samples_all.pkl')
    ds = PolicyDataset(files, args.head, cache_path=cache, verbose=True,
                        parse_workers=1)

    _, val_ds = ds.make_subsets(val_fraction=args.val_fraction)
    print(f"[diag/{args.head}] val={len(val_ds)}", flush=True)

    from torch.utils.data import DataLoader

    def collate(batch):
        chs = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.stack([b[1] for b in batch], dim=0)
        return chs, labels

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate)

    # --- Chargement du modele ---
    ckpt_path = os.path.join(args.checkpoint_dir, f'{args.head}.pt')
    print(f"[diag/{args.head}] chargement {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = build_model(
        args.head,
        channels=cfg.get('channels', 256),
        n_blocks=cfg.get('n_blocks', 16),
        binary_hidden=cfg.get('binary_hidden', 256),
        in_channels=N_CHANNELS,   # [ROUGES] ancien ckpt adapte au chargement
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[diag/{args.head}] val_acc annonce = {ckpt.get('val_acc', '?')}",
          flush=True)

    # --- Diagnostic ---
    n_classes = HEAD_TO_N_CLASSES[args.head]
    conf, loss = run_diagnostic(model, val_loader, n_classes, device)
    labels = get_labels(args.head)

    print_confusion(conf, labels)
    f1_macro, acc = print_per_class(conf, labels)
    print_conditional(conf, labels, args.head)

    # --- Resume ---
    print(f"\n=== RESUME {args.head} ===")
    print(f"  val_loss  = {loss:.4f}")
    print(f"  val_acc   = {acc:.4f}")
    print(f"  F1 macro  = {f1_macro:.4f}")

    # --- Verdict automatique ---
    print(f"\n=== VERDICT ===")
    if args.head == 'discard':
        if acc > 0.6:
            print(f"  ✅ Bon (>{0.6:.2f})")
        else:
            print(f"  ⚠️  Faible (<{0.6:.2f})")
    else:
        if f1_macro > 0.60:
            print(f"  ✅ Excellent (F1 macro > 0.60)")
        elif f1_macro > 0.45:
            print(f"  ✅ Bon compromis (F1 macro > 0.45)")
        elif f1_macro > 0.30:
            print(f"  🟡 Biais persistant (F1 macro 0.30-0.45)")
        else:
            print(f"  🔴 Modele inutilisable (F1 macro < 0.30)")


if __name__ == '__main__':
    main()