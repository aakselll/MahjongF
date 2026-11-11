"""
Entraine UN modele Suphx SL (discard / riichi / chow / pong / kong).
Utilise un cache PARTAGE entre les 5 tetes : le premier run parse tout
et ecrit /_shards/ ; les runs suivants chargent ce cache en quelques
secondes et filtrent en memoire.

[MODIF 2026-10-14] Ajout de --max-samples : cap memoire sur le nombre
de samples charges en RAM (protection 16GB). Default auto = 500k.

[FIX 2025-XX-8] Weighted loss UNIQUEMENT pour chow.
- chow : desequilibre extreme (96% PASS / 4% chow) -> weighted loss
  CRITIQUE, sinon le modele collapse vers PASS.
- riichi/pong/kong : desequilibre modere (64-74% majoritaire) ->
  weighted loss DEGRADE le val_acc (le modele sur-predit la classe
  rare et perd en accuracy globale). Loss standard.
- discard : distribution ~uniforme sur 34 classes -> weighted loss
  sans effet (poids ~1 partout). Loss standard.

[FIX 2025-XX-7] Ajout --warm-start : charge un checkpoint existant (meme
tete) avant d'entrainer. Le LR doit etre reduit (ex: 1e-4) pour eviter
l'oscillation.

Usage :
python -m mahjong_ai.sl.train_policy_sl ^
    --head discard --data-dir data/tenhou_tonpusen ^
    --cache-dir cache/sl_tonpusen ^
    --output checkpoints/sl_tonpusen_light/discard.pt ^
    --max-files 2900 --epochs 2 --batch-size 128 --num-workers 0 ^
    --parse-workers 5 --max-samples 400000
"""
import argparse
import os
import time
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from mahjong_ai.sl.dataset import (
    PolicyDataset, discover_files, DEFAULT_MAX_SAMPLES
)
from mahjong_ai.agents.suphx_cnn import build_model, N_CHANNELS

# Tete(s) pour laquelle/lesquelles on utilise la weighted loss.
# Ajouter ici si tu constates un collapse sur une autre tete.
WEIGHTED_LOSS_HEADS = {'chow', 'riichi'}


def make_collate():
    def collate(batch):
        chs = torch.stack([b[0] for b in batch], dim=0)      # (B, N_CHANNELS, 34)
        labels = torch.stack([b[1] for b in batch], dim=0)   # (B,)
        return chs, labels
    return collate


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    """Evalue le modele. La loss est ponderee de la meme facon qu'en
    training pour que val_loss soit comparable a train_loss."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for chs, labels in loader:
        chs = chs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(chs)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    model.train()
    return total_loss / max(n, 1), correct / max(n, 1)


def compute_class_weights(train_ds, ds, n_classes, device, head_type):
    """Ponderation par l'inverse de la frequence, UNIQUEMENT pour les tetes
    listees dans WEIGHTED_LOSS_HEADS. Retourne None sinon (loss standard).

    Formule : weight[c] = n_total / (n_classes * count[c])
    - Classes frequentes -> poids < 1 (leur erreur compte moins)
    - Classes rares    -> poids > 1 (leur erreur compte plus)
    """
    if head_type not in WEIGHTED_LOSS_HEADS:
        return None
    train_labels = [ds.samples[i][1] for i in train_ds.indices]
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights_list = []
    for c in range(n_classes):
        cnt = max(counts.get(c, 1), 1)   # evite div par 0 pour classes absentes
        #weights_list.append(total / (n_classes * cnt))
        weights_list.append((total / (n_classes * cnt)) ** 0.5)
    return torch.tensor(weights_list, dtype=torch.float32, device=device)


def _format_indices_from_metadata(dataset):
    """Retourne les indices des echantillons tonpuu et hanchan.
    Utilise la metadata (dataset.sample_formats), pas les canaux.
    """
    tonpuu, hanchan = [], []
    for index in range(len(dataset)):
        fmt = dataset.sample_formats[index]
        if fmt == "tonpuu":
            tonpuu.append(index)
        else:
            hanchan.append(index)
    return tonpuu, hanchan


def _balanced_sampler(dataset, indices):
    """Echantillonne a parts egales tonpuu/hanchan via metadata."""
    counts = {"tonpuu": 0, "hanchan": 0}
    labels = []
    for index in indices:
        fmt = dataset.sample_formats[index]
        labels.append(fmt)
        counts[fmt] += 1
    weights = [1.0 / max(counts[fmt], 1) for fmt in labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True,
    )


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[train/{args.head}] device={device}', flush=True)

    # ---------------- Donnees ----------------
    files = discover_files(args.data_dir, max_files=args.max_files)
    cache = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache = os.path.join(args.cache_dir, 'samples_all.pkl')

    # Resolution du cap : None -> default auto, 0 -> pas de cap
    if args.max_samples is None:
        max_samples = DEFAULT_MAX_SAMPLES
    elif args.max_samples <= 0:
        max_samples = None
    else:
        max_samples = args.max_samples

    ds = PolicyDataset(
        files, args.head, cache_path=cache, verbose=True,
        parse_workers=args.parse_workers,
        max_samples=max_samples,
    )

    if len(ds) == 0:
        raise RuntimeError(f'Aucun echantillon pour {args.head}')

    train_ds, val_ds = ds.make_subsets(val_fraction=args.val_fraction)
    print(f'[train/{args.head}] train={len(train_ds)}  val={len(val_ds)}',
          flush=True)

    # Statistiques par format via metadata
    train_tonpuu, train_hanchan = _format_indices_from_metadata(ds)
    print(f'[train/{args.head}] format total: '
          f'tonpuu={len(train_tonpuu)} hanchan={len(train_hanchan)}',
          flush=True)

    # ---------------- Nombre de classes ----------------
    if args.head == 'discard':
        n_classes = 34
    elif args.head == 'chow':
        n_classes = 4          # PASS / LOWER / MID / UPPER
    else:
        n_classes = 2          # riichi / pong / kong

    # ---------------- Poids de classe ----------------
    class_weights = compute_class_weights(
        train_ds, ds, n_classes, device, head_type=args.head)
    if class_weights is not None:
        weight_display = [round(w, 3) for w in class_weights.tolist()]
        print(f'[train/{args.head}] class weights = {weight_display} '
              f'(weighted loss)', flush=True)
    else:
        print(f'[train/{args.head}] class weights = None (loss standard)',
              flush=True)

    # Distribution pour info
    train_labels = [ds.samples[i][1] for i in train_ds.indices]
    label_counts = Counter(train_labels)
    dist = {c: round(label_counts.get(c, 0) / max(len(train_labels), 1), 4)
            for c in range(n_classes)}
    print(f'[train/{args.head}] train distribution = {dist}', flush=True)

    # ---------------- DataLoaders ----------------
    pin = (device == 'cuda')

    # Sampler equilibre si les deux formats sont presents
    sampler = None
    shuffle = True
    n_ton = len(train_tonpuu)
    n_han = len(train_hanchan)
    if n_ton > 0 and n_han > 0:
        sampler = _balanced_sampler(ds, train_ds.indices)
        shuffle = False
        print(f'[train/{args.head}] sampler format equilibre 50/50 '
              f'(tonpuu={n_ton}, hanchan={n_han})', flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers, collate_fn=make_collate(),
        pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=make_collate(),
        pin_memory=pin,
    )

    # Validateurs par format via metadata
    val_tonpuu = [i for i in val_ds.indices
                  if ds.sample_formats[i] == 'tonpuu']
    val_hanchan = [i for i in val_ds.indices
                   if ds.sample_formats[i] == 'hanchan']
    val_format_loaders = {}
    for name, indices in (('tonpuu', val_tonpuu), ('hanchan', val_hanchan)):
        if indices:
            val_format_loaders[name] = DataLoader(
                Subset(ds, indices), batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=make_collate(), pin_memory=pin,
            )

    # ---------------- Modele ----------------
    model = build_model(
        args.head,
        channels=args.channels,
        n_blocks=args.n_blocks,
        binary_hidden=args.binary_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[train/{args.head}] params : {n_params/1e6:.3f}M  '
          f'(channels={args.channels}, blocks={args.n_blocks}, '
          f'in_ch={N_CHANNELS})', flush=True)

    # ---------------- Warm-start (optionnel) ----------------
    if args.warm_start:
        if not os.path.isfile(args.warm_start):
            print(f'[train/{args.head}] warm-start ignore : '
                  f'{args.warm_start} introuvable', flush=True)
        else:
            print(f'[train/{args.head}] warm-start depuis {args.warm_start}',
                  flush=True)
            try:
                ckpt = torch.load(args.warm_start, map_location=device)
                ckpt_head = ckpt.get('head_type', '?')
                if ckpt_head != args.head:
                    print(f'[train/{args.head}] ATTENTION : le checkpoint est '
                          f'pour {ckpt_head}, warm-start SKIP', flush=True)
                else:
                    missing, unexpected = model.load_state_dict(
                        ckpt['model_state_dict'], strict=False)
                    if missing:
                        print(f'[train/{args.head}] cles manquantes : '
                              f'{missing}', flush=True)
                    if unexpected:
                        print(f'[train/{args.head}] cles inattendues : '
                              f'{unexpected}', flush=True)
                    print(f'[train/{args.head}] poids charges '
                          f'(val_acc precedent = {ckpt.get("val_acc", "?")}, '
                          f'epoch precedent = {ckpt.get("epoch", "?")})',
                          flush=True)
            except Exception as e:
                print(f'[train/{args.head}] ECHEC warm-start : '
                      f'{type(e).__name__}: {e}', flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(args.epochs * len(train_loader), 1))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, correct, n = 0.0, 0, 0

        for step, (chs, labels) in enumerate(train_loader):
            chs = chs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(chs)

            # Loss ponderee SEULEMENT si class_weights n'est pas None
            loss = F.cross_entropy(logits, labels, weight=class_weights)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            scheduler.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            n += labels.size(0)

            if (step + 1) % 100 == 0:
                print(f'  step {step+1:5d}/{len(train_loader)} '
                      f'loss={loss.item():.4f}', flush=True)

        tr_loss = total_loss / max(n, 1)
        tr_acc = correct / max(n, 1)
        va_loss, va_acc = evaluate(model, val_loader, device,
                                   class_weights=class_weights)

        # Metriques par format
        format_metrics = []
        for format_name, format_loader in val_format_loaders.items():
            _, format_acc = evaluate(
                model, format_loader, device, class_weights=class_weights)
            format_metrics.append(f'{format_name}_acc={format_acc:.4f}')

        dt = time.time() - t0
        print(f'[epoch {epoch+1:2d}/{args.epochs}] '
              f'train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  '
              f'val_loss={va_loss:.4f} val_acc={va_acc:.4f} '
              f'{" ".join(format_metrics)} ({dt:.1f}s)',
              flush=True)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'head_type': args.head,
                'config': {
                    'channels': args.channels,
                    'n_blocks': args.n_blocks,
                    'binary_hidden': args.binary_hidden,
                    'in_channels': N_CHANNELS,
                },
                'val_acc': va_acc,
                'epoch': epoch + 1,
                'class_weights': (class_weights.tolist()
                                  if class_weights is not None else None),
            }, args.output)
            print(f'   -> save ({args.output}, val_acc={va_acc:.4f})',
                  flush=True)

    print(f'[train/{args.head}] best val_acc = {best_val_acc:.4f}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--head', required=True,
                   choices=['discard', 'riichi', 'chow', 'pong', 'kong'])
    p.add_argument('--data-dir', required=True)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--max-files', type=int, default=None)
    p.add_argument('--output', required=True)

    # Warm-start : charge un checkpoint existant de la MEME tete.
    p.add_argument('--warm-start', type=str, default=None,
                   help='chemin vers un .pt existant a charger avant '
                        'entrainement (head_type doit correspondre). '
                        'Avec warm-start, utiliser --lr 1e-4.')

    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=0,
                   help='DataLoader workers (training), 0 sur Windows')

    # Multiprocessing du PARSING (pas du training)
    p.add_argument('--parse-workers', type=int, default=1,
                   help='nombre de process pour le parsing ; 1 = serie, '
                        '>1 = multiprocessing (Windows-safe)')

    # [MODIF] Cap memoire sur le nombre de samples charges en RAM
    p.add_argument('--max-samples', type=int, default=None,
                   help='cap memoire : nombre max de samples charges en RAM '
                        'par tete. Default : auto (500k, ~4.4GB). '
                        '0 = tout charger (16GB+ requis pour discard). '
                        'Ex : --max-samples 400000 pour utiliser moins de RAM.')

    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--n-blocks', type=int, default=16)
    p.add_argument('--binary-hidden', type=int, default=256)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()