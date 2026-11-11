"""
Etend un cache samples_all.pkl existant en parsant des fichiers
supplementaires, sans re-parser ceux deja presents.

Usage :
    py -3.14 -m mahjong_ai.sl.extend_cache ^
        --data-dir data/tenhou/2015 ^
        --cache-path cache/sl_2015/samples_all.pkl ^
        --existing-count 1000 ^
        --new-count 10000 ^
        --parse-workers 5
"""
import argparse
import os
import pickle
import time
from multiprocessing import Pool

from mahjong_ai.sl.dataset import discover_files, _parse_one_file, _format_duration


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--cache-path', required=True)
    p.add_argument('--existing-count', type=int, required=True,
                   help='nombre de fichiers deja parses (file_id 0..N-1)')
    p.add_argument('--new-count', type=int, required=True,
                   help='nombre TOTAL de fichiers voulus au final')
    p.add_argument('--parse-workers', type=int, default=5)
    args = p.parse_args()

    # --------- charge le cache existant --------- #
    print(f'[extend] chargement {args.cache_path}...', flush=True)
    t0 = time.time()
    with open(args.cache_path, 'rb') as f:
        existing = pickle.load(f)
    print(f'[extend] {len(existing)} samples existants '
          f'en {time.time()-t0:.1f}s', flush=True)

    existing_file_ids = {s[3] for s in existing}
    print(f'[extend] file_ids existants : 0..{max(existing_file_ids)} '
          f'({len(existing_file_ids)} fichiers)', flush=True)

    # --------- fichiers a parser --------- #
    # Meme seed=42, meme shuffle -> on recupere le meme ordre
    all_files = discover_files(args.data_dir, max_files=None)
    if args.new_count > len(all_files):
        raise RuntimeError(f'Seulement {len(all_files)} fichiers dispo, '
                           f'{args.new_count} demandes')
    # Renumerote : les nouveaux file_id commencent a args.existing_count
    new_tasks = [
        (args.existing_count + i, all_files[args.existing_count + i])
        for i in range(args.new_count - args.existing_count)
    ]
    print(f'[extend] {len(new_tasks)} nouveaux fichiers a parser '
          f'(file_id {args.existing_count}..{args.new_count-1})', flush=True)

    # --------- parse avec multiprocessing --------- #
    t_start = time.time()
    new_samples = []
    skipped = 0
    log_every = max(1, len(new_tasks) // 20)

    with Pool(processes=args.parse_workers) as pool:
        for done, (fid, samples) in enumerate(
                pool.imap_unordered(_parse_one_file, new_tasks, chunksize=1), 1):
            if samples is None:
                skipped += 1
                continue
            new_samples.extend(samples)
            if done % log_every == 0 or done == len(new_tasks):
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                eta = (len(new_tasks) - done) / max(rate, 1e-6)
                print(f'  [{done:6d}/{len(new_tasks)}] '
                      f'elapsed={_format_duration(elapsed)}  '
                      f'ETA={_format_duration(eta)}  '
                      f'{rate:.3f} f/s  '
                      f'total_new={len(new_samples)}', flush=True)

    print(f'[extend] parsing termine : {len(new_samples)} nouveaux samples '
          f'en {_format_duration(time.time()-t_start)} '
          f'({skipped} echecs)', flush=True)

    # --------- fusion --------- #
    print(f'[extend] fusion...', flush=True)
    all_samples = existing + new_samples
    print(f'[extend] total : {len(all_samples)} samples', flush=True)

    # --------- sauvegarde --------- #
    # On ecrit dans un nouveau fichier puis on remplace (atomicite)
    tmp = args.cache_path + '.tmp'
    print(f'[extend] ecriture {tmp}...', flush=True)
    t0 = time.time()
    with open(tmp, 'wb') as f:
        pickle.dump(all_samples, f, protocol=4)
    size_gb = os.path.getsize(tmp) / 1e9
    print(f'[extend] ecrit {size_gb:.2f} GB en {time.time()-t0:.1f}s',
          flush=True)

    os.replace(tmp, args.cache_path)
    print(f'[extend] cache final : {args.cache_path} ({size_gb:.2f} GB)',
          flush=True)


if __name__ == '__main__':
    main()