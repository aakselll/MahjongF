"""
Telecharge un shard annuel depuis nikketryhard/tenhou-houou-mjai sur HF.

Les fichiers sont des .tar NON compresses contenant des .mjai.json.zst
(un JSON par ligne, compresse en zstd).

Usage :
    python -m mahjong_ai.sl.download_mjai --year 2024
    python -m mahjong_ai.sl.download_mjai --year 2024 --extract-only
"""
import argparse
import os
import tarfile
import time

from huggingface_hub import hf_hub_download

REPO_ID = "nikketryhard/tenhou-houou-mjai"


def download_shard(year, cache_dir=None):
    """Telecharge tenhou-houou-YYYY.tar depuis HF."""
    filename = f"tenhou-houou-{year}.tar"
    print(f"[download] {REPO_ID}/{filename}...")
    t0 = time.time()
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    size_gb = os.path.getsize(path) / 1e9
    print(f"[download] {path} ({size_gb:.2f} Go) en {time.time()-t0:.1f}s")
    return path


def extract_tar(tar_path, output_dir):
    """Extrait le .tar dans output_dir/YYYY/."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[extract] {tar_path} -> {output_dir}...")
    t0 = time.time()
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        print(f"[extract] {len(members)} fichiers dans l'archive")
        tar.extractall(output_dir, members=members)
    print(f"[extract] termine en {time.time()-t0:.1f}s")
    return output_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--extract-only", action="store_true",
                   help="ne pas re-telecharger, juste extraire (le .tar doit deja etre la)")
    args = p.parse_args()

    output_dir = args.output_dir or f"data/tenhou/{args.year}"

    if args.extract_only:
        # Chercher le .tar dans le cache HF
        tar_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"tenhou-houou-{args.year}.tar",
            repo_type="dataset",
            cache_dir=args.cache_dir,
            local_files_only=True,
        )
    else:
        tar_path = download_shard(args.year, args.cache_dir)

    extract_tar(tar_path, output_dir)
    print(f"[done] fichiers extraits dans {output_dir}/")


if __name__ == "__main__":
    main()