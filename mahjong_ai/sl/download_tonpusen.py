"""
Telecharge les parties tonpusen du dataset hhim8826/tenhou-houou-mjai
et les convertit en fichiers .mjai.json.zst compatibles avec le parser.

Source : https://huggingface.co/datasets/hhim8826/tenhou-houou-mjai
Filtre : kyoku_first == 4 (tonpusen / 鳳東)
Sortie : data/tenhou_tonpusen/*.mjai.json.zst

Usage :
    python -m mahjong_ai.sl.download_tonpusen \
        --output-dir data/tenhou_tonpusen \
        --max-games 50000

Le dataset hhim8826 contient ~29 875 parties tonpusen sur un total de
~2 120 954 parties (98.59% hanchan, 1.41% tonpusen).
"""
import argparse
import json
import os
import sys
import time

import pyarrow.parquet as pq
import zstandard as zstd

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("huggingface_hub requis : pip install huggingface_hub")
    sys.exit(1)


def download_tonpusen(output_dir, max_games=None):
    """
    Telecharge les shards parquet du dataset hhim8826, filtre les tonpusen
    (kyoku_first=4), et ecrit chaque partie en .mjai.json.zst individuel.

    Returns:
        int: nombre de parties tonpusen extraites
    """
    os.makedirs(output_dir, exist_ok=True)
    repo_id = "hhim8826/tenhou-houou-mjai"

    # Lister les fichiers parquet disponibles dans le repo
    print(f"[download] listing des fichiers dans {repo_id}...")
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"[ERREUR] Impossible de lister les fichiers : {e}")
        print("Verifie ta connexion et que huggingface_hub est a jour.")
        return 0

    # Filtrer les fichiers parquet (shards)
    parquet_files = sorted(
        f for f in all_files if f.endswith('.parquet')
    )
    print(f"[download] {len(parquet_files)} shards parquet trouves")

    if not parquet_files:
        print("[ERREUR] Aucun fichier parquet trouve dans le repo")
        return 0

    total_tonpusen = 0
    total_processed = 0
    t_start = time.time()

    # Compresser en zstd pour compatibilite avec le parser
    cctx = zstd.ZstdCompressor(level=3)

    for shard_idx, shard_name in enumerate(parquet_files):
        if max_games and total_tonpusen >= max_games:
            break

        print(f"\n[download] shard {shard_idx+1}/{len(parquet_files)}: "
              f"{shard_name}")

        # Telecharger le shard
        try:
            shard_path = hf_hub_download(
                repo_id=repo_id,
                filename=shard_name,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"  [skip] {shard_name} : {e}")
            continue

        # Lire le parquet en batchs
        try:
            pf = pq.ParquetFile(shard_path)
            n_rows = pf.metadata.num_rows
            print(f"  {n_rows} rows dans ce shard")
        except Exception as e:
            print(f"  [skip] impossible de lire {shard_name} : {e}")
            continue

        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=512)):
            if max_games and total_tonpusen >= max_games:
                break

            columns = batch.to_pylist()

            for row in columns:
                total_processed += 1

                # Filtrer : garder uniquement les tonpusen
                kyoku_first = row.get("kyoku_first")
                if kyoku_first != 4:
                    continue

                # Extraire le log MJAI
                events_text = row.get("events", "")
                if not events_text:
                    continue

                game_id = row.get("game_id", "")
                if not game_id:
                    game_id = f"unknown_{total_tonpusen:08d}"

                # Verifications basiques de validite
                if '"start_game"' not in events_text[:200]:
                    continue
                if '"end_game"' not in events_text[-200:]:
                    continue

                # Verifier que ce n'est pas un duplicata
                output_path = os.path.join(
                    output_dir, f"{game_id}.mjai.json.zst"
                )
                if os.path.exists(output_path):
                    continue

                # Compresser et ecrire
                try:
                    compressed = cctx.compress(
                        events_text.encode("utf-8"))
                    with open(output_path, "wb") as f:
                        f.write(compressed)
                    total_tonpusen += 1
                except Exception as e:
                    print(f"  [warn] echec ecriture {game_id} : {e}")
                    continue

                # Log de progression
                if total_tonpusen % 500 == 0:
                    elapsed = time.time() - t_start
                    print(f"  ... {total_tonpusen} tonpusen extraits "
                          f"({total_processed} parties scannees, "
                          f"{elapsed:.0f}s)")

        # Supprimer le shard telecharge pour liberer l'espace disque
        try:
            os.remove(shard_path)
        except Exception:
            pass

    elapsed = time.time() - t_start
    print(f"\n[done] {total_tonpusen} parties tonpusen extraites "
          f"depuis {total_processed} parties totales "
          f"en {elapsed:.0f}s")
    print(f"[done] ratio tonpusen : "
          f"{total_tonpusen/max(total_processed,1)*100:.2f}%")
    print(f"[done] output : {output_dir}")

    # Afficher la taille totale
    total_size = 0
    for f in os.listdir(output_dir):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)
    print(f"[done] taille totale : {total_size / 1024 / 1024:.1f} MB")

    return total_tonpusen


def main():
    p = argparse.ArgumentParser(
        description="Telecharge les tonpusen de hhim8826/tenhou-houou-mjai"
    )
    p.add_argument('--output-dir', type=str, required=True,
                   help='dossier de sortie pour les .mjai.json.zst')
    p.add_argument('--max-games', type=int, default=None,
                   help='limite du nombre de parties tonpusen a extraire')
    p.add_argument('--repo-id', type=str,
                   default='hhim8826/tenhou-houou-mjai',
                   help='repo HuggingFace (default: hhim8826)')
    args = p.parse_args()

    download_tonpusen(args.output_dir, args.max_games)


if __name__ == '__main__':
    main()