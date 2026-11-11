"""
Dataset SL partage entre les 5 tetes Suphx, avec cache incremental par shards.

[MODIF 2026-10-13] Suppression du parametre game_format. Le format
(tonpuu/hanchan) est detecte automatiquement par fichier (voir
mjai_parser._detect_format) et stocke en METADATA (5e element des
samples), pas dans les canaux d'entree.

[FIX MEM 2026-10-14] Cap memoire adaptatif. Sur 16GB de RAM, le dataset
complet ne tient pas (discard seul ~5M samples x 8.7KB = ~43GB). Si le
nombre de samples correspondants depasse max_samples, un taux
d'echantillonnage uniforme est calcule depuis le PREMIER shard (les
fichiers etant shuffles avant le parse, chaque shard est un echantillon
representatif du dataset) et applique a tous les shards. Deterministe
(seed fixe) : deux runs avec le meme cap chargent les memes samples.

[FIX 2025-XX-3] Systeme de shards pour reprise sur crash :
  - Le parse est decoupe en shards de SHARD_SIZE fichiers chacun.
  - Chaque shard est ecrit atomiquement (.tmp puis os.replace).
  - Si crash, on reprend au dernier shard ecrit (pas de re-parse).
  - Signature des fichiers verifiee au resume.

Layout du cache :
  cache/sl_v2/
  +-- _shards/
  |   +-- shard_000000_000100.pkl   (fichiers 0..99)
  |   +-- ...
  +-- _meta.pkl                     (signature + n_channels + version)

Chaque shard contient des tuples :
  (obs_u8, atype, act, fid, fmt)
    - obs_u8 : np.array (N_CHANNELS, 34) uint8
    - atype  : int (type d'action)
    - act    : int (label)
    - fid    : int (file_id)
    - fmt    : str ("tonpuu" ou "hanchan") -- METADATA, pas une feature
"""
import glob
import hashlib
import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from mahjong_ai.sl.mjai_parser import parse_mjai_zst_file, events_to_samples_all_povs
from mahjong_ai.envs.multi_agent import DISCARD, PONG, CHOW, KONG, RIICHI
from mahjong_ai.envs.suphx_features import N_CHANNELS


HEAD_TO_ACTION_TYPE = {
    'discard': DISCARD,
    'riichi': RIICHI,
    'chow': CHOW,
    'pong': PONG,
    'kong': KONG,
}

SHARD_SIZE = 100        # nombre de fichiers par shard
SIG_SAMPLE = 1000       # nombre de fichiers utilises pour la signature
LOG_INTERVAL_S = 20     # intervalle max entre deux logs de progression
SAMPLE_VERSION = 2      # format des samples : 5-tuples avec fmt en metadata

# [FIX MEM] Cap par defaut : ~500k samples x 8.7KB ~ 4.4GB de RAM.
# Les petites tetes (kong, pong, riichi) ne l'atteignent generalement pas
# et passent a 100%. Passer None (ou 0 en CLI) pour tout charger.
DEFAULT_MAX_SAMPLES = 500_000


# ====================================================================== #
# Helpers top-level (picklables pour multiprocessing Windows)
# ====================================================================== #
def _parse_one_file(task):
    """Parse UN fichier -> (file_id, samples_list_or_None, game_format).

    task : tuple (file_id, path) -- 2 elements.
    Retour : tuple de 3 (file_id, samples, game_format).
    """
    file_id, path = task
    try:
        events = parse_mjai_zst_file(path)
        if not events:
            return file_id, [], None
        # Nouvelle signature : pas de game_format=, retour = tuple
        samples, game_format = events_to_samples_all_povs(events)
        out = []
        for obs, atype, action, _mask in samples:
            obs_u8 = np.clip(
                np.round(np.asarray(obs, dtype=np.float32) * 255.0),
                0, 255,
            ).astype(np.uint8)
            # fmt stocke en 5e element (metadata, pas une feature)
            out.append((obs_u8, int(atype), int(action), file_id, game_format))
        return file_id, out, game_format
    except Exception:
        return file_id, None, None


def _format_duration(seconds):
    if seconds < 0 or seconds != seconds:
        return '?'
    if seconds < 60:
        return f'{seconds:.1f}s'
    if seconds < 3600:
        return f'{int(seconds//60)}m{int(seconds%60):02d}s'
    return f'{int(seconds//3600)}h{int((seconds%3600)//60):02d}m'


def _files_signature(files):
    """Hash deterministe sur les SIG_SAMPLE premiers basenames."""
    h = hashlib.md5()
    for f in files[:SIG_SAMPLE]:
        h.update(os.path.basename(f).encode())
        h.update(b'\n')
    return h.hexdigest()


def discover_files(data_dir, max_files=None, seed=42):
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.zst'), recursive=True))
    if not files:
        raise RuntimeError(f'Aucun .zst trouve dans {data_dir}')
    random.Random(seed).shuffle(files)
    if max_files:
        files = files[:max_files]
    return files


def _has_shards(shard_dir):
    if not os.path.isdir(shard_dir):
        return False
    return any(f.startswith('shard_') and f.endswith('.pkl')
               for f in os.listdir(shard_dir))


def _scan_last_file_id(shard_dir):
    """Renvoie l'ID exclusif du dernier fichier parse (0 si aucun shard)."""
    if not os.path.isdir(shard_dir):
        return 0
    shards = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.pkl')))
    if not shards:
        return 0
    name = os.path.basename(shards[-1]).replace('.pkl', '')
    # format : shard_XXXXXX_YYYYYY.pkl, YYYYYY = exclusif
    return int(name.split('_')[2])


def _count_shards(shard_dir):
    if not os.path.isdir(shard_dir):
        return 0
    return len(glob.glob(os.path.join(shard_dir, 'shard_*.pkl')))


def _load_all_shards_filtered(shard_dir, action_type, verbose=False,
                              head_type='?', max_samples=None, seed=42):
    """Charge tous les shards, filtre par action_type.

    [FIX MEM] Si max_samples est fixe et que le total estime depasse le cap,
    un taux d'echantillonnage uniforme est calcule depuis les 5 PREMIERS
    shards (la variance inter-shards etant elevee, un seul shard sous-estime
    parfois le total d'un facteur 4x). Un trim final de securite garantit
    ensuite le cap quoi qu'il arrive, meme si l'estimation etait encore
    trop basse. Deterministe (seeds fixes).

    Retourne une liste de 4-tuples (obs_u8, act, fid, fmt).
    """
    shard_files = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.pkl')))
    if not shard_files:
        return []

    # --- Estimation du taux sur 5 shards + marge de securite ---
    rate = 1.0
    if max_samples is not None:
        n_est = min(5, len(shard_files))
        est_matching = 0
        for sf in shard_files[:n_est]:
            with open(sf, 'rb') as f:
                s = pickle.load(f)
            est_matching += sum(1 for x in s if len(x) >= 2
                                and x[1] == action_type)
            del s
        est_total = est_matching * len(shard_files) / n_est
        if est_total > max_samples and est_total > 0:
            # Marge de 15% : si l'estimation sous-estime encore, le trim
            # final corrige ; si elle surestime, on perd au pire 15% de cap.
            rate = (0.85 * max_samples) / float(est_total)
        if verbose:
            print(f'  [load/{head_type}] estimation : ~{int(est_total)} '
                  f'correspondants, cap={max_samples} -> rate={rate:.3f}',
                  flush=True)

    rng = random.Random(seed)
    out = []
    total_raw = 0
    total_matching = 0
    for i, sf in enumerate(shard_files):
        with open(sf, 'rb') as f:
            shard = pickle.load(f)
        total_raw += len(shard)
        for sample in shard:
            obs_u8, atype, act, fid, fmt = sample
            if atype == action_type:
                total_matching += 1
                if rate >= 1.0 or rng.random() < rate:
                    out.append((obs_u8, act, fid, fmt))
        del shard  # liberer le shard transient immediatement
        if verbose and (i + 1) % 20 == 0:
            print(f'  [load/{head_type}] {i+1}/{len(shard_files)} shards '
                  f'charges, {len(out)} samples gardes', flush=True)

    # --- Trim final de securite : le cap est JAMAIS depasse ---
    if max_samples is not None and len(out) > max_samples:
        trim_rng = random.Random(seed + 1)
        out = trim_rng.sample(out, max_samples)
        if verbose:
            print(f'  [load/{head_type}] trim securite : '
                  f'{max_samples} / {total_matching} gardes '
                  f'(estimation trop basse, corrigee)', flush=True)

    if verbose:
        mem_gb = len(out) * 8.8 / 1024 / 1024
        print(f'  [load/{head_type}] {len(shard_files)} shards, '
              f'{total_raw} samples bruts, {total_matching} correspondants, '
              f'{len(out)} gardes (rate={rate:.3f}, ~{mem_gb:.1f} GB RAM)',
              flush=True)
    return out
# ====================================================================== #
# Dataset
# ====================================================================== #
class PolicyDataset(Dataset):
    """Echantillons (obs (N_CHANNELS, 34) uint8, label, file_id, format)
    pour UNE tete.

    [MODIF] Plus de parametre game_format : detecte par fichier.
    self.samples : liste de 4-tuples (obs_u8, act, fid, fmt).
    self.sample_formats : liste parallele des formats (pour sampler).
    Cache partage entre les 5 tetes : contient tous les types d'action.
    Filtre en memoire au chargement.

    [FIX MEM] max_samples : cap sur le nombre de samples charges en RAM.
    None = tout charger (a ne faire que si assez de RAM). Le default
    (DEFAULT_MAX_SAMPLES) protege les machines 16GB.
    """

    def __init__(self, files, head_type, cache_path=None, verbose=True,
                 parse_workers=1, shard_size=SHARD_SIZE,
                 max_samples=DEFAULT_MAX_SAMPLES):
        assert head_type in HEAD_TO_ACTION_TYPE, f'Tete inconnue : {head_type}'
        self.head_type = head_type
        self.action_type = HEAD_TO_ACTION_TYPE[head_type]
        self._parse_workers = max(1, int(parse_workers))
        self._shard_size = max(1, int(shard_size))
        self._max_samples = max_samples
        self.sample_formats = []

        if cache_path is None:
            # Pas de cache -> parse en memoire sans shards
            self._parse_no_cache(files, verbose)
            return

        cache_dir = os.path.dirname(cache_path) or '.'
        shard_dir = os.path.join(cache_dir, '_shards')
        meta_path = os.path.join(cache_dir, '_meta.pkl')
        os.makedirs(cache_dir, exist_ok=True)

        # 1) Verification du meta (invalidation auto si layout change)
        if os.path.isfile(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            if (meta.get('n_channels') != N_CHANNELS
                    or meta.get('sample_version') != SAMPLE_VERSION):
                raise RuntimeError(
                    f'Cache incompatible : '
                    f'n_channels={meta.get("n_channels")}/'
                    f'{N_CHANNELS}, sample_version='
                    f'{meta.get("sample_version")}/{SAMPLE_VERSION}. '
                    f'Supprimez {shard_dir}/ et {meta_path}.'
                )
        elif _has_shards(shard_dir):
            raise RuntimeError(
                f'Metadonnees absentes pour le cache {shard_dir}. '
                'Supprimez ce cache pour le regenerer.'
            )

        # 2) Etat actuel
        n_files_total = len(files)
        start_file_id = _scan_last_file_id(shard_dir)

        if verbose:
            n_shards = _count_shards(shard_dir)
            if n_shards:
                print(f'[dataset/{head_type}] etat cache : '
                      f'{n_shards} shards, {start_file_id}/{n_files_total} '
                      f'fichiers parses', flush=True)
            else:
                print(f'[dataset/{head_type}] cache vide, parse de '
                      f'{n_files_total} fichiers', flush=True)

        # 3) Parse le reste si necessaire
        if start_file_id < n_files_total:
            self._parse_range(files, start_file_id, n_files_total,
                              shard_dir, meta_path, verbose)
        elif start_file_id > n_files_total:
            if verbose:
                print(f'[dataset/{head_type}] ATTENTION : {start_file_id} '
                      f'fichiers deja parses mais seulement {n_files_total} '
                      f'demandes (le cache contient plus de donnees que '
                      f'necessaire, on charge tout)', flush=True)

        # 4) Chargement filtre (avec cap memoire)
        if verbose:
            t0 = time.time()
            cap_str = (f'cap={max_samples}' if max_samples
                       else 'cap=aucun (tout charger)')
            print(f'[dataset/{head_type}] chargement + filtre des shards '
                  f'({cap_str})...', flush=True)
        filtered = _load_all_shards_filtered(
            shard_dir, self.action_type, verbose=verbose,
            head_type=head_type,
            max_samples=self._max_samples, seed=42,
        )
        self.samples = filtered
        self.sample_formats = [s[3] for s in filtered]

        if verbose:
            n_ton = sum(1 for f in self.sample_formats if f == 'tonpuu')
            n_han = sum(1 for f in self.sample_formats if f == 'hanchan')
            print(f'[dataset/{head_type}] {len(self.samples)} echantillons '
                  f'(tonpuu={n_ton}, hanchan={n_han}) en '
                  f'{time.time()-t0:.1f}s', flush=True)

    # ------------------------------------------------------------------ #
    def _parse_no_cache(self, files, verbose):
        """Parse sans cache (mode in-memory).

        [FIX MEM] Trim incremental : si le nombre de samples depasse le cap,
        on resample aleatoirement a max_samples. Le pic memoire reste borne
        a cap + 100k samples.
        """
        if verbose:
            print(f'[dataset/{self.head_type}] parse sans cache '
                  f'({self._parse_workers} worker(s))...', flush=True)
        t0 = time.time()
        tasks = [(i, path) for i, path in enumerate(files)]
        out = []

        def _maybe_trim():
            nonlocal out
            if (self._max_samples is not None
                    and len(out) >= self._max_samples + 100_000):
                out = random.Random(123).sample(out, self._max_samples)

        if self._parse_workers <= 1:
            for task in tasks:
                _fid, samples, _fmt = _parse_one_file(task)
                if samples:
                    for s in samples:
                        obs_u8, atype, act, fid, fmt = s
                        if atype == self.action_type:
                            out.append((obs_u8, act, fid, fmt))
                _maybe_trim()
        else:
            import multiprocessing as mp
            with mp.Pool(processes=self._parse_workers) as pool:
                for _fid, samples, _fmt in pool.imap(
                        _parse_one_file, tasks, chunksize=1):
                    if samples:
                        for s in samples:
                            obs_u8, atype, act, fid, fmt = s
                            if atype == self.action_type:
                                out.append((obs_u8, act, fid, fmt))
                    _maybe_trim()

        self.samples = out
        self.sample_formats = [s[3] for s in out]
        if verbose:
            print(f'[dataset/{self.head_type}] {len(out)} samples '
                  f'en {_format_duration(time.time()-t0)}', flush=True)

    # ------------------------------------------------------------------ #
    def _parse_range(self, files, start_id, end_id, shard_dir, meta_path, verbose):
        """Parse files[start_id:end_id] et ecrit les shards."""
        n = end_id - start_id
        if n <= 0:
            return

        # Verification de la signature
        sig = _files_signature(files)
        os.makedirs(shard_dir, exist_ok=True)
        if os.path.isfile(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            if meta.get('signature') != sig:
                raise RuntimeError(
                    f'Signature mismatch !\n'
                    f'  Cache cree pour signature : {meta.get("signature")}\n'
                    f'  Fichiers actuels signature : {sig}\n'
                    f'Le data-dir ou les fichiers ont change. Supprimez '
                    f'{shard_dir}/ et {meta_path} pour reparcourir, ou '
                    f'revenez a la config d\'origine.'
                )
        else:
            with open(meta_path, 'wb') as f:
                pickle.dump({
                    'signature': sig,
                    'created': time.time(),
                    'n_channels': N_CHANNELS,
                    'sample_version': SAMPLE_VERSION,
                }, f)

        if verbose:
            print(f'[dataset/{self.head_type}] parsing {n} fichiers '
                  f'({start_id}..{end_id-1}), {self._parse_workers} worker(s), '
                  f'shard_size={self._shard_size}...', flush=True)

        t_start = time.time()
        tasks = [(fid, files[fid]) for fid in range(start_id, end_id)]

        # Buffers de shard (mutables via dict)
        state = {
            'buffer': [],
            'start': start_id,
            'end': start_id,
            'files_in_shard': 0,
            'done': 0,
        }
        log_every = max(1, min(200, n // 100))
        last_log_time = [time.time()]

        def flush_shard():
            if not state['buffer']:
                return
            name = f"shard_{state['start']:06d}_{state['end']:06d}.pkl"
            path = os.path.join(shard_dir, name)
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(state['buffer'], f, protocol=4)
            os.replace(tmp, path)
            if verbose:
                print(f"  [shard] {name} ({len(state['buffer'])} samples, "
                      f"{state['files_in_shard']} fichiers)", flush=True)
            last_log_time[0] = time.time()
            state['buffer'] = []
            state['start'] = state['end']
            state['files_in_shard'] = 0

        def process(fid, samples):
            state['done'] += 1
            if samples:
                state['buffer'].extend(samples)
            state['end'] = max(state['end'], fid + 1)
            state['files_in_shard'] += 1
            if state['files_in_shard'] >= self._shard_size:
                flush_shard()

        def log_progress():
            if not verbose:
                return
            now = time.time()
            should_log = (
                state['done'] % log_every == 0
                or (now - last_log_time[0]) >= LOG_INTERVAL_S
                or state['done'] == n
            )
            if not should_log:
                return
            last_log_time[0] = now
            elapsed = now - t_start
            rate = state['done'] / max(elapsed, 1e-6)
            eta = (n - state['done']) / max(rate, 1e-6)
            print(f"  [{state['done']}/{n}] "
                  f"elapsed={_format_duration(elapsed)}  "
                  f"ETA={_format_duration(eta)}  "
                  f"{rate:.3f} f/s  "
                  f"buffer={len(state['buffer'])}", flush=True)

        if self._parse_workers <= 1:
            for task in tasks:
                _fid, samples, _fmt = _parse_one_file(task)
                process(_fid, samples)
                log_progress()
        else:
            import multiprocessing as mp
            with mp.Pool(processes=self._parse_workers) as pool:
                for fid, samples, _fmt in pool.imap(
                        _parse_one_file, tasks, chunksize=1):
                    process(fid, samples)
                    log_progress()

        flush_shard()
        if verbose:
            print(f'[dataset/{self.head_type}] parse termine : '
                  f'{state["done"]} fichiers en '
                  f'{_format_duration(time.time()-t_start)}', flush=True)

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        obs_u8, label, _fid, _fmt = self.samples[i]
        obs_f32 = obs_u8.astype(np.float32) / 255.0
        return torch.from_numpy(obs_f32), torch.tensor(label, dtype=torch.long)

    def split_by_file(self, val_fraction=0.1, seed=42):
        file_ids = sorted({s[2] for s in self.samples})
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(file_ids))
        n_val = max(int(len(file_ids) * val_fraction), 1)
        val_files = set(np.array(file_ids)[perm[:n_val]].tolist())

        train_idx, val_idx = [], []
        for i, (_, _, fid, _fmt) in enumerate(self.samples):
            (val_idx if fid in val_files else train_idx).append(i)
        return train_idx, val_idx

    def make_subsets(self, val_fraction=0.1, seed=42):
        tr, va = self.split_by_file(val_fraction, seed)
        return Subset(self, tr), Subset(self, va)