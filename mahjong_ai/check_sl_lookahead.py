"""
Diagnostic : un checkpoint SL a-t-il ete entraine avec le lookahead ACTIF ?

Le layout d'entree contient TOUJOURS les 80 canaux lookahead
(canaux _OFFSET['lookahead'] .. +80, i.e. 188:268 sur le layout courant) --
qu'ils aient ete REMPLIS par le DFS au moment du parse ou laisses a zero.
Un checkpoint seul ne prouve donc rien : on croise trois niveaux de preuve.

  [1] CONFIG   in_channels du checkpoint (layout de l'epoque).
               279 = courant (base 268 dont lookahead 188:268, + 11 rouges)
               268 = avant les canaux rouges (lookahead deja a 188:268)
               autre = layout non reconnu (probablement pre-lookahead)
                       -> modele aveugle par construction.
  [2] POIDS    normes des colonnes d'entree de trunk.input_conv.
               Optimiseur = AdamW : un canal dont l'entree est TOUJOURS
               nulle ne recoit aucun gradient -> ses poids restent a
               leur valeur d'initialisation. On compare la norme du bloc
               lookahead a celle de modeles frais (N seeds).
  [3] DONNEES  stats du cache : % de samples dont le bloc lookahead est
     + ABLATION non nul. Puis ablation : accuracy du modele avec lookahead
               intact vs lookahead mis a zero, sur ces memes samples.
                 - entraîne sur donnees remplies : priver du lookahead
                   FAIT CHUTER l'accuracy (delta > 0)
                 - entraîne sur des zeros : le lookahead non nul traverse
                   des poids jamais entraînes = BRUIT -> l'accuracy
                   REMONTE quand on le met a zero (delta < 0)

Usage (a la racine du repo, Windows) :
  py -3.14 -m mahjong_ai.check_sl_lookahead ^
      --checkpoint checkpoints/sl_tonpusen_la/discard.pt ^
      --cache-dir  cache/sl_tonpusen_la

  --cache-dir est optionnel (sans lui : niveaux 1-2 seulement).
  L'ablation n'est calculee que pour la tete discard (labels 0..33).
"""
import argparse
import glob
import os
import pickle

import numpy as np
import torch

from mahjong_ai.agents.suphx_cnn import SuphxModel
from mahjong_ai.envs.suphx_features import (
    N_CHANNELS, N_BASE_CHANNELS, N_RED_CHANNELS, LOOKAHEAD_CHANNELS,
    N_TILES, _OFFSET)
from mahjong_ai.envs.multi_agent import DISCARD

LA0 = _OFFSET['lookahead']
LA1 = LA0 + LOOKAHEAD_CHANNELS
ORACLE_DIM = 4          # cf. suphx_cnn.IN_CHANNELS_ORACLE (RL uniquement)

KNOWN_LAYOUTS = {
    N_BASE_CHANNELS: 'base 268, lookahead en %d:%d' % (LA0, LA1),
    N_BASE_CHANNELS + N_RED_CHANNELS:
        'base 268 (lookahead %d:%d) + 11 rouges -- layout COURANT'
        % (LA0, LA1),
    N_BASE_CHANNELS + ORACLE_DIM:
        'base + 4 oracle (RL, pas SL)',
    N_BASE_CHANNELS + N_RED_CHANNELS + ORACLE_DIM:
        'base + rouges + oracle (RL, pas SL)',
}

TYPE_NAMES = {DISCARD: 'discard', 1: 'riichi', 2: 'chow', 3: 'pong', 4: 'kong'}


# ====================================================================== #
# [1] Checkpoint
# ====================================================================== #

def load_checkpoint(path):
    try:
        ck = torch.load(path, map_location='cpu')
    except Exception:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        return ck['model_state_dict'], ck
    return ck, {}          # checkpoint "nu" (state_dict seul)


def infer_arch(sd, cfg):
    """Canaux/blocs depuis le state_dict si le config est incomplet."""
    channels = cfg.get('channels')
    n_blocks = cfg.get('n_blocks')
    binary_hidden = cfg.get('binary_hidden', 256)
    if channels is None:
        w = sd.get('trunk.blocks.0.conv1.weight')
        channels = int(w.shape[0]) if w is not None else 256
    if n_blocks is None:
        n_blocks = len([k for k in sd
                        if k.startswith('trunk.blocks.')
                        and k.endswith('.conv1.weight')])
        n_blocks = max(n_blocks, 1)
    return int(channels), int(n_blocks), int(binary_hidden)


# ====================================================================== #
# [2] Poids d'entree
# ====================================================================== #

def per_channel_norms(w):
    """w : (C_out, C_in, 3, 1) -> norme L2 par canal d'entree (C_in,)."""
    return np.sqrt((w.astype(np.float64) ** 2).sum(axis=(0, 2, 3)))


def analyze_weights(sd, cfg, in_ch, seeds):
    key = 'trunk.input_conv.weight'
    if key not in sd:
        print('  [!] cle %s introuvable : analyse de poids impossible' % key)
        return None
    w = sd[key].float().numpy()
    cin = w.shape[1]
    if cin != in_ch:
        print('  [!] in_channels config (%d) != poids (%d) -- on prend %d'
              % (in_ch, cin, cin))
        in_ch = cin

    norms = per_channel_norms(w)
    base_mask = np.ones(cin, dtype=bool)
    base_mask[LA0:LA1] = False
    base_mask[N_BASE_CHANNELS:] = False      # exclut rouges (+ oracle si RL)
    la_norms = norms[LA0:LA1]
    base_norms = norms[base_mask]
    zero_frac = float((w[:, LA0:LA1, :, :] == 0).mean())

    # Reference : init fraiche avec la meme archi / meme in_channels
    channels, n_blocks, binary_hidden = infer_arch(sd, cfg)
    fresh = []
    for s in range(seeds):
        torch.manual_seed(1000 + s)
        m = SuphxModel('discard', in_channels=in_ch, channels=channels,
                       n_blocks=n_blocks, binary_hidden=binary_hidden)
        fresh.append(per_channel_norms(
            m.trunk.input_conv.weight.detach().numpy())[LA0:LA1].mean())
    fresh = np.array(fresh)
    dev = (la_norms.mean() - fresh.mean()) / fresh.mean() * 100.0

    print('  norme L2 moyenne par canal d\'entree :')
    print('    base (hors lookahead/rouges) : %.4f' % base_norms.mean())
    print('    lookahead %d:%d        : %.4f   (ratio/base %.2f)'
          % (LA0, LA1, la_norms.mean(),
             la_norms.mean() / max(base_norms.mean(), 1e-12)))
    if cin >= N_CHANNELS:
        red = norms[N_BASE_CHANNELS:N_CHANNELS]
        print('    rouges %d:%d             : %.4f   (ratio/base %.2f)'
              % (N_BASE_CHANNELS, N_CHANNELS, red.mean(),
                 red.mean() / max(base_norms.mean(), 1e-12)))
    print('  poids lookahead exactement nuls : %.2f %%' % (zero_frac * 100))
    print('  init fraiche (%d seeds)        : %.4f +/- %.4f'
          % (seeds, fresh.mean(), fresh.std()))
    print('  -> ecart du checkpoint          : %+.1f %%' % dev)

    if zero_frac > 0.99:
        print('  >> INDICATION : bloc lookahead ENTIEREMENT NUL '
              '(zero-padding adapt_in_channels) -- aveugle par construction.')
    elif abs(dev) < 3.0:
        print('  >> INDICATION : normes lookahead a l\'initialisation -- '
              'ces canaux n\'ont probablement jamais recu de gradient\n'
              '     (entree toujours nulle pendant l\'entrainement = '
              'donnees SANS lookahead). A confirmer par l\'ablation [3].')
    else:
        print('  >> INDICATION : normes lookahead nettement ecartees de '
              'l\'init -- ces canaux ont ete ENTRAINES (donnees avec '
              'lookahead rempli).')
    return {'la_mean': float(la_norms.mean()), 'dev_pct': float(dev),
            'zero_frac': zero_frac}


# ====================================================================== #
# [3] Cache + ablation
# ====================================================================== #

def read_cache(cache_dir, n_max):
    shard_files = sorted(glob.glob(
        os.path.join(cache_dir, '_shards', 'shard_*.pkl')))
    if not shard_files:                      # ancien format samples_all.pkl
        shard_files = [f for f in sorted(glob.glob(
            os.path.join(cache_dir, '*.pkl')))
            if not os.path.basename(f).startswith('_')]

    meta = None
    mp = os.path.join(cache_dir, '_meta.pkl')
    if os.path.isfile(mp):
        with open(mp, 'rb') as f:
            meta = pickle.load(f)

    samples, by_type, obs_channels = [], {}, {}
    n_bad, n_shards = 0, 0
    for sp in shard_files:
        with open(sp, 'rb') as f:
            buf = pickle.load(f)
        n_shards += 1
        for tup in buf:
            if len(tup) != 5:
                n_bad += 1
                continue
            obs, atype, act, _fid, _fmt = tup
            obs_channels[obs.shape[0]] = obs_channels.get(obs.shape[0], 0) + 1
            by_type[atype] = by_type.get(atype, 0) + 1
            if atype == DISCARD and len(samples) < n_max:
                samples.append((obs, int(act)))
        if len(samples) >= n_max:
            break

    stats = {'n_shards': n_shards, 'by_type': by_type,
             'obs_channels': obs_channels, 'n_bad': n_bad, 'meta': meta,
             'total_shard_files': len(shard_files)}
    return samples, stats


def make_canvas(obs_u8, ckpt_in):
    """obs (n,34) uint8 -> canvas (ckpt_in,34) uint8 (base commun copie)."""
    n = obs_u8.shape[0]
    if n == ckpt_in:
        return obs_u8.copy()
    if n not in KNOWN_LAYOUTS or ckpt_in not in KNOWN_LAYOUTS:
        return None
    canvas = np.zeros((ckpt_in, N_TILES), dtype=np.uint8)
    canvas[:N_BASE_CHANNELS] = obs_u8[:N_BASE_CHANNELS]
    if n >= N_CHANNELS and ckpt_in >= N_CHANNELS:      # rouges des deux cotes
        canvas[N_BASE_CHANNELS:N_CHANNELS] = \
            obs_u8[N_BASE_CHANNELS:N_CHANNELS]
    return canvas


def ablation(model, samples, ckpt_in, device, batch_size):
    model.eval().to(device)
    la_present = np.array([s[0][LA0:LA1].sum() > 0 for s in samples])
    n_ok = 0
    accs = {'intact': 0.0, 'zero': 0.0}
    agree, kl_sum = 0.0, 0.0
    n_la = int(la_present.sum())

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            canv = [make_canvas(o, ckpt_in) for o, _ in chunk]
            if any(c is None for c in canv):
                return None
            x = np.stack(canv).astype(np.float32) / 255.0
            x_int = torch.from_numpy(x).to(device)
            x_zer = x_int.clone()
            x_zer[:, LA0:LA1, :] = 0.0
            lg_i = model(x_int)
            lg_z = model(x_zer)
            pi = torch.softmax(lg_i, dim=1)
            pz = torch.softmax(lg_z, dim=1)
            ai, az = lg_i.argmax(1), lg_z.argmax(1)
            lab = torch.tensor([a for _, a in chunk]).to(device)
            m = torch.from_numpy(la_present[i:i + batch_size]).to(device)
            n_ok += int(m.sum())
            accs['intact'] += float(((ai == lab) & m).sum())
            accs['zero'] += float(((az == lab) & m).sum())
            agree += float((ai == az)[m].sum()) if m.any() else 0.0
            kl_sum += float(((pi * (pi.add(1e-9).log()
                                      - pz.add(1e-9).log())).sum(1) * m).sum())

    if n_ok == 0:
        return None
    acc_i, acc_z = accs['intact'] / n_ok, accs['zero'] / n_ok
    delta = acc_i - acc_z
    p_bar = (acc_i + acc_z) / 2.0
    noise = 1.96 * np.sqrt(max(p_bar * (1 - p_bar), 1e-9) / n_ok)
    return {'n_la': n_ok, 'acc_intact': acc_i, 'acc_zero': acc_z,
            'delta': delta, 'noise': noise,
            'agreement': agree / n_ok, 'kl': kl_sum / n_ok}


# ====================================================================== #
# Main
# ====================================================================== #

def main():
    ap = argparse.ArgumentParser(
        description='Le checkpoint SL a-t-il ete entraine avec lookahead ?')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache-dir', default=None,
                    help='dossier du cache SL (_shards/ ou samples_all.pkl)')
    ap.add_argument('--n-samples', type=int, default=3000,
                    help='max samples discard pour les stats + ablation')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--device', default=None,
                    help='cpu (defaut) / cuda si dispo')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 68)
    print('[1/3] CHECKPOINT : %s' % args.checkpoint)
    print('=' * 68)
    sd, ck = load_checkpoint(args.checkpoint)
    cfg = ck.get('config', {}) or {}
    head = ck.get('head_type', '?')
    channels, n_blocks, binary_hidden = infer_arch(sd, cfg)
    key = 'trunk.input_conv.weight'
    in_ch = int(cfg.get('in_channels', 0)) if key not in sd \
        else int(sd[key].shape[1])
    print('  head=%s  arch=%dx%d (binary_hidden=%d)'
          % (head, channels, n_blocks, binary_hidden))
    if 'val_acc' in ck:
        print('  val_acc=%.4f (epoch %s)' % (ck['val_acc'], ck.get('epoch')))
    lay = KNOWN_LAYOUTS.get(in_ch)
    if lay:
        print('  in_channels = %d  (%s)' % (in_ch, lay))
    else:
        print('  in_channels = %d  -> LAYOUT NON RECONNU (probablement '
              'pre-lookahead).\n  Le modele n\'a pas de canaux lookahead : '
              'aveugle par construction.' % in_ch)

    print()
    print('=' * 68)
    print('[2/3] POIDS D\'ENTREE (trunk.input_conv, noyau 3x1)')
    print('=' * 68)
    w_info = None
    if lay:
        w_info = analyze_weights(sd, cfg, in_ch, args.seeds)
    else:
        print('  saute : layout non reconnu')

    a_info = None
    if args.cache_dir:
        print()
        print('=' * 68)
        print('[3/3] CACHE : %s' % args.cache_dir)
        print('=' * 68)
        samples, stats = read_cache(args.cache_dir, args.n_samples)
        if stats['meta']:
            mt = stats['meta']
            print('  _meta.pkl : n_channels=%s sample_version=%s'
                  % (mt.get('n_channels'), mt.get('sample_version')))
        print('  %d shard(s) sur %d, %d sample(s) discard retenus (cap %d)'
              % (stats['n_shards'], stats['total_shard_files'],
                 len(samples), args.n_samples))
        ty = ', '.join('%s=%d' % (TYPE_NAMES.get(k, k), v)
                       for k, v in sorted(stats['by_type'].items()))
        print('  tous types confondus : %s' % ty)
        if stats['n_bad']:
            print('  [!] %d tuples au format inconnu (ignores)'
                  % stats['n_bad'])
        if not samples:
            print('  [!] aucun sample discard -> rien a faire')
        else:
            obs_ch = max(stats['obs_channels'], key=stats['obs_channels'].get)
            if len(stats['obs_channels']) > 1:
                print('  [!] caches mixes : canaux %s'
                      % dict(stats['obs_channels']))
            print('  layout des obs : %d canaux' % obs_ch)
            if obs_ch < N_BASE_CHANNELS:
                print('  -> CACHE PRE-LOOKAHEAD (%d canaux) : le bloc '
                      'lookahead n\'existait pas au parse.\n     Tout '
                      'modele entraîne sur CE cache est aveugle.'
                      % obs_ch)
            else:
                la_cnt = np.array([s[0][LA0:LA1].sum() > 0
                                   for s in samples])
                pct = 100.0 * la_cnt.mean()
                print('  lookahead %d:%d non nul : %.1f %% des samples'
                      % (LA0, LA1, pct))
                if obs_ch >= N_CHANNELS:
                    rd = np.array([s[0][N_BASE_CHANNELS:N_CHANNELS].sum() > 0
                                   for s in samples])
                    print('  rouges %d:%d non nuls   : %.1f %% (sanity)'
                          % (N_BASE_CHANNELS, N_CHANNELS, 100.0 * rd.mean()))
                if pct < 1.0:
                    print('  -> CACHE SANS LOOKAHEAD (0 %%) : le DFS n\'a '
                          'pas tourne au parse.\n     Tout modele entraîne '
                          'sur CE cache est aveugle au lookahead.')
                elif head != 'discard':
                    print('  ablation sautee (head=%s ; labels non 0..33).'
                          ' Les stats de cache ci-dessus restent valables.'
                          % head)
                elif in_ch not in KNOWN_LAYOUTS:
                    print('  ablation sautee : layout checkpoint non reconnu')
                else:
                    model = SuphxModel(
                        'discard', in_channels=in_ch, channels=channels,
                        n_blocks=n_blocks, binary_hidden=binary_hidden)
                    missing, unexpected = model.load_state_dict(
                        sd, strict=False)
                    if missing:
                        print('  [!] %d cles manquantes -> ablation sautee'
                              % len(missing))
                    else:
                        a_info = ablation(model, samples, in_ch, device,
                                          args.batch_size)
                        if a_info is None:
                            print('  [!] obs et checkpoint incompatibles -> '
                                  'ablation sautee')
                        else:
                            print('  ABLATION (%d samples avec lookahead '
                                  'non nul, device=%s) :'
                                  % (a_info['n_la'], device))
                            print('    accuracy lookahead INTACT : %.4f'
                                  % a_info['acc_intact'])
                            print('    accuracy lookahead ZERO   : %.4f'
                                  % a_info['acc_zero'])
                            print('    delta = %+.2f pt  (bruit +/-2sigma '
                                  '= %.2f pt)'
                                  % (a_info['delta'] * 100.0,
                                     a_info['noise'] * 100.0))
                            print('    accord argmax : %.1f %%   '
                                  'KL moyen : %.4f'
                                  % (a_info['agreement'] * 100.0,
                                     a_info['kl']))

    # ------------------------- verdict ------------------------------- #
    print()
    print('=' * 68)
    print('VERDICT')
    print('=' * 68)
    verdict = None
    if args.cache_dir is None:
        verdict = ('PARTIEL (pas de --cache-dir) : niveaux 1-2 seulement. '
                   'Ajoutez --cache-dir pour l\'ablation, test decisif.')
    elif a_info is not None:
        d = a_info['delta'] * 100.0
        nz = a_info['noise'] * 100.0
        if d > max(0.3, nz):
            verdict = ('LE MODELE EXPLOITE LE LOOKAHEAD (+%.2f pt sans '
                       'lookahead -> avec).' % d)
        elif d < -max(0.3, nz):
            verdict = ('MODELE AVEUGLE : le lookahead le DEGRADE (%.2f pt) '
                       '-- poids jamais entraînes, le signal passe comme du '
                       'bruit. Entraîne sur des donnees sans lookahead.'
                       % d)
        elif a_info['agreement'] > 99.5 and a_info['kl'] < 1e-3:
            verdict = ('INSENSIBLE : le bloc lookahead ne change pas les '
                       'sorties (poids ~nuls).')
        else:
            verdict = ('AMBIGU : delta = %+.2f pt (bruit %.2f pt). Penchez '
                       'sur "aveugle" si delta <= 0, "exploite faiblement" '
                       'sinon.' % (d, nz))
    else:
        verdict = ('VOIR les stats de cache ci-dessus (ablation non '
                   'calculee : head non-discard, cache pre-lookahead ou '
                   'layouts incompatibles).')
    if w_info and w_info['zero_frac'] > 0.99 and 'EXPLOITE' not in (verdict or ''):
        verdict = (verdict or '') + \
            '\n  (poids lookahead exactement nuls : aveugle par construction)'
    print('  %s' % args.checkpoint)
    if verdict:
        print('  %s' % verdict)
    print()
    print('Rappel RL : lookahead cote env = --compute-lookahead ; il ne sert'
          '\na rien sans un SL entraîne sur un cache lookahead REMPLI.')


if __name__ == '__main__':
    main()
