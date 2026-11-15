"""[V2.4] De quoi est fait r_k ? Instrumentation du flux de récompenses.

CONTEXTE (échec rl_defense2, 50k steps) : PPO dégrade le jeu échantillonné
(mr 2.35 -> 2.67) alors que l'argmax tient (2.53) et que le proxy myopique
recule (ret_mean -0.14 -> -0.87). Hypothèse H1 : sous l'attribution
« plateau par manche » (le delta GRP de la manche est réparti sur TOUTES
les transitions de la manche), la plupart des transitions portent une
récompense ~plate et l'avantage batch_mean devient du bruit — voire un
biais uniforme. Ce diagnostic MESURE la structure au lieu de la deviner.

Protocole = trainer rl_defense2 (champ identique) :
  policy SL figée (init), ÉCHANTILLONNÉE, mode opponent, gamma_t=0,
  GRP + correction terminale, lookahead ON, danger_penalty 2.5,
  n_envs 16, batch_games 32 (1 batch = 1 fenêtre d'update du trainer).

Le runner est INSTRUMENTÉ (jamais modifié) : chaque transition du siège
agent reçoit ses composantes, avec l'invariant vérifié en dur :
    t['reward'] == _crr - _pen + _rnd + _fls + _cor
  _crr  pending reporté (deltas de manches SANS transition du siège ->
        plié dans la prochaine transition ; artefact de comptabilité)
  _pen  pénalité danger (2.5) au moment de la défausse dangereuse
  _rnd  delta GRP de la manche, attribué à la fin de manche à TOUTES les
        transitions du siège dans cette manche (le « plateau »)
  _fls  flush des pendings à la fin de partie (dernière transition)
  _cor  correction terminale RANK_REWARDS[rank] - somme des deltas
        (dernière transition ; rend SOMME(r_k) = reward de rang vrai)

Cibles dérivées (pour diag_critic_learnability v2.4, même cache) :
  ghd   « G_hand » = to-go DANS la manche courante, coupé à la fin de
        CETTE manche, attribution causale par événement :
        ghd_i = _rnd_i - (pénalités de j>=i dans la manche) + _fls + _cor
        (le _crr est exclu : argent d'une manche antérieure)
  ghn   ghd sans la correction terminale.

RAPPORT (la réponse à H1) :
  % de transitions « ordinaires » (aucun événement : pen=0, fls=0, cor=0,
  crr=0) ; décomposition de Var(r_k) entre-mains / intra-main ; avantage
  batch_mean des ordinaires (H1 prédit : ~ -ret_mean > 0, donc > 50% de
  signe positif) ; parmi les transitions pénalisées, % d'avantage > 0
  (si > 50%, la pénalité danger est noyée par le plateau -> dng% monte).

Usage (racine du repo, ~3 min/batch sur RTX 2060 ; 20 batches ~1 h) :
    py -3.14 -m mahjong_ai.diag_reward_structure ^
        --sl-discard checkpoints/sl_tonpusen_la/discard.pt ^
        --sl-frozen checkpoints/sl_tonpusen_la ^
        --grp-checkpoint checkpoints/grp_tonpusen.pt ^
        --compute-lookahead --n-batches 20 ^
        --cache checkpoints/diag_struct_cache.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner
from mahjong_ai.training.game_env import RANK_REWARDS


# ---------------------------------------------------------------------- #
# Runner instrumenté : observe, ne modifie jamais.
# ---------------------------------------------------------------------- #
class InstrumentedRunner(VecRunner):
    """VecRunner + composantes par transition.

    Les hooks capturent l'état AVANT super() (ce que super() va consommer
    ou réinitialiser) puis étiquettent APRÈS super(). Aucune branche du
    flux de récompense n'est dupliquée autrement que pour l'observation.
    """

    def _apply_decision(self, slot, action, ex, collect, danger_penalty):
        if ex is None or not collect:
            return super()._apply_decision(
                slot, action, ex, collect, danger_penalty)
        seat = slot.req['seat']
        carried = float(slot.pending_reward[seat])
        pen = 0.0
        if ex['danger'] is not None and float(ex['danger'][int(action)]) > 0:
            pen = float(danger_penalty)
        super()._apply_decision(slot, action, ex, collect, danger_penalty)
        # la transition vient d'être créée : c'est la dernière de la liste
        if slot.transitions and slot.transitions[-1]['seat'] == seat:
            tr = slot.transitions[-1]
            tr['_crr'] = carried
            tr['_pen'] = pen

    def _after_step(self, slot, result, collect, grp_terminal_correction,
                    return_rho=0.85):
        round_done = bool(result.get('round_done', False))
        pre_idx = pre_pend = None
        if round_done:
            # super() réinitialise round_idx_per_seat et pending_reward
            # après attribution : capturés avant.
            pre_idx = {s: list(slot.round_idx_per_seat.get(s, []))
                       for s in range(4)}
            pre_pend = {s: float(slot.pending_reward[s]) for s in range(4)}
        super()._after_step(slot, result, collect, grp_terminal_correction,
                             return_rho)
        if not round_done:
            return
        rr = result.get('final_rewards', {})
        for s in range(4):
            try:
                r_s = float(rr.get(s, 0.0))
            except (TypeError, ValueError):
                r_s = 0.0
            r_s = max(-20.0, min(20.0, r_s))       # clamp identique à super
            if r_s == 0.0:
                continue
            for idx in pre_idx.get(s, []):
                t = slot.transitions[idx]
                t['_rnd'] = t.get('_rnd', 0.0) + r_s
        if result.get('game_done', False):
            scores = slot.env.game_scores
            order = sorted(range(4), key=lambda p: (-scores[p], p))
            for rank, p in enumerate(order, start=1):
                last = self._last_transition_of(slot, p)
                if last is None:
                    continue
                if pre_pend.get(p, 0.0) != 0.0:
                    last['_fls'] = last.get('_fls', 0.0) + pre_pend[p]
                corr = RANK_REWARDS[rank] - slot.grp_delta_sum[p]
                if abs(corr) >= 1e-9:
                    last['_cor'] = last.get('_cor', 0.0) + corr


# ---------------------------------------------------------------------- #
# Batches : composantes + cibles dérivées (ghd/ghn)
# ---------------------------------------------------------------------- #
def _derive_ghand(trs):
    """ghd/ghn par transition d'UNE partie (liste chronologique).

    ghd_i = _rnd_i - somme(pénalités de j >= i dans la même manche)
            + _fls_i + _cor_i
    Groupes = manches consécutives (même valeur de 'round').
    """
    n = len(trs)
    ghd = np.zeros(n, dtype=np.float64)
    ghn = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and trs[j]['round'] == trs[i]['round']:
            j += 1
        # suffixe inclusif des pénalités sur [i, j), rempli de la fin
        suf = 0.0
        suffix = [0.0] * (j - i)
        for k in range(j - 1, i - 1, -1):
            suf += float(trs[k].get('_pen', 0.0))
            suffix[k - i] = suf
        for k in range(i, j):
            t = trs[k]
            rnd = float(t.get('_rnd', 0.0))
            fls = float(t.get('_fls', 0.0))
            cor = float(t.get('_cor', 0.0))
            ghd[k] = rnd - suffix[k - i] + fls + cor
            ghn[k] = ghd[k] - cor
        i = j
    return ghd, ghn


def batch_from_episodes(eps):
    """Format v2.3 (obs/ret/rew) + composantes + ghd/ghn. Flat."""
    flats = [t for ep in eps for t in ep]
    ghd_parts, ghn_parts = [], []
    for ep in eps:                       # dérivation PAR PARTIE (manches)
        g, h = _derive_ghand(ep)
        ghd_parts.append(g)
        ghn_parts.append(h)
    return {
        'obs_n': np.stack([t['obs_normal'] for t in flats]).astype(
            np.float16),
        'obs_of': np.stack([t['obs_oracle_full'] for t in flats]).astype(
            np.float16),
        'round': np.array([t['round'] for t in flats], dtype=np.int32),
        'ret': np.array([t['return'] for t in flats], dtype=np.float32),
        'rew': np.array([t['reward'] for t in flats], dtype=np.float32),
        'crr': np.array([t.get('_crr', 0.0) for t in flats],
                        dtype=np.float32),
        'pen': np.array([t.get('_pen', 0.0) for t in flats],
                        dtype=np.float32),
        'rnd': np.array([t.get('_rnd', 0.0) for t in flats],
                        dtype=np.float32),
        'fls': np.array([t.get('_fls', 0.0) for t in flats],
                        dtype=np.float32),
        'cor': np.array([t.get('_cor', 0.0) for t in flats],
                        dtype=np.float32),
        'ghd': np.concatenate(ghd_parts).astype(np.float32),
        'ghn': np.concatenate(ghn_parts).astype(np.float32),
    }


COMPONENT_KEYS = ('crr', 'pen', 'rnd', 'fls', 'cor', 'ghd', 'ghn')
# clés scalaires sauvées à part (round = index de main, pour le rapport
# depuis le cache ; les cibles du diag critic n'en ont pas besoin)
ARRAY_KEYS = ('round', 'ret', 'rew') + COMPONENT_KEYS


def save_cache(path, batches):
    arrays = {}
    for i, b in enumerate(batches):
        arrays[f'b{i}_obs_n'] = b['obs_n']
        arrays[f'b{i}_obs_of'] = b['obs_of']
        for k in ARRAY_KEYS:
            arrays[f'b{i}_{k}'] = b[k]
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, **arrays)
    mb = os.path.getsize(path) / 1e6
    print(f"[diag] cache sauvé : {path} ({mb:.0f} Mo, "
          f"{len(batches)} batches)", flush=True)


def load_cache(path):
    z = np.load(path)
    n = sum(1 for k in z.files if k.endswith('_ret'))
    batches = []
    for i in range(n):
        b = {
            'obs_n': np.asarray(z[f'b{i}_obs_n']),
            'obs_of': np.asarray(z[f'b{i}_obs_of']),
            'ret': np.asarray(z[f'b{i}_ret']),
            'rew': np.asarray(z[f'b{i}_rew']),
        }
        for k in ARRAY_KEYS:                # étendu : présent ou non
            if f'b{i}_{k}' in z.files:
                b[k] = np.asarray(z[f'b{i}_{k}'])
        batches.append(b)
    print(f"[diag] cache chargé : {path} ({n} batches, collection sautée)",
          flush=True)
    return batches


# ---------------------------------------------------------------------- #
# Invariant : le flux observé == le flux réel
# ---------------------------------------------------------------------- #
def check_invariant(eps, batch_id):
    """reward == crr - pen + rnd + fls + cor, transition par transition."""
    worst = 0.0
    for ep in eps:
        for t in ep:
            lhs = float(t['reward'])
            rhs = (float(t.get('_crr', 0.0)) - float(t.get('_pen', 0.0))
                   + float(t.get('_rnd', 0.0)) + float(t.get('_fls', 0.0))
                   + float(t.get('_cor', 0.0)))
            worst = max(worst, abs(lhs - rhs))
    if worst > 1e-4:
        raise RuntimeError(
            f"[diag] INVARIANT CASSE (batch {batch_id}, écart max "
            f"{worst:.3g}) : l'instrumentation diverge du flux réel — "
            f"ne pas utiliser ces chiffres, corriger le hook.")
    return worst


# ---------------------------------------------------------------------- #
# Rapport
# ---------------------------------------------------------------------- #
def _hand_slices(round_arr):
    """Indices de DÉBUT de chaque main dans un tableau plat de 'round'.

    Une main = suite maximale de 'round' constant. 'round' croît à chaque
    nouvelle main d'une même partie ; une chute de valeur = nouvelle
    PARTIE. Les deux produisent une frontière de main.
    """
    n = len(round_arr)
    starts = [0]
    for k in range(1, n):
        if round_arr[k] != round_arr[k - 1]:
            starts.append(k)
    return starts, n


def _n_games(round_arr):
    """Nombre de parties dans un batch : 1 + nb de chutes de 'round'."""
    n = len(round_arr)
    return 1 + sum(1 for k in range(1, n) if round_arr[k] < round_arr[k - 1])


def report_batch_line(i, n_batches, b, t0):
    rew = b['rew'].astype(np.float64)
    ghd = b['ghd'].astype(np.float64)
    corr = float(np.corrcoef(ghd, rew)[0, 1]) if len(rew) > 2 else 1.0
    d = np.abs(ghd - rew)
    print(f"  [{i+1:2d}/{n_batches}] {len(rew)} tr "
          f"| r_k: mean={rew.mean():+.2f} var={rew.var():.1f} "
          f"| G(rho) var={b['ret'].var():.1f} "
          f"| ghd~rew corr={corr:.4f} |Δ|moy={d.mean():.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)


def _fmt_component(name, vals, note=''):
    vals = np.asarray(vals, dtype=np.float64)
    nz = float((vals != 0.0).mean())
    return (f"  {name:22s} : mean={vals.mean():+.3f} "
            f"std={vals.std():.2f} |  {nz:5.1%} des transitions"
            f"{('  ' + note) if note else ''}")


def final_report(batches):
    # pool global (tous batches confondus) + découpe par batch
    rew = np.concatenate([b['rew'] for b in batches]).astype(np.float64)
    rnds = np.concatenate([b['rnd'] for b in batches]).astype(np.float64)
    round_all = np.concatenate([b['round'] for b in batches])
    n = len(rew)
    n_games = sum(_n_games(b['round']) for b in batches)

    pens = np.concatenate([b['pen'] for b in batches]).astype(np.float64)
    cors = np.concatenate([b['cor'] for b in batches]).astype(np.float64)
    flss = np.concatenate([b['fls'] for b in batches]).astype(np.float64)
    crrs = np.concatenate([b['crr'] for b in batches]).astype(np.float64)

    starts, _ = _hand_slices(round_all)
    ends = starts[1:] + [n]
    hand_rews = [rew[a:b] for a, b in zip(starts, ends)]
    hand_means = np.array([h.mean() for h in hand_rews])
    sizes = np.array([len(h) for h in hand_rews], dtype=np.float64)

    total_var = float(rew.var())
    grand = float(rew.mean())
    between = float(np.sum(sizes * (hand_means - grand) ** 2) / n)
    within = max(total_var - between, 0.0)

    ghd_all = np.concatenate(
        [b['ghd'] for b in batches]).astype(np.float64)
    corr_eq = float(np.corrcoef(ghd_all, rew)[0, 1])
    deq = np.abs(ghd_all - rew)

    # avantage batch_mean, fenêtre = 1 batch (comme l'update du trainer)
    adv_nev, adv_pen = [], []
    for b in batches:
        r = b['rew'].astype(np.float64)
        m = float(r.mean())
        ord_mask = ((b['pen'] == 0) & (b['fls'] == 0) & (b['cor'] == 0)
                    & (b['crr'] == 0))
        pen_mask = b['pen'] > 0
        a = r - m
        adv_nev.append(a[ord_mask])
        adv_pen.append(a[pen_mask])
    adv_nev = np.concatenate(adv_nev) if adv_nev else np.array([])
    adv_pen = np.concatenate(adv_pen) if adv_pen else np.array([])
    n_ord = len(adv_nev)

    print()
    print("=" * 70)
    print(f"[STRUCTURE] {len(batches)} batches | {n_games} parties "
          f"| {n} transitions | {len(hand_rews)} mains "
          f"({n / max(len(hand_rews), 1):.1f} tr/main, "
          f"{len(hand_rews) / max(n_games, 1):.2f} mains/partie)")
    print(f"[STRUCTURE] r_k : mean={grand:+.3f} std={rew.std():.2f} "
          f"var={total_var:.1f} médiane={np.median(rew):+.3f} | "
          f"|r_k|<0.05 : {(np.abs(rew) < 0.05).mean():.1%} | "
          f"r_k>0 : {(rew > 0).mean():.1%}")
    print("[STRUCTURE] composantes (par transition) :")
    print(_fmt_component('rnd (delta de main)', rnds,
                         'le plateau, attribué en fin de main'))
    print(_fmt_component('pen (danger, -2.5)', pens,
                         'seule composante PAR transition (contrib = -pen)'))
    print(_fmt_component('cor (correction rang)', cors,
                         'une par partie, dernière transition'))
    print(_fmt_component('fls (flush pending)', flss))
    print(_fmt_component('crr (pending reporté)', crrs))
    print("[STRUCTURE] décomposition de Var(r_k) :")
    print(f"  entre-mains : {between:6.2f}  ({between / max(total_var, 1e-9):5.1%})"
          f"   <- varie AVEC la main, pas avec la transition")
    print(f"  intra-main  : {within:6.2f}  ({within / max(total_var, 1e-9):5.1%})"
          f"   <- ce que la policy peut influencer transition par transition")
    print(f"[STRUCTURE] transitions ordinaires (pen=fls=cor=crr=0) : "
          f"{n_ord / n:5.1%} des transitions")
    if n_ord:
        print(f"  leur avantage batch_mean : mean={adv_nev.mean():+.3f} | "
              f">0 : {(adv_nev > 0).mean():.1%}"
              f"   [H1 prédit ~ -ret_mean > 0 et > 50%]")
    if len(adv_pen):
        print(f"[STRUCTURE] transitions pénalisées (n={len(adv_pen)}) : "
              f"avantage mean={adv_pen.mean():+.3f} | "
              f">0 : {(adv_pen > 0).mean():.1%}"
              f"   [> 50% = pénalité noyée par le plateau]")
    print(f"[STRUCTURE] équivalence ghand ≡ r_k : corr={corr_eq:.4f}, "
          f"|Δ| moyen={deq.mean():.3f}, "
          f"|Δ|>0.01 : {(deq > 0.01).mean():.1%}")

    # ---------------- verdict ---------------- #
    pct_ord = n_ord / max(n, 1)
    share_within = within / max(total_var, 1e-9)
    nev_pos = float((adv_nev > 0).mean()) if n_ord else 0.0
    pen_pos = float((adv_pen > 0).mean()) if len(adv_pen) else 0.0
    print()
    print("=" * 70)
    print("[VERDICT STRUCTURE] (à croiser avec l'éval A, pas seul)")
    print(f"  [1] ordinaires : {pct_ord:.1%} des transitions "
          f"{'-> > 80% : prémisse H1 OK' if pct_ord > 0.8 else ''}")
    print(f"  [2] part intra-main de Var(r_k) : {share_within:.1%} "
          f"{'-> < 10% : avantage plat par main, aucun crédit intra-main'
             if share_within < 0.10 else ''}")
    print(f"  [3] avantage des ordinaires > 0 : {nev_pos:.1%} "
          f"{'-> > 50% : biais positif quasi universel (H1)'
             if nev_pos > 0.5 else ''}")
    print(f"  [4] pénalisées avec avantage > 0 : {pen_pos:.1%} "
          f"{'-> > 50% : la pénalité danger est NOYÉE (dng% montera)'
             if pen_pos > 0.5 else ''}")
    print("  Rapporter ce bloc + le tableau du diag critic (C) avant "
          "toute relance.")
    print("=" * 70)


# ---------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.
                                RawDescriptionHelpFormatter)
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--grp-checkpoint', default=None)
    p.add_argument('--compute-lookahead', action='store_true')
    p.add_argument('--n-envs', type=int, default=16)
    p.add_argument('--batch-games', type=int, default=32)
    p.add_argument('--n-batches', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--danger-penalty', type=float, default=2.5,
                   help='doit rester égal au --danger-penalty du trainer '
                        '(flux de production)')
    p.add_argument('--return-rho', type=float, default=0.85,
                   help='rho du G_t stocké (champ ret, continuité v2.3 ; '
                        'r_k/ghd n\'en dépendent pas)')
    p.add_argument('--cache', default='checkpoints/diag_struct_cache.npz',
                   help='npz de cache. Si le fichier existe : chargé tel '
                        'quel, la collecte est sautée (le rapport utilise '
                        'alors les composantes du cache).')
    return p


def main():
    args = build_parser().parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[diag] GPU : {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_num_threads(args.torch_threads)

    if os.path.isfile(args.cache):
        batches = load_cache(args.cache)
        # rapport depuis le cache (les 'round' y sont stockés) ; l'invariant
        # a été vérifié à la collecte
        final_report(batches)
        return

    agent = PPOAgent(args.sl_discard, args.sl_frozen, device=device)
    print(f"[diag] protocole : {args.n_batches} batches x "
          f"{args.batch_games} parties, policy SL figée échantillonnée, "
          f"gamma_t=0, mode=opponent, danger_penalty={args.danger_penalty}, "
          f"rho(G)={args.return_rho}", flush=True)
    runner = InstrumentedRunner(
        agent, n_envs=args.n_envs, seed=args.seed,
        grp_path=args.grp_checkpoint,
        compute_lookahead=args.compute_lookahead, danger_labels=True,
        mode='opponent', device=device)

    batches = []
    t0 = time.time()
    for i in range(args.n_batches):
        eps, _ = runner.collect_batch(
            args.batch_games, gamma_t=0.0,
            danger_penalty=args.danger_penalty,
            return_rho=args.return_rho)
        worst = check_invariant(eps, i)
        b = batch_from_episodes(eps)
        batches.append(b)
        report_batch_line(i, args.n_batches, b, t0)
        if worst > 1e-6:
            print(f"         (invariant max |Δ|={worst:.2g} : OK mais "
                  f"non nul — flottements fp, sans importance)")

    save_cache(args.cache, batches)
    final_report(batches)


if __name__ == '__main__':
    main()
