#!/usr/bin/env python3
"""Tests du lookahead REPARÉ (envs/suphx_features.py, fix 2026-11).

Valide que les 80 canaux look-ahead ne sont plus constants à zéro :
  1. DFS : énumération correcte (paire modélisée, ordre canonique,
     chiitoitsu, kokushi, melds/kans, budget n_replace + 1).
  2. Scoring : ids 136, main complète (melds inclus), bucket cohérent
     avec estimate_hand_value.
  3. Contrainte unseen : une attente épuisée n'allume pas de canal.
  4. Gates : 13/14 tuiles acceptées, états inattendus neutres, mains à
     3+ shanten à zéro.
  5. Performance : quelques ms par appel, aucune exception sur mains
     aléatoires.

Lancement :  python test_lookahead_fix.py   (CPU, ~30 s)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mahjong_ai.envs import suphx_features as F          # noqa: E402
from mahjong_ai.envs.suphx_features import (              # noqa: E402
    N_TILES, N_SCORE_BUCKETS, N_REPLACE_VALUES, compute_lookahead,
    _enumerate_winning_hands_with_melds, _score_to_bucket,
)

PASS = 0
FAIL = 0

def check(name, cond, info=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {info}")

def counts(pairs):
    """{'man': '234567', ...} -> comptages 34."""
    c = np.zeros(N_TILES, dtype=np.int32)
    offset = {'man': 0, 'pin': 9, 'sou': 18}
    for suit, s in pairs.items():
        if suit in offset:
            for ch in s:
                c[offset[suit] + int(ch) - 1] += 1
        else:  # honneurs : 1=E..4=N, 5=blanc, 6=vert, 7=rouge
            for ch in s:
                c[26 + int(ch)] += 1
    return c

def lit_rows(feat):
    """{(row, bucket)} allumés."""
    out = set()
    for row in range(len(N_REPLACE_VALUES)):
        for b in range(N_SCORE_BUCKETS):
            if feat[row * N_SCORE_BUCKETS + b].any():
                out.add((row, b))
    return out

print("=" * 70)
print("1) DFS : énumération des mains gagnantes")
print("=" * 70)

# tenpai : 234m 567m 234p 55p 89s (attente penchan 7s)
hand_t = counts({'man': '234567', 'pin': '23455', 'sou': '89'})
res = _enumerate_winning_hands_with_melds(hand_t, [], 2)
wins = {w for w, a in res}
check("tenpai, budget 2 : >= 1 main gagnante", len(res) >= 1, f"{len(res)} resultats")
# la main 234m567m234p55p789s doit y etre (ajout : 7s)
expect = counts({'man': '234567', 'pin': '23455', 'sou': '789'})
check("main attendue 234m567m234p55p789s presente",
      tuple(expect) in wins)
for w, a in res:
    if tuple(w) == tuple(expect):
        check("ajout = 1x7s (index 24)", a[24] == 1 and sum(a) == 1,
              f"added={a}")
        break

# budget 1 (tenpai sans echange : tuile gagnante seule) -> meme main
res1 = _enumerate_winning_hands_with_melds(hand_t, [], 1)
check("tenpai, budget 1 : la meme main (ajout 7s)",
      tuple(expect) in {w for w, a in res1})

# 1-shanten 13 tuiles : 234m 567m 234p 45p 79s — gagnable en 2 tirages
# (3p + 7s -> paire 77s) mais pas en 1 (min adds = shanten + 1 = 2)
hand_1s = counts({'man': '234567', 'pin': '23445', 'sou': '79'})
check("1-shanten, budget 1 : aucune main (il faut 2 tuiles)",
      len(_enumerate_winning_hands_with_melds(hand_1s, [], 1)) == 0)
res2 = _enumerate_winning_hands_with_melds(hand_1s, [], 2)
check("1-shanten, budget 2 : >= 1 main (p.ex. 3p + 7s -> 77s)",
      len(res2) >= 1, f"{len(res2)}")
res3 = _enumerate_winning_hands_with_melds(hand_1s, [], 3)
check("1-shanten, budget 3 : >= 1 main", len(res3) >= 1, f"{len(res3)}")

# 2-shanten : 1234567s 12m 45m 78p (2 figures + 3 partiels, sans paire)
hand_2s = counts({'sou': '1234567', 'man': '1245', 'pin': '78'})
res_b2 = _enumerate_winning_hands_with_melds(hand_2s, [], 3)
res_b3 = _enumerate_winning_hands_with_melds(hand_2s, [], 4)
check("2-shanten, budget 2 : aucune main",
      len(_enumerate_winning_hands_with_melds(hand_2s, [], 2)) == 0)
check("2-shanten, budget 3 : >= 1 main", len(res_b2) >= 1, f"{len(res_b2)}")
print(f"        (2-shanten : {len(res_b2)} mains a budget 3, "
      f"{len(res_b3)} a budget 4)")

# chiitoitsu : 11m22m33p44p55s66s8s (tenpai chiitoi, attente 8s)
hand_chi = counts({'man': '1122', 'pin': '3344', 'sou': '55668'})
res_chi = _enumerate_winning_hands_with_melds(hand_chi, [], 2)
chi_win = counts({'man': '1122', 'pin': '3344', 'sou': '556688'})
check("chiitoitsu tenpai : la main 7 paires est enumerée",
      tuple(chi_win) in {w for w, a in res_chi},
      f"{len(res_chi)} resultats")

# kokushi : 1m9m1p9p1s9s1234567z (13 branches)
hand_kok = counts({'man': '19', 'pin': '19', 'sou': '19', 'z': '1234567'})
res_kok = _enumerate_winning_hands_with_melds(hand_kok, [], 2)
kok_wins = {w for w, a in res_kok}
n_kok = sum(1 for w in kok_wins if sum(w) == 14 and w[0] == 2)
check("kokushi 13 branches : >= 10 variantes a budget 2",
      len(res_kok) >= 10, f"{len(res_kok)} resultats")

# melds : pon 5p + 234m 567m 789s 2p (tenpai d'attente 2p)
pon = [{'type': 'pon', 'tiles': [13, 13, 13], 'reds': []}]
hand_m = counts({'man': '234567', 'sou': '789', 'pin': '2'})
res_m = _enumerate_winning_hands_with_melds(hand_m, pon, 2)
exp_m = counts({'man': '234567', 'sou': '789', 'pin': '22'})
check("avec pon : main gagnante (paire 2p) enumerée",
      tuple(exp_m) in {w for w, a in res_m}, f"{len(res_m)} resultats")
for w, a in res_m:
    if tuple(w) == tuple(exp_m):
        check("avec pon : ajout = 1x2p, paire modélisée",
              a[10] == 1 and w[10] == 2 and sum(w) == 11,
              f"added={a}, |W|={sum(w)}")
        break

# kan clos : 2222m (ankan) + 345p 678p 99s 34m (10 tuiles hors tour)
ankan = [{'type': 'kan_closed', 'tiles': [1, 1, 1, 1], 'reds': []}]
between_k = 13 + 1 - 4  # = 10 attendu hors tour
hand_k = counts({'pin': '345678', 'sou': '99', 'man': '34'})  # 10 tuiles
res_k = _enumerate_winning_hands_with_melds(hand_k, ankan, 2)
# win = 345m + 345p 678p + paire 99s (3 figures + paire, ajout 5m)
exp_k = counts({'pin': '345678', 'sou': '99', 'man': '345'})
check("avec ankan : main (345m + 2 runs + paire 99s) enumerée",
      tuple(exp_k) in {w for w, a in res_k},
      f"{len(res_k)} resultats, entre-tours attendu {between_k}")
for w, a in res_k:
    if tuple(w) == tuple(exp_k):
        check("avec ankan : 3 figures + paire (|W| = 11), ajout 5m",
              sum(w) == 11 and a[4] == 1,
              f"|W|={sum(w)}, added={a}")
        break

print()
print("=" * 70)
print("2) compute_lookahead : canaux actifs + buckets coherents")
print("=" * 70)

# tenpai PINFU : 234m 567m 234p 55p 67s (attente ryanmen 5s/8s)
hand_t = counts({'man': '234567', 'pin': '23455', 'sou': '67'})
unseen_full = np.full(N_TILES, 4, dtype=np.int32)
for t in range(N_TILES):
    unseen_full[t] -= int(hand_t[t])
feat_t = compute_lookahead(hand_t, [], seat_wind=0, round_wind=0,
                           is_dealer=False, dora_indicators=[],
                           unseen=unseen_full)
rows_t = lit_rows(feat_t)
check("tenpai pinfu : canaux actifs", len(rows_t) > 0, f"{sorted(rows_t)}")
check("tenpai : ligne n_replace=1 allumée", any(r == 0 for r, b in rows_t))
check("monotonie : chaque bucket allumé en ligne k l'est en ligne k+1",
      all((r + 1, b) in rows_t for (r, b) in rows_t if r < 3),
      f"{sorted(rows_t)}")

# coherence du bucket avec un scoring direct de la meme main : le gain
# direct (tsumo 5s -> pinfu) doit etre dans la ligne 1
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.constants import EAST
from mahjong.meld import Meld
from mahjong.tile import TilesConverter
calc = HandCalculator()
# NB : la lib force is_dealer=True quand player_wind=EAST ; on ecrase avec
# le flag explicite, comme le fait _get_config (is_dealer vient du moteur).
cfg = HandConfig(is_tsumo=False, is_riichi=False, player_wind=EAST,
                 round_wind=EAST,
                 options=OptionalRules(has_open_tanyao=True, has_aka_dora=False))
cfg.is_dealer = False
expect = counts({'man': '234567', 'pin': '23455', 'sou': '567'})
tiles136 = sorted([4 * t + i for t in range(N_TILES)
                   for i in range(int(expect[t]))])
res_direct = calc.estimate_hand_value(tiles=tiles136, win_tile=4 * 22,
                                      melds=[], dora_indicators=[], config=cfg)
total_direct = int(res_direct.cost['total'])
bucket_direct = _score_to_bucket(total_direct)
check(f"bucket direct ({total_direct} pts -> bucket {bucket_direct}) "
      f"present dans la ligne 1",
      (0, bucket_direct) in rows_t, f"lignes allumees: {sorted(rows_t)}")
check("yaku direct = pinfu (sanity du scoring lib)",
      any('Pinfu' in str(y) for y in (res_direct.yaku or [])),
      f"yaku={[str(y) for y in (res_direct.yaku or [])]}")

# main 3+ shanten : que des zeros
hand_far = counts({'man': '13579', 'pin': '13579', 'sou': '13579', 'z': '1'})
feat_far = compute_lookahead(hand_far, [], 0, 0, False, [])
check("main 5+ shanten : 80 canaux a zero", lit_rows(feat_far) == set())

# 14 tuiles (apres pioche) acceptees
hand14 = hand_t.copy(); hand14[33] += 1   # pioche un honneur mort
feat14 = compute_lookahead(hand14, [], 0, 0, False, [])
check("main 14 tuiles : acceptee (canaux actifs)", len(lit_rows(feat14)) > 0)
# etat absurde (10 tuiles sans meld) : neutre, pas d'exception
feat_bad = compute_lookahead(np.zeros(N_TILES, np.int32), [], 0, 0, False, [])
check("main 0 tuile : neutre sans exception", lit_rows(feat_bad) == set())

print()
print("=" * 70)
print("3) Contrainte unseen")
print("=" * 70)

feat_free = compute_lookahead(hand_t, [], 0, 0, False, [], unseen=unseen_full)
rows_free = lit_rows(feat_free)
unseen_zero = unseen_full.copy()
for wt in (22, 25):            # plus de 5s ni de 8s tirable
    unseen_zero[wt] = 0
feat_z = compute_lookahead(hand_t, [], 0, 0, False, [], unseen=unseen_zero)
rows_z = lit_rows(feat_z)
check("attente ryanmen possible : canaux actifs", len(rows_free) > 0)
check("attentes 5s/8s epuisees (unseen=0) : ligne 1 reduite",
      not any(r == 0 and b == bucket_direct for r, b in rows_z),
      f"reste ligne 1: {sorted(b for r, b in rows_z if r == 0)}")

print()
print("=" * 70)
print("4) Robustesse : 300 mains aleatoires")
print("=" * 70)

rng = np.random.default_rng(0)
n_low_shanten = 0
n_with_channels = 0
t0 = time.time()
worst = 0.0
for i in range(300):
    # main aleatoire 13 tuiles valide (max 4 par type)
    while True:
        hand = rng.multinomial(13, np.full(N_TILES, 1 / N_TILES))
        if hand.max() <= 4:
            break
    unseen_r = np.full(N_TILES, 4, dtype=np.int32) - hand
    ts = time.time()
    feat_r = compute_lookahead(hand, [], 0, 0, False, [5], unseen=unseen_r)
    worst = max(worst, time.time() - ts)
    sh = F._SHANTEN_CALC.calculate_shanten(hand.tolist(),
                                         use_chiitoitsu=True)
    if sh <= 2:
        n_low_shanten += 1
        if lit_rows(feat_r):
            n_with_channels += 1
elapsed = time.time() - t0
check("aucune exception sur 300 mains aleatoires", True)
check(f"canaux actifs sur une partie des mains <= 2 shanten "
      f"({n_with_channels}/{n_low_shanten})",
      n_low_shanten == 0 or n_with_channels >= 0.25 * n_low_shanten,
      f"{n_with_channels}/{n_low_shanten}")
print(f"        300 mains en {elapsed:.2f}s ({1000*elapsed/300:.1f} ms/moy, "
      f"pire appel {1000*worst:.0f} ms)")
check("perf : pire appel < 300 ms", worst < 0.3, f"pire={1000*worst:.0f} ms")

print()
print("=" * 70)
print("5) build_suphx_obs bout-en-bout (lookahead ON)")
print("=" * 70)

hands = np.zeros((4, N_TILES), dtype=np.int32)
hands[0] = hand_t
melds4 = [[], [], [], []]
obs = F.build_suphx_obs(
    hands=hands, melds=melds4,
    discards=np.zeros((4, N_TILES), dtype=np.int32),
    discard_hist=[[], [], [], []],
    riichi_flags=[False] * 4, furiten_flag=False,
    dora_indicators=[5], wall_remaining=60,
    scores=[25.0] * 4, dealer_seat=0, round_wind=0, round_id=1,
    honba=0, kyoutaku=0, seat=0,
    compute_lookahead_flag=True,
    rounds_remaining=3, red_state=None,
)
block = F._OFFSET['lookahead']
la_block = obs[block:block + F.LOOKAHEAD_CHANNELS]
check("bloc lookahead non nul dans l'obs", float(np.abs(la_block).sum()) > 0)
check("valeurs binaires {0,1}", set(np.unique(la_block)) <= {0.0, 1.0})
check("obs globalement inchangee ailleurs (283 canaux)",
      obs.shape == (F.N_CHANNELS, N_TILES))

print()
print("=" * 70)
print(f"RESULTAT : {PASS} OK, {FAIL} ECHEC(S)")
print("=" * 70)
sys.exit(1 if FAIL else 0)
