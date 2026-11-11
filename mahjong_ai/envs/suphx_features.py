"""
Features d'entree Suphx (papier §2.2). Module partage parser <-> env.
Layout : (N_CHANNELS, 34).
Look-ahead : 80 canaux (4 n_replace x 20 score_buckets), valeurs booleennes,
melds pris en compte dans le DFS et le scoring.

[FIX LOOKAHEAD 2026-11] Le DFS look-ahead d'origine etait mort (3 bugs :
UnboundLocalError avale, paire non modelisee, comptages 34 passes a une
lib qui attend des ids 136) => les 80 canaux etaient constants a zero en
SL comme en RL. Repare + teste (test_lookahead_fix.py, 31 assertions).
DETAILS IMPORTANTS pour le re-entrainement : les checkpoints SL existants
ont ete entraînes avec ces canaux a zero — activer le look-ahead (flag
compute_lookahead / --compute-lookahead) change la distribution d'entree
sur 80 des 283 canaux : a reserver a un SL re-entraine (ou fine-tune)
avec le look-ahead actif. Le RL reste a False par defaut => les
checkpoints actuels marchent tels quels.

[MODIF 2026-10-13] Suppression game_format et round_progress des canaux.
rounds_remaining devient ABSOLU (0-7, one-hot) pour permettre le transfert
hanchan/tonpusen : E4-tonpusen et S4-hanchan partagent le bucket 0.

[OPTIM 2025-XX] Separation en 2 caches distincts :
- _enumerate_hands_cached : DFS pur, depend de (H_prime, melds, max_add)
- _score_with_dora_cached : scoring, depend de (T, melds, sw, rw, dl, dora)
La meme main gagnante T peut etre atteinte depuis plusieurs tuiles
discardables differentes -> le cache scoring a un hit rate eleve (>60%
observe). Ajout d'un cache HandConfig par (seat_wind, round_wind,
is_dealer). Gain mesure : ~3-5x sur le look-ahead, aucun changement de
signal.
"""
import functools
import json
import numpy as np
from mahjong.constants import EAST, SOUTH, WEST, NORTH
from mahjong.shanten import Shanten
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld

N_TILES = 34
SEAT_WINDS = [EAST, SOUTH, WEST, NORTH]

_MELD_TYPE_MAP = {
    'chi':        Meld.CHI,
    'pon':        Meld.PON,
    'kan_closed': Meld.KAN,
    'kan_open':   Meld.KAN,
    'shouminkan': Meld.SHOUMINKAN,
}

# ====================================================================== #
# Layout des canaux  [MODIFIE]
# ====================================================================== #

# --- Rouges (aka dora) : un 5 rouge par famille (0m, 0p, 0s) ---------- #
RED_TYPES = (4, 13, 22)                    # types 34 : 5m, 5p, 5s
RED_INDEX = {t: k for k, t in enumerate(RED_TYPES)}
N_RED = len(RED_TYPES)

N_REPLACE_VALUES = (1, 2, 3, 4)
N_SCORE_BUCKETS = 20
LOOKAHEAD_CHANNELS = len(N_REPLACE_VALUES) * N_SCORE_BUCKETS   # 80
MAX_DFS_RESULTS = 30
# [FIX LOOKAHEAD] borne sur le nombre de mains scorees par etat (les moins
# couteuses d'abord) : le scoring (estimate_hand_value) domine le cout.
_LOOKAHEAD_SCORE_CAP = 40
MAX_LOOKAHEAD_SHANTEN = 2

_BASE_GROUPS = [
    ('own_hand', 4),
    ('own_meld_open', 4), ('own_meld_ankan', 4),
    ('discards_s0', 4), ('discards_s1', 4), ('discards_s2', 4), ('discards_s3', 4),
    ('last_discard_s0', 1), ('last_discard_s1', 1), ('last_discard_s2', 1), ('last_discard_s3', 1),
    ('tsumogiri_s0', 1), ('tsumogiri_s1', 1), ('tsumogiri_s2', 1), ('tsumogiri_s3', 1),
    ('opp0_meld', 4), ('opp1_meld', 4), ('opp2_meld', 4),
    ('dora_ind', 1), ('dora_tile', 1), ('pending', 1), ('unseen', 1),
    ('own_riichi', 1), ('opp_riichi', 3), ('own_furiten', 1),
    ('seat_wind', 4), ('round_wind', 4), ('dealer', 1), ('round_id', 4),
    # [SUPPRIME] ('game_format', 2),
    # [SUPPRIME] ('round_progress', 10),
    ('is_last_normal_round', 1), ('is_extension', 1),
    ('extension_index', 8),
    ('rounds_remaining', 8),   # [MODIFIE] absolu 0-7, one-hot
    ('honba', 5), ('kyoutaku', 5),
    ('own_score', 20), ('opp0_score', 20), ('opp1_score', 20), ('opp2_score', 20),
    ('wall_remaining', 10),
    ('lookahead', LOOKAHEAD_CHANNELS),
]

# [ROUGES] Canaux ajoutes EN FIN de layout : les indices 0..N_BASE_CHANNELS-1
# restent ceux des anciens checkpoints (les nouveaux poids d'entree sont
# initialises a 0 au chargement, cf. agents/suphx_cnn.adapt_in_channels).
# Chaque canal met 1.0 sur la colonne (type 34) du 5 concerne.
_RED_GROUPS = [
    ('red_hand', 1),          # mon 5 rouge est dans ma main
    ('red_meld_own', 1),      # mon 5 rouge est dans un de mes melds
    ('red_river_s0', 1), ('red_river_s1', 1),
    ('red_river_s2', 1), ('red_river_s3', 1),   # 5 rouge dans la riviere du siege
    ('red_meld_opp0', 1), ('red_meld_opp1', 1), ('red_meld_opp2', 1),
    ('red_unseen', 1),        # rouge pas encore visible pour moi
    ('pending_red', 1),       # la tuile en attente d'appel/ron est rouge
]

CHANNEL_GROUPS = _BASE_GROUPS + _RED_GROUPS

N_BASE_CHANNELS = sum(n for _, n in _BASE_GROUPS)       # 268 (avant rouges)
N_RED_CHANNELS = sum(n for _, n in _RED_GROUPS)         # 11
N_CHANNELS = sum(n for _, n in CHANNEL_GROUPS)
OBS_FLAT_DIM = N_CHANNELS * N_TILES

_OFFSET = {}
_o = 0
for _name, _n in CHANNEL_GROUPS:
    _OFFSET[_name] = _o
    _o += _n

# ====================================================================== #
# HandConfig cache (immutable, safe to reuse)
# ====================================================================== #

_CONFIG_CACHE = {}
_OPTIONS = OptionalRules(has_open_tanyao=True, has_aka_dora=False)

def _get_config(seat_wind, round_wind, is_dealer=False):
    # [FIX LOOKAHEAD] is_dealer entre dans la cle de cache : l'ancienne
    # version mutait l'objet cache, partage entre sieges dealer/non-dealer.
    key = (seat_wind, round_wind, bool(is_dealer))
    cfg = _CONFIG_CACHE.get(key)
    if cfg is None:
        cfg = HandConfig(
            is_tsumo=False, is_riichi=False,
            player_wind=SEAT_WINDS[seat_wind],
            round_wind=SEAT_WINDS[round_wind],
            options=_OPTIONS,
        )
        cfg.is_dealer = bool(is_dealer)
        _CONFIG_CACHE[key] = cfg
    return cfg

# ====================================================================== #
# Look-ahead : DFS + scoring
# ====================================================================== #
# [FIX LOOKAHEAD 2026-11] L'implementation d'origine ne produisait JAMAIS
# de canal actif (les 80 canaux etaient constants a zero, en SL comme en
# RL — cf. ANALYSE_RL.md P8) :
#   1. UnboundLocalError dans le DFS : 'remaining' etait rendu local par
#      l'affectation en fin de dfs() puis lu en tete de boucle ->
#      exception remontee jusqu'a compute_lookahead et avalee par un
#      'except Exception: continue' => 0 main enumeree.
#   2. La paire n'etait pas modelisee (_MELD_OPTIONS = triplets + runs
#      seulement) : la condition d'emission sum(remaining)==0 etait
#      structurellement inatteignable (il reste toujours au moins la
#      tuile de la paire) => 0 resultat meme sans le bug 1.
#   3. Le scoring passait des COMPTAGES 34 a estimate_hand_value, qui
#      attend des ids 136 (tiles ET melds) et la main COMPLETE (tuiles
#      des melds incluses).
#   4. _get_config mutait is_dealer sur un objet cache partage entre
#      sieges (fuite d'etat dealer / non-dealer).
#
# Semantique (papier Suphx §2.2) : le canal (n_replace, bucket) = 1 s'il
# existe une main gagnante atteignable en tirant au plus n_replace + 1
# tuiles du mur (n_replace echanges + la tuile gagnante). Tenpai ->
# n_replace = 1 ; 1-shanten -> n_replace = 1 ; 2-shanten -> n_replace
# >= 2 (d'ou le gate shanten <= MAX_LOOKAHEAD_SHANTEN = 2, les mains a
# 3+ shanten restent a zero par choix d'origine).
# Kans : comptes pour une figure de 3 tuiles (la 4e est hors structure,
# cf. _shanten_counts du moteur). Chiitoitsu et kokushi enumeres quand
# aucune figure n'est declaree. Chaque tuile tiree doit rester presente
# dans le mur vu du siege (contrainte unseen, si fournie).
# Config de scoring inchangee (ron sans riichi, semantique d'origine) :
# une main sans yaku intrinseque (ex. suite pure sans pinfu/tanyao/
# yakuhai) n'allume aucun canal.

_SCORE_UPPER = [
    0, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000,
    10000, 12000, 16000, 20000, 24000, 32000, 48000, 64000, 96000, 10**9,
]

_MELD_OPTIONS = []
for _t in range(N_TILES):
    _MELD_OPTIONS.append({_t: 3})
for _suit in range(3):
    for _start in range(7):
        _t = _suit * 9 + _start
        _MELD_OPTIONS.append({_t: 1, _t + 1: 1, _t + 2: 1})

_KOKUSHI_TYPES = tuple([0, 8, 9, 17, 18, 26] + list(range(27, N_TILES)))

_HAND_CALC = HandCalculator()
# [FIX LOOKAHEAD] une seule instance de Shanten (etait reinstanciee a
# chaque appel de compute_lookahead).
_SHANTEN_CALC = Shanten()

def _score_to_bucket(score):
    if score is None or score <= 0:
        return 0
    for i, upper in enumerate(_SCORE_UPPER):
        if score <= upper:
            return max(i, 1)
    return N_SCORE_BUCKETS - 1

def _melds_from_key(melds_key):
    if not melds_key:
        return []
    return [json.loads(s) for s in melds_key]

def _enumerate_winning_hands_with_melds(hand13, existing_melds, max_add,
                                        max_results=MAX_DFS_RESULTS):
    """Enumere les mains gagnantes atteignables depuis hand13 (comptages
    34) en tirant au plus max_add tuiles du mur.

    Main gagnante = (4 - len(existing_melds)) figures + 1 paire, ou
    7 paires distinctes / kokushi (si aucune figure declaree). Les kans
    declares comptent pour une figure de 3 (leur 4e tuile est hors
    structure). Rend une liste de (win_counts, added_counts) :
      win_counts   : comptages 34 de la partie cachee gagnante (figures
                     + paire, tuiles tirees incluses, melds exclus)
      added_counts : comptages 34 des tuiles tirees du mur
    Dedupliquee par win_counts, bornee a max_results.
    """
    n_melds = len(existing_melds)
    n_sets = 4 - n_melds
    hand13 = np.asarray(hand13, dtype=np.int32)
    results = []
    seen = set()
    state = {'count': 0, 'visited': 0}
    MAX_VISITED = 4000
    zero = np.zeros(N_TILES, dtype=np.int32)

    def emit(win, added):
        key = tuple(int(x) for x in win)
        if key in seen:
            return
        seen.add(key)
        results.append((key, tuple(int(x) for x in added)))
        state['count'] += 1

    # --- formes standards : n_sets figures + 1 paire ------------------- #
    # Les figures sont choisies en ordre canonique (index croissant dans
    # _MELD_OPTIONS) pour eviter d'explorer les permutations.
    def sets_dfs(remaining, sets_left, adds_left, used, added, min_idx):
        if state['count'] >= max_results or state['visited'] >= MAX_VISITED:
            return
        state['visited'] += 1
        if adds_left < 0:
            return
        # borne inferieure du cout restant : il faut 3*sets_left + 2
        # tuiles, dont au plus sum(remaining) peuvent venir de la main.
        if 3 * sets_left + 2 - int(remaining.sum()) > adds_left:
            return
        if sets_left == 0:
            for t in range(N_TILES):
                cost = max(0, 2 - int(remaining[t]))
                if cost > adds_left:
                    continue
                win = used.copy(); win[t] += 2
                ad = added.copy(); ad[t] += cost
                emit(win, ad)
                if state['count'] >= max_results:
                    return
            return
        for idx in range(min_idx, len(_MELD_OPTIONS)):
            meld_counts = _MELD_OPTIONS[idx]
            cost = 0
            affordable = True
            for t, need in meld_counts.items():
                miss = need - int(remaining[t])
                if miss > 0:
                    cost += miss
                    if cost > adds_left:
                        affordable = False
                        break
            if not affordable:
                continue
            new_remaining = remaining.copy()
            new_used = used.copy()
            new_added = added.copy()
            for t, need in meld_counts.items():
                take = min(need, int(remaining[t]))
                new_remaining[t] -= take
                new_used[t] += need
                new_added[t] += need - take
            sets_dfs(new_remaining, sets_left - 1, adds_left - cost,
                     new_used, new_added, idx)

    sets_dfs(hand13.copy(), n_sets, int(max_add), zero.copy(),
             zero.copy(), 0)

    if n_melds == 0:
        # --- chiitoitsu : 7 paires distinctes -------------------------- #
        def pairs_dfs(start, remaining, pairs_left, adds_left, used, added):
            if state['count'] >= max_results or state['visited'] >= MAX_VISITED:
                return
            state['visited'] += 1
            if pairs_left == 0:
                emit(used, added)
                return
            if N_TILES - start < pairs_left:
                return
            if 2 * pairs_left - int(remaining[start:].sum()) > adds_left:
                return
            for t in range(start, N_TILES):
                cost = max(0, 2 - int(remaining[t]))
                if cost > adds_left:
                    continue
                nr = remaining.copy(); nr[t] = max(0, nr[t] - 2)
                nu = used.copy(); nu[t] += 2
                na = added.copy(); na[t] += cost
                pairs_dfs(t + 1, nr, pairs_left - 1, adds_left - cost,
                          nu, na)

        pairs_dfs(0, hand13.copy(), 7, int(max_add), zero.copy(), zero.copy())

        # --- kokushi musou : 13 orphelins + 1 doublon ------------------ #
        for dup in _KOKUSHI_TYPES:
            cost = 0
            for t in _KOKUSHI_TYPES:
                need = 2 if t == dup else 1
                miss = need - int(hand13[t])
                if miss > 0:
                    cost += miss
            if cost > max_add:
                continue
            win = zero.copy(); ad = zero.copy()
            for t in _KOKUSHI_TYPES:
                need = 2 if t == dup else 1
                take = min(need, int(hand13[t]))
                win[t] = need
                ad[t] = need - take
            emit(win, ad)

    return results

def _score_winning_hand_with_melds(win_counts, existing_melds, seat_wind,
                                   round_wind, is_dealer, dora_indicators,
                                   win_tile_type=None):
    """Score une main gagnante complete via la lib mahjong.

    [FIX LOOKAHEAD] tiles ET melds en ids 136 (la lib attend du 136, pas
    des comptages 34) ; tiles = partie cachee gagnante (win_counts) +
    tuiles de TOUS les melds declares (4e tuile des kans incluse).
    Rend le total (int) ou None (main invalide / pas de yaku).
    """
    full = np.asarray(win_counts, dtype=np.int32).copy()
    meld_objs = []
    for m in existing_melds:
        tiles = sorted(int(t) for t in m['tiles'])
        ids = [4 * t + i for i, t in enumerate(tiles)]
        meld_objs.append(Meld(
            meld_type=_MELD_TYPE_MAP.get(m['type'], Meld.PON),
            tiles=ids,
            opened=m['type'] != 'kan_closed'))
        for t in tiles:
            full[t] += 1
    tiles_136 = []
    for t in range(N_TILES):
        tiles_136.extend(4 * t + i for i in range(int(full[t])))
    if win_tile_type is None or full[int(win_tile_type)] <= 0:
        nz = np.nonzero(full)[0]
        if len(nz) == 0:
            return None
        win_tile_type = int(nz[-1])
    win_tile_136 = 4 * int(win_tile_type)
    config = _get_config(seat_wind, round_wind, is_dealer)
    try:
        result = _HAND_CALC.estimate_hand_value(
            tiles=tiles_136,
            win_tile=win_tile_136,
            melds=meld_objs,
            dora_indicators=[4 * int(d) for d in (dora_indicators or [])],
            config=config,
        )
    except Exception:
        return None
    if result is None or getattr(result, 'error', None) is not None:
        return None
    if result.cost is None or not result.han:
        return None
    return int(result.cost.get('total', 0) or 0)

@functools.lru_cache(maxsize=200_000)
def _enumerate_hands_cached(hand_tuple, melds_key, max_add):
    hand13 = np.array(hand_tuple, dtype=np.int32)
    existing_melds = _melds_from_key(melds_key)
    return _enumerate_winning_hands_with_melds(
        hand13, existing_melds, max_add)

@functools.lru_cache(maxsize=200_000)
def _score_with_dora_cached(win_tuple, melds_key, seat_wind, round_wind,
                            is_dealer, dora_key, win_tile_type):
    win = np.array(win_tuple, dtype=np.int32)
    existing_melds = _melds_from_key(melds_key)
    return _score_winning_hand_with_melds(
        win, existing_melds, seat_wind, round_wind, is_dealer,
        list(dora_key), win_tile_type)

def compute_lookahead(hand13_counts, melds, seat_wind, round_wind,
                      is_dealer, dora_indicators, unseen=None,
                      wall_remaining=None):
    """Look-ahead binaire (papier §2.2). 0/1 dans (n_replace, score_bucket).

    Canal (n_replace, bucket) = 1 s'il existe une main gagnante
    atteignable en <= n_replace + 1 tuiles tirees du mur (n_replace
    echanges + la tuile gagnante) avec un score dans ce bucket. La main
    peut faire 13 ou 14 tuiles (hors tour / apres pioche), ajustee du
    nombre de melds ; tout autre etat rend des canaux nuls.

    Enumeration par approfondissement iteratif (budgets 2..5) : chaque
    niveau trouve les mains de son cout tant que l'elaguage est encore
    fort, puis les resultats sont dedupliques, tries par cout croissant
    et bornees (les mains les plus proches d'abord). Les lignes sont
    ensuite filtrees par cout => monotonie exacte par construction
    (ligne n_replace+k allumee => lignes n_replace>k allumees).
    """
    features = np.zeros((LOOKAHEAD_CHANNELS, N_TILES), dtype=np.float32)
    hand13 = np.asarray(hand13_counts, dtype=np.int32)
    melds = list(melds or [])
    n_kans = sum(1 for m in melds
                 if m.get('type') in ('kan_open', 'kan_closed', 'shouminkan'))
    meld_tiles = sum(len(m.get('tiles', ())) for m in melds)
    between = 13 + n_kans - meld_tiles     # taille attendue hors tour
    n_hand = int(hand13.sum())
    if n_hand not in (between, between + 1):
        return features

    try:
        shanten = _SHANTEN_CALC.calculate_shanten(
            hand13.tolist(), use_chiitoitsu=True)
    except Exception:
        return features
    if shanten > MAX_LOOKAHEAD_SHANTEN:
        return features

    melds_key = tuple(json.dumps(m, sort_keys=True, default=int)
                      for m in melds)
    hand_tuple = tuple(int(x) for x in hand13)
    dora_key = tuple(int(d) for d in (dora_indicators or []))
    if unseen is not None:
        unseen = np.asarray(unseen, dtype=np.int32)

    # --- approfondissement iteratif : budgets 2 .. max(N_REPLACE)+1 ----- #
    max_budget = max(N_REPLACE_VALUES) + 1
    all_wins = {}
    for budget in range(2, max_budget + 1):
        try:
            level = _enumerate_hands_cached(hand_tuple, melds_key, budget)
        except Exception:
            continue
        for win, added in level:
            if win not in all_wins:
                all_wins[win] = added

    # --- tri par cout croissant puis scoring (borné) -------------------- #
    ordered = sorted(all_wins.items(), key=lambda kv: (sum(kv[1]), kv[0]))
    scored = []
    for win, added in ordered[:_LOOKAHEAD_SCORE_CAP]:
        if unseen is not None:
            drawable = True
            for t in range(N_TILES):
                if added[t] > int(unseen[t]):
                    drawable = False
                    break
            if not drawable:
                continue
        win_tile_type = None
        for t in range(N_TILES):
            if added[t] > 0:
                win_tile_type = t
                break
        total = _score_with_dora_cached(
            win, melds_key, seat_wind, round_wind, bool(is_dealer),
            dora_key, win_tile_type)
        if not total:
            continue
        scored.append((sum(added), _score_to_bucket(total)))

    # --- allumage des canaux, filtre par budget => monotonie ------------ #
    for row, n_replace in enumerate(N_REPLACE_VALUES):
        budget = n_replace + 1
        for cost, bucket in scored:
            if cost <= budget:
                features[row * N_SCORE_BUCKETS + bucket] = 1.0

    return features

def _next_dora(ind):
    if ind < 27:
        return (ind // 9) * 9 + (ind % 9 + 1) % 9
    if ind < 31:
        return 27 + (ind - 27 + 1) % 4
    return 31 + (ind - 31 + 1) % 3

# ====================================================================== #
# Score bucketing
# ====================================================================== #

_SCORES_UPPER = [
    -10**9, 0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000,
    12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 48000, 10**9,
]

def _score_bucket(score_k):
    """score_k : score en milliers (score - 25000) / 1000."""
    for i, upper in enumerate(_SCORES_UPPER):
        if score_k <= upper:
            return max(i, 1)
    return len(_SCORES_UPPER) - 1

def _wall_bucket(wall_remaining):
    wr = min(max(wall_remaining, 0), 70)
    return min(int(wr / 70.0 * 10), 9)

# ====================================================================== #
# Builder principal  [MODIFIE]
# ====================================================================== #

def empty_red_state():
    """Etat des rouges vide (aucun 5 rouge). Tous les tableaux sont (3,) bool.

    hand/meld/river : listes de 4 tableaux (un par siege).
    indicator       : tableau (3,) -- rouge utilise comme indicateur de dora.
    pending         : indice 0..2 du rouge si la tuile en attente est rouge, sinon -1.
    """
    return {
        'hand': [np.zeros(N_RED, dtype=bool) for _ in range(4)],
        'meld': [np.zeros(N_RED, dtype=bool) for _ in range(4)],
        'river': [np.zeros(N_RED, dtype=bool) for _ in range(4)],
        'indicator': np.zeros(N_RED, dtype=bool),
        'pending': -1,
    }


def _fill_red_channels(C, seat, opps, red_state):
    o_hand = _OFFSET['red_hand']
    o_meld = _OFFSET['red_meld_own']
    o_unseen = _OFFSET['red_unseen']
    ind = red_state.get('indicator')
    for k, t in enumerate(RED_TYPES):
        if red_state['hand'][seat][k]:
            C[o_hand][t] = 1.0
        if red_state['meld'][seat][k]:
            C[o_meld][t] = 1.0
        for s in range(4):
            if red_state['river'][s][k]:
                C[_OFFSET[f'red_river_s{s}']][t] = 1.0
        for i, opp in enumerate(opps):
            if red_state['meld'][opp][k]:
                C[_OFFSET[f'red_meld_opp{i}']][t] = 1.0
        visible = bool(red_state['hand'][seat][k]
                       or any(red_state['meld'][s][k] for s in range(4))
                       or any(red_state['river'][s][k] for s in range(4))
                       or (ind is not None and ind[k]))
        if not visible:
            C[o_unseen][t] = 1.0
    pend = int(red_state.get('pending', -1))
    if 0 <= pend < N_RED:
        C[_OFFSET['pending_red']][RED_TYPES[pend]] = 1.0


def build_suphx_obs(*, hands, melds, discards, discard_hist,
                    riichi_flags, furiten_flag, dora_indicators,
                    wall_remaining, scores, dealer_seat, round_wind,
                    round_id, honba, kyoutaku, seat,
                    pending_claim_tile=None,
                    compute_lookahead_flag=True,
                    is_last_normal_round=False, is_extension=False,
                    extension_index=0, normal_round_count=None,
                    rounds_remaining=None, red_state=None):
    """
    Construit l'observation (N_CHANNELS, 34) float32.

    rounds_remaining : int, nombre ABSOLU de manches normales restantes.
      - hanchan E1=7, E2=6, E3=5, E4=4, S1=3, S2=2, S3=1, S4=0
      - tonpusen E1=3, E2=2, E3=1, E4=0
      - extension (enchousen/Ouest) : toujours 0
    Si None, derive depuis normal_round_count et round_id.

    [SUPPRIME] game_format et round_progress.

    red_state : dict de `empty_red_state()` (position des 3 rouges), ou None
      = aucune info rouge (les 11 derniers canaux restent a zero).
    """
    C = np.zeros((N_CHANNELS, N_TILES), dtype=np.float32)
    def block(name):
        return _OFFSET[name]

    # --- Main propre ---
    o = block('own_hand')
    for n in range(4):
        C[o + n] = (hands[seat] >= (n + 1)).astype(np.float32)

    # --- Melds propres ---
    own_open = np.zeros(N_TILES, dtype=np.int32)
    own_ankan = np.zeros(N_TILES, dtype=np.int32)
    for m in melds[seat]:
        if m['type'] == 'chi':
            for t in m['tiles']:
                own_open[t] += 1
        elif m['type'] == 'kan_closed':
            own_ankan[m['tiles'][0]] += len(m['tiles'])
        else:
            own_open[m['tiles'][0]] += len(m['tiles'])
    o = block('own_meld_open')
    for n in range(4):
        C[o + n] = (own_open >= (n + 1)).astype(np.float32)
    o = block('own_meld_ankan')
    for n in range(4):
        C[o + n] = (own_ankan >= (n + 1)).astype(np.float32)

    # --- Discards par siege ---
    for s in range(4):
        o = block(f'discards_s{s}')
        for n in range(4):
            C[o + n] = (discards[s] >= (n + 1)).astype(np.float32)

    # --- Dernier discard + tsumogiri par siege ---
    for s in range(4):
        hist = discard_hist[s] if discard_hist is not None and s < len(discard_hist) else []
        if hist:
            last_tile = int(hist[-1][0])
            if 0 <= last_tile < N_TILES:
                C[block(f'last_discard_s{s}')][last_tile] = 1.0
        tg = np.zeros(N_TILES, dtype=np.float32)
        for (tile, ts) in hist:
            if ts and 0 <= tile < N_TILES:
                tg[tile] += 1.0
        C[block(f'tsumogiri_s{s}')] = tg

    # --- Melds adversaires ---
    opps = [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]
    for i, opp in enumerate(opps):
        opp_meld = np.zeros(N_TILES, dtype=np.int32)
        for m in melds[opp]:
            if m['type'] == 'chi':
                for t in m['tiles']:
                    opp_meld[t] += 1
            else:
                opp_meld[m['tiles'][0]] += len(m['tiles'])
        o = block(f'opp{i}_meld')
        for n in range(4):
            C[o + n] = (opp_meld >= (n + 1)).astype(np.float32)

    # --- Dora ---
    o = block('dora_ind')
    for ind in dora_indicators:
        if 0 <= ind < N_TILES:
            C[o][ind] += 1.0
    o = block('dora_tile')
    for ind in dora_indicators:
        dt = _next_dora(ind)
        if 0 <= dt < N_TILES:
            C[o][dt] += 1.0

    # --- Tuile en attente d'appel ---
    if pending_claim_tile is not None and 0 <= pending_claim_tile < N_TILES:
        C[block('pending')][pending_claim_tile] = 1.0

    # --- Tuiles non vues ---
    seen = hands[seat].astype(np.int32).copy()
    for s in range(4):
        for m in melds[s]:
            for t in m['tiles']:
                seen[t] += 1
        # [FIX UNSEEN] etait HORS de la boucle : seules les defausses du
        # siege 3 etaient comptees (s valait 3 apres la boucle).
        seen += discards[s]
    for ind in dora_indicators:
        if 0 <= ind < N_TILES:
            seen[ind] += 1
    unseen = np.clip(4 - seen, 0, 4).astype(np.float32) / 4.0
    C[block('unseen')] = unseen

    # --- Riichi ---
    if riichi_flags[seat]:
        C[block('own_riichi')] = 1.0
    for i, opp in enumerate(opps):
        if riichi_flags[opp]:
            C[block('opp_riichi') + i] = 1.0

    # --- Furiten ---
    if furiten_flag:
        C[block('own_furiten')] = 1.0

    # --- Vent de siege, vent du tour, dealer, numero de round ---
    seat_wind = (seat - dealer_seat) % 4
    o = block('seat_wind')
    for k in range(4):
        C[o + k] = 1.0 if k == seat_wind else 0.0
    o = block('round_wind')
    for k in range(4):
        C[o + k] = 1.0 if k == round_wind else 0.0
    C[block('dealer')] = 1.0 if seat == dealer_seat else 0.0
    o = block('round_id')
    for k in range(4):
        C[o + k] = 1.0 if k == (round_id - 1) else 0.0

    # [SUPPRIME] game_format -- le format n'est plus encode
    # [SUPPRIME] round_progress

    # --- Flags de fin de partie ---
    C[block('is_last_normal_round')] = 1.0 if is_last_normal_round else 0.0
    C[block('is_extension')] = 1.0 if is_extension else 0.0
    C[block('extension_index') + min(max(int(extension_index), 0), 7)] = 1.0

    # --- rounds_remaining ABSOLU (one-hot 0-7) ---
    if rounds_remaining is None:
        if normal_round_count is None:
            normal_round_count = 4  # default tonpuu
        rounds_remaining = max(normal_round_count - round_id, 0)
    rr = min(max(int(rounds_remaining), 0), 7)
    C[block('rounds_remaining') + rr] = 1.0

    # --- Rouges (canaux ajoutes en fin) ---
    if red_state is not None:
        _fill_red_channels(C, seat, opps, red_state)

    # --- Honba, kyoutaku ---
    C[block('honba') + min(int(honba), 4)] = 1.0
    C[block('kyoutaku') + min(int(kyoutaku), 4)] = 1.0

    # --- Scores ---
    C[block('own_score') + _score_bucket(scores[seat])] = 1.0
    for i, opp in enumerate(opps):
        C[block(f'opp{i}_score') + _score_bucket(scores[opp])] = 1.0

    # --- Mur restant ---
    C[block('wall_remaining') + _wall_bucket(wall_remaining)] = 1.0

    # --- Look-ahead ---
    if compute_lookahead_flag:
        is_dealer = (seat == dealer_seat)
        unseen_raw = np.clip(4 - seen, 0, 4).astype(np.int32)
        la = compute_lookahead(
            hands[seat], melds[seat],
            seat_wind=seat_wind, round_wind=round_wind,
            is_dealer=is_dealer, dora_indicators=dora_indicators,
            unseen=unseen_raw, wall_remaining=int(wall_remaining),
        )
        C[block('lookahead'):block('lookahead') + LOOKAHEAD_CHANNELS] = la

    return C