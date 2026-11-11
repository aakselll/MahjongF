"""
Parseur MJAI pour les datasets nikketryhard/tenhou-houou-mjai et
hhim8826/tenhou-houou-mjai (tonpusen).

[MODIF 2026-10-13] Suppression des vues tonpuu tronquees et du parametre
game_format. Le format est detecte automatiquement depuis le contenu du log.
Les donnees tonpusen doivent provenir de VRAIES parties tonpusen (pas de
hanchan tronques).

Extrait aussi les SKIPS (appels/riichi possibles non pris) pour eviter
que les tetes binaires n'apprennent que "toujours oui".
"""
import json
import os
import numpy as np
import zstandard as zstd
from mahjong_ai.envs.tiles import shanten_of, N_TILES, HAND_SIZE
from mahjong_ai.envs.suphx_features import build_suphx_obs, N_CHANNELS, N_RED
from mahjong_ai.envs.multi_agent import (
    DISCARD, PONG, CHOW, KONG, RIICHI, RON,
    CHOW_PASS, CHOW_LOWER, CHOW_MID, CHOW_UPPER,
)

OBS_DIM = N_CHANNELS * N_TILES

# ====================================================================== #
# Conversion tuile MJAI -> index 0-33
# ====================================================================== #

_TILE_MAP = {}
for i in range(1, 10):
    _TILE_MAP[f"{i}m"] = i - 1
    _TILE_MAP[f"{i}p"] = 9 + i - 1
    _TILE_MAP[f"{i}s"] = 18 + i - 1
_TILE_MAP.update({"E": 27, "S": 28, "W": 29, "N": 30,
                   "P": 31, "F": 32, "C": 33, "?": -1})
for red, normal in [("0m", "5m"), ("0p", "5p"), ("0s", "5s"),
                     ("5mr", "5m"), ("5pr", "5p"), ("5sr", "5s")]:
    if normal in _TILE_MAP:
        _TILE_MAP[red] = _TILE_MAP[normal]

# [ROUGES] 5 rouges : "5mr"/"5pr"/"5sr" (mjai) ou "0m"/"0p"/"0s" -> 0, 1, 2
_RED_STR = {"5mr": 0, "0m": 0, "5pr": 1, "0p": 1, "5sr": 2, "0s": 2}


def mjai_red_index(tile_str):
    """Indice du rouge (0=5m, 1=5p, 2=5s) si la tuile est un 5 rouge, sinon -1."""
    return _RED_STR.get(tile_str, -1)


def mjai_tile_to_34(tile_str):
    if tile_str is None or tile_str == "?":
        return -1
    return _TILE_MAP.get(tile_str, -1)

def mjai_tiles_to_counts(tiles):
    counts = np.zeros(N_TILES, dtype=np.int32)
    for t in tiles:
        idx = mjai_tile_to_34(t)
        if 0 <= idx < N_TILES:
            counts[idx] += 1
    return counts

# ====================================================================== #
# Detection du format (tonpuu vs hanchan)  [NOUVEAU]
# ====================================================================== #

def _detect_format(events):
    """Detecte si la partie est un tonpusen ou un hanchan.

    Strategie :
    1. Si start_game contient kyoku_first, l'utiliser directement.
       (hhim8826 : kyoku_first=0 hanchan, 4 tonpusen)
    2. Sinon, heuristique : si le log contient un start_kyoku avec
       bakaze="S", c'est soit un hanchan (S1-S4 normaux) soit un
       tonpusen-enchousen. Pour nikketryhard (tout hanchan), retourner
       hanchan.
    """
    start_game = next(
        (ev for ev in events if ev.get("type") == "start_game"), {}
    )
    kyoku_first = start_game.get("kyoku_first", None)

    if kyoku_first is not None:
        return "tonpuu" if kyoku_first == 4 else "hanchan"

    # Heuristique pour logs sans metadata explicite.
    # Les fichiers nikketryhard sont tous hanchan.
    # Les fichiers hhim8826 ont kyoku_first.
    return "hanchan"

def _compute_round_context(event, normal_round_count):
    """Calcule le contexte de tour pour un start_kyoku donne.

    Args:
        event : dict start_kyoku avec bakaze, kyoku, honba
        normal_round_count : 4 pour tonpusen, 8 pour hanchan

    Retourne un dict avec rounds_remaining ABSOLU et flags.
    """
    bakaze = event.get("bakaze", "E")
    kyoku = int(event.get("kyoku", 1) or 1)

    if bakaze == "E":
        absolute_round = kyoku
    elif bakaze == "S":
        absolute_round = 4 + kyoku
    elif bakaze == "W":
        absolute_round = 8 + kyoku
    else:  # "N"
        absolute_round = 12 + kyoku

    rounds_remaining = max(normal_round_count - absolute_round, 0)
    is_extension = absolute_round > normal_round_count
    is_last_normal_round = absolute_round == normal_round_count
    extension_index = max(0, absolute_round - normal_round_count)

    return {
        "is_last_normal_round": is_last_normal_round,
        "is_extension": is_extension,
        "extension_index": extension_index,
        "rounds_remaining": rounds_remaining,
    }

# ====================================================================== #
# Lecture
# ====================================================================== #

def parse_mjai_zst_file(filepath):
    events = []
    dctx = zstd.ZstdDecompressor()
    try:
        with open(filepath, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                raw = reader.read()
    except Exception:
        return events
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events

# ====================================================================== #
# Etat de jeu reconstruit
# ====================================================================== #

class GameState:
    def __init__(self):
        self.kyoutaku = 0
        self.reset_kyoku()

    def reset_kyoku(self):
        self.oya = 0
        self.bakaze = "E"
        self.kyoku = 1
        self.honba = 0
        self.scores = [25000, 25000, 25000, 25000]
        self.hands = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.discards = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
        # [FIX FURITEN] defausses completes (tuiles prises par chi/pon/kan
        # comprises) : `discards` les retire, le furiten doit les compter.
        self.river_all = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
        self.discard_history = [[] for _ in range(4)]
        self.riichi = [False, False, False, False]
        self.dora_indicators = []
        self.wall_remaining = 70
        self.last_draw = [None, None, None, None]
        self.pending_tile = None
        self.last_discarder = None
        self._feature_context = None
        # [ROUGES] position des 3 rouges (voir envs/suphx_features.empty_red_state)
        self.red_hand = [np.zeros(N_RED, dtype=bool) for _ in range(4)]
        self.red_river = [np.zeros(N_RED, dtype=bool) for _ in range(4)]
        self.red_indicator = np.zeros(N_RED, dtype=bool)
        self.last_dahai_red = -1       # rouge de la derniere defausse (si pending_tile)

    def red_state(self):
        meld = []
        for s in range(4):
            a = np.zeros(N_RED, dtype=bool)
            for m in self.melds[s]:
                for k in m.get("reds", ()):
                    a[k] = True
            meld.append(a)
        return {'hand': self.red_hand, 'meld': meld, 'river': self.red_river,
                'indicator': self.red_indicator,
                'pending': (self.last_dahai_red
                            if self.pending_tile is not None else -1)}

    def get_hand_counts(self, seat):
        return self.hands[seat].copy()

    def get_meld_counts(self, seat):
        counts = np.zeros(N_TILES, dtype=np.int32)
        for m in self.melds[seat]:
            for t in m["tiles"]:
                if 0 <= t < N_TILES:
                    counts[t] += 1
        return counts

    def get_full_counts(self, seat):
        counts = self.hands[seat] + self.get_meld_counts(seat)
        for m in self.melds[seat]:
            if m["type"] in ("kan_closed", "shouminkan", "kan_open"):
                t0 = m["tiles"][0]
                if 0 <= t0 < N_TILES:
                    counts[t0] -= 1
        return counts

    def get_shanten_counts(self, seat):
        return self.get_full_counts(seat)

    def get_opponents(self, pov):
        return [(pov + 1) % 4, (pov + 2) % 4, (pov + 3) % 4]

# ====================================================================== #
# Furiten pour le POV (permanent seulement)
# ====================================================================== #

def _pov_furiten(state, pov):
    counts = state.get_shanten_counts(pov)
    if int(counts.sum()) >= 14:
        return False
    try:
        if shanten_of(counts.astype(np.int32).tolist()) != 0:
            return False
    except Exception:
        return False
    for t in range(N_TILES):
        if state.river_all[pov][t] > 0:
            trial = counts.copy()
            trial[t] += 1
            try:
                if shanten_of(trial.astype(np.int32).tolist()) == -1:
                    return True
            except Exception:
                continue
    return False

# ====================================================================== #
# Build obs via module partage  [MODIFIE]
# ====================================================================== #

_RW_MAP = {"E": 0, "S": 1, "W": 2, "N": 3}

def build_obs(state, pov):
    round_wind = _RW_MAP.get(state.bakaze, 0)
    scores_k = [(s - 25000.0) / 1000.0 for s in state.scores]
    context = state._feature_context

    if context is None:
        # Fallback si pas de contexte (ne devrait pas arriver)
        context = _compute_round_context(
            {"bakaze": state.bakaze, "kyoku": state.kyoku}, 8)

    return build_suphx_obs(
        hands=state.hands,
        melds=state.melds,
        discards=state.discards,
        discard_hist=state.discard_history,
        riichi_flags=state.riichi,
        furiten_flag=_pov_furiten(state, pov),
        dora_indicators=state.dora_indicators,
        wall_remaining=state.wall_remaining,
        scores=scores_k,
        dealer_seat=state.oya,
        round_wind=round_wind,
        round_id=int(state.kyoku),
        honba=int(state.honba),
        kyoutaku=int(state.kyoutaku),
        seat=pov,
        pending_claim_tile=state.pending_tile,
        compute_lookahead_flag=True,
        is_last_normal_round=context["is_last_normal_round"],
        is_extension=context["is_extension"],
        extension_index=context["extension_index"],
        rounds_remaining=context["rounds_remaining"],
        red_state=state.red_state(),
    )

# ====================================================================== #
# Labels
# ====================================================================== #

def extract_labels_from_event(event, state, pov):
    ev_type = event.get("type")
    actor = event.get("actor")
    if actor != pov:
        return None
    if ev_type == "dahai":
        tile = mjai_tile_to_34(event.get("pai"))
        if tile < 0:
            return None
        mask = (state.hands[pov] > 0).astype(np.float32)
        if mask[tile] == 0:
            mask[tile] = 1.0
        return DISCARD, tile, mask
    if ev_type == "reach":
        return RIICHI, 1, np.array([1.0, 1.0], dtype=np.float32)
    if ev_type == "pon":
        return PONG, 1, np.array([1.0, 1.0], dtype=np.float32)
    if ev_type == "chi":
        consumed = [mjai_tile_to_34(t) for t in event.get("consumed", [])]
        called = mjai_tile_to_34(event.get("pai"))
        if len(consumed) != 2 or called < 0:
            return None
        all_tiles = sorted(consumed + [called])
        variant = CHOW_PASS
        if len(all_tiles) == 3:
            if all_tiles == [called - 2, called - 1, called]:
                variant = CHOW_LOWER
            elif all_tiles == [called - 1, called, called + 1]:
                variant = CHOW_MID
            elif all_tiles == [called, called + 1, called + 2]:
                variant = CHOW_UPPER
        mask = np.zeros(4, dtype=np.float32)
        mask[CHOW_PASS] = 1.0
        mask[variant] = 1.0
        return CHOW, variant, mask
    if ev_type in ("daiminkan", "kakan", "ankan"):
        return KONG, 1, np.array([1.0, 1.0], dtype=np.float32)
    if ev_type == "hora":
        target = event.get("target")
        if target is not None and target != actor:
            return RON, 1, np.array([1.0, 1.0], dtype=np.float32)
    return None

# ====================================================================== #
# Skips
# ====================================================================== #

def _compute_pending_calls(state, discarder, tile):
    pending = {}
    for s in range(4):
        if s == discarder or state.riichi[s]:
            continue
        calls = {"pon": False, "chi": False, "kan": False, "ron": False}
        if state.hands[s][tile] >= 2:
            calls["pon"] = True
        if state.hands[s][tile] >= 3:
            calls["kan"] = True
        chi_variants = []
        if s == (discarder + 1) % 4 and tile < 27:
            col = tile % 9
            h = state.hands[s]
            if col >= 2 and h[tile - 2] >= 1 and h[tile - 1] >= 1:
                chi_variants.append(CHOW_LOWER)
            if 1 <= col <= 7 and h[tile - 1] >= 1 and h[tile + 1] >= 1:
                chi_variants.append(CHOW_MID)
            if col <= 6 and h[tile + 1] >= 1 and h[tile + 2] >= 1:
                chi_variants.append(CHOW_UPPER)
        if chi_variants:
            calls["chi"] = True
            calls["chi_variants"] = chi_variants
        try:
            trial = state.get_shanten_counts(s).copy().astype(np.int32)
            trial[tile] += 1
            if shanten_of(trial.tolist()) == -1 and state.river_all[s][tile] == 0:
                calls["ron"] = True
        except Exception:
            pass
        if any(calls[k] for k in ("pon", "chi", "kan", "ron")):
            pending[s] = calls
    return pending

def _extract_pov_skip(samples, state, pov, calls):
    obs = build_obs(state, pov)
    if calls.get("ron"):
        samples.append((obs, RON, 0, np.array([1.0, 1.0], dtype=np.float32)))
        return
    if calls.get("kan"):
        samples.append((obs, KONG, 0, np.array([1.0, 1.0], dtype=np.float32)))
        return
    if calls.get("pon"):
        samples.append((obs, PONG, 0, np.array([1.0, 1.0], dtype=np.float32)))
        return
    if calls.get("chi"):
        mask = np.zeros(4, dtype=np.float32)
        mask[CHOW_PASS] = 1.0
        for v in calls.get("chi_variants", []):
            mask[v] = 1.0
        samples.append((obs, CHOW, 0, mask))
        return

def _flush_skips(samples, state, pov, pending_calls):
    if pov in pending_calls:
        _extract_pov_skip(samples, state, pov, pending_calls[pov])

# ====================================================================== #
# Conversion partie -> echantillons  [MODIFIE]
# ====================================================================== #

def events_to_samples(events, pov=0):
    """Extrait les echantillons pour UN point de vue.

    [MODIFIE] Le format est detecte automatiquement depuis le log.
    [SUPPRIME] Plus de parametre game_format ni de vue tonpuu tronquee.
    """
    samples = []
    state = GameState()
    pending_calls = {}
    pending_riichi_check = False

    # Detecter le format UNE FOIS par partie
    game_format = _detect_format(events)
    normal_round_count = 4 if game_format == "tonpuu" else 8

    for ev in events:
        ev_type = ev.get("type")

        if ev_type == "start_game":
            state.kyoutaku = 0
            state.reset_kyoku()
            continue

        if ev_type in ("end_kyoku", "end_game", "ryukyoku"):
            if pending_calls:
                _flush_skips(samples, state, pov, pending_calls)
                pending_calls = {}
                pending_riichi_check = False
            state.pending_tile = None
            continue

        if ev_type == "start_kyoku":
            saved_kyoutaku = state.kyoutaku
            state.reset_kyoku()
            state.kyoutaku = saved_kyoutaku
            state.oya = ev.get("oya", 0)
            state.bakaze = ev.get("bakaze", "E")
            state.kyoku = ev.get("kyoku", 1)
            state.honba = ev.get("honba", 0)
            state.scores = ev.get("scores", [25000] * 4)
            tehais = ev.get("tehais", [])
            for s in range(4):
                if s < len(tehais):
                    state.hands[s] = mjai_tiles_to_counts(tehais[s])
                    for t_str in tehais[s]:
                        k = mjai_red_index(t_str)
                        if k >= 0:
                            state.red_hand[s][k] = True
            dora = ev.get("dora_marker")
            if dora:
                state.dora_indicators = [mjai_tile_to_34(dora)]
                if mjai_red_index(dora) >= 0:
                    state.red_indicator[mjai_red_index(dora)] = True
            state.wall_remaining = 70
            state._feature_context = _compute_round_context(
                ev, normal_round_count)
            pending_calls = {}
            pending_riichi_check = False
            continue

        if ev_type == "tsumo":
            actor = ev.get("actor")
            tile = mjai_tile_to_34(ev.get("pai"))
            if pending_calls:
                _flush_skips(samples, state, pov, pending_calls)
                pending_calls = {}
                state.pending_tile = None
            if actor is not None and tile >= 0:
                state.hands[actor][tile] += 1
                if mjai_red_index(ev.get("pai")) >= 0:
                    state.red_hand[actor][mjai_red_index(ev.get("pai"))] = True
                state.last_draw[actor] = tile
                state.wall_remaining = max(0, state.wall_remaining - 1)
                if actor == pov and not state.riichi[pov]:
                    try:
                        sh = shanten_of(
                            state.get_shanten_counts(pov).astype(np.int32).tolist())
                        hand_closed = all(
                            m["type"] == "kan_closed" for m in state.melds[pov])
                        if sh == 0 and hand_closed:
                            pending_riichi_check = True
                    except Exception:
                        pass
            continue

        if ev_type == "dahai":
            actor = ev.get("actor")
            tile = mjai_tile_to_34(ev.get("pai"))
            tsumogiri = ev.get("tsumogiri", False)
            if actor == pov:
                obs = build_obs(state, pov)
                mask = (state.hands[pov] > 0).astype(np.float32)
                if tile >= 0 and mask[tile] == 0:
                    mask[tile] = 1.0
                samples.append((obs, DISCARD, tile if tile >= 0 else 0, mask))
                if pending_riichi_check and not state.riichi[pov]:
                    obs_r = build_obs(state, pov)
                    samples.append((obs_r, RIICHI, 0,
                                    np.array([1.0, 1.0], dtype=np.float32)))
                    pending_riichi_check = False
            if actor is not None and tile >= 0:
                state.hands[actor][tile] -= 1
                state.discards[actor][tile] += 1
                state.river_all[actor][tile] += 1
                k_red = mjai_red_index(ev.get("pai"))
                state.last_dahai_red = k_red
                if k_red >= 0:
                    state.red_hand[actor][k_red] = False
                    state.red_river[actor][k_red] = True
                state.discard_history[actor].append((tile, bool(tsumogiri)))
                state.last_discarder = actor
                state.pending_tile = tile
                pending_calls = _compute_pending_calls(state, actor, tile)
            continue

        if ev_type in ("chi", "pon", "daiminkan", "kakan", "ankan"):
            actor = ev.get("actor")
            if actor == pov:
                obs = build_obs(state, pov)
                label = extract_labels_from_event(ev, state, pov)
                if label is not None:
                    samples.append((obs,) + label)
            else:
                if pov in pending_calls:
                    _extract_pov_skip(samples, state, pov, pending_calls[pov])
            pending_calls = {}
            state.pending_tile = None
            called = mjai_tile_to_34(ev.get("pai"))
            called_red = mjai_red_index(ev.get("pai"))
            from_discard = (ev_type in ("chi", "pon", "daiminkan")
                            and called >= 0 and state.last_discarder is not None
                            and state.last_discarder != actor)
            if (called >= 0 and state.last_discarder is not None
                    and state.last_discarder != actor):
                state.discards[state.last_discarder][called] -= 1
                if state.discards[state.last_discarder][called] < 0:
                    state.discards[state.last_discarder][called] = 0
            if actor is not None:
                consumed_raw = list(ev.get("consumed", []))
                consumed = [mjai_tile_to_34(t) for t in consumed_raw]
                # [ROUGES] rouges qui entrent dans le meld
                meld_reds = []
                if from_discard and called_red >= 0:
                    state.red_river[state.last_discarder][called_red] = False
                    meld_reds.append(called_red)
                if ev_type == "kakan":
                    # [FIX KAKAN] en mjai, `consumed` = les 3 tuiles du pon
                    # DEJA expose et `pai` = la tuile ajoutee depuis la main.
                    # L'ancien code retirait 3 tuiles de la main au lieu
                    # d'une (compteurs negatifs, tuile ajoutee jamais retiree).
                    if called >= 0:
                        state.hands[actor][called] -= 1
                    if called_red >= 0:
                        state.red_hand[actor][called_red] = False
                        meld_reds.append(called_red)
                else:
                    for t_str, t in zip(consumed_raw, consumed):
                        if t >= 0:
                            state.hands[actor][t] -= 1
                        k = mjai_red_index(t_str)
                        if k >= 0:
                            state.red_hand[actor][k] = False
                            meld_reds.append(k)
                if ev_type == "chi" and called >= 0:
                    state.melds[actor].append(
                        {"type": "chi", "tiles": sorted(consumed + [called]),
                         "reds": meld_reds})
                elif ev_type == "pon" and called >= 0:
                    state.melds[actor].append(
                        {"type": "pon", "tiles": [called] * 3,
                         "reds": meld_reds})
                elif ev_type == "daiminkan" and called >= 0:
                    state.melds[actor].append(
                        {"type": "kan_open", "tiles": [called] * 4,
                         "reds": meld_reds})
                elif ev_type == "kakan" and called >= 0:
                    for m in state.melds[actor]:
                        if m["type"] == "pon" and m["tiles"][0] == called:
                            m["type"] = "shouminkan"
                            m["tiles"].append(called)
                            m.setdefault("reds", []).extend(meld_reds)
                            break
                elif ev_type == "ankan" and consumed:
                    state.melds[actor].append(
                        {"type": "kan_closed", "tiles": [consumed[0]] * 4,
                         "reds": meld_reds})
            continue

        if ev_type == "reach":
            actor = ev.get("actor")
            if actor == pov:
                obs = build_obs(state, pov)
                samples.append((obs, RIICHI, 1,
                                np.array([1.0, 1.0], dtype=np.float32)))
            if actor is not None:
                state.riichi[actor] = True
                pending_riichi_check = False
            continue

        if ev_type == "reach_accepted":
            actor = ev.get("actor")
            if actor is not None:
                state.kyoutaku += 1
            continue

        if ev_type == "dora":
            dora = ev.get("dora_marker")
            if dora:
                idx = mjai_tile_to_34(dora)
                if idx >= 0:
                    state.dora_indicators.append(idx)
                    if mjai_red_index(dora) >= 0:
                        state.red_indicator[mjai_red_index(dora)] = True
            continue

        if ev_type == "hora":
            actor = ev.get("actor")
            target = ev.get("target")
            if target is not None and actor == pov and target != pov:
                obs = build_obs(state, pov)
                samples.append((obs, RON, 1,
                                np.array([1.0, 1.0], dtype=np.float32)))
            elif target is not None and pov in pending_calls:
                if pending_calls[pov].get("ron"):
                    _extract_pov_skip(samples, state, pov, pending_calls[pov])
            pending_calls = {}
            state.pending_tile = None
            state.kyoutaku = 0
            continue

    return samples

def events_to_samples_all_povs(events):
    """Extrait les echantillons pour les 4 POV.
    [MODIFIE] Retourne (samples, game_format) pour metadata.
    [SUPPRIME] Plus de parametre game_format ni de vue tonpuu tronquee.
    """
    game_format = _detect_format(events)
    samples = []
    for pov in range(4):
        samples.extend(events_to_samples(events, pov=pov))
    return samples, game_format