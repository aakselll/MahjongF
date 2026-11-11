"""
Environnement multi-agent ACTIF (couvre toutes les mecaniques : Chow/Pong/
Kong dont daiminkan, Riichi, vrai scoring han/fu/dora via HandCalculator,
rotation du dealer, Ron decisionnel + furiten permanent et temporaire).

[REFONTE FEATURES] _build_obs produit maintenant une matrice (N_CHANNELS, 34)
via le module partage envs/suphx_features.build_suphx_obs, pour que SL et RL
utilisent EXACTEMENT le meme layout d'entree (conforme au papier Suphx §2.2 :
canaux 34-wide, look-ahead, bucketing, categoriques one-hot).
"""
import os

import numpy as np

from mahjong.constants import EAST, SOUTH, WEST, NORTH
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld

from mahjong_ai.envs.suphx_features import (
    build_suphx_obs, N_CHANNELS, RED_TYPES, RED_INDEX, N_RED)
from mahjong_ai.envs.tiles import shanten_of, N_TILES, COPIES_PER_TILE, HAND_SIZE

# [ROUGES] 3 rouges (un 5 par famille : 0m, 0p, 0s), comme Mahjong Soul /
# Tenhou. MJ_RED_FIVES=0 pour retrouver l'ancien env sans rouges.
USE_RED_FIVES = os.environ.get("MJ_RED_FIVES", "1") != "0"

DEAD_WALL_SIZE = 14
RINSHAN_POOL_SIZE = 4
N_DORA_PAIRS = 5

DISCARD = 0
PONG = 1
CHOW = 2
KONG = 3
RIICHI = 4
RON = 5

CHOW_PASS, CHOW_LOWER, CHOW_MID, CHOW_UPPER = 0, 1, 2, 3
KONG_NO, KONG_YES = 0, 1
RIICHI_NO, RIICHI_YES = 0, 1
RON_NO, RON_YES = 0, 1

SEAT_WINDS = [EAST, SOUTH, WEST, NORTH]
ROUND_WIND = EAST

STEP_PENALTY = 0.0
REWARD_SCALE = 1.0 / 1000.0
RIICHI_STICK_COST = 1000.0
NOTEN_PAYMENT_POOL = 3000.0
RON_REFUSE_PENALTY = 0.05

RIICHI_STICK_NET_COST = 1000.0
RIICHI_DECLARE_BONUS = 0.0
N_ROUNDS_CTX = 8
SCORE_NORM = 20.0
RECENCY_DECAY = 0.85
RECENCY_WINDOW = 12
# --- Vecteur oracle a plat (pour _pending_oracle initialisation) ---
ORACLE_FLAT_DIM = 4 * 34       # 136 : 3 mains + mur, aplati
# --- Canaux oracle pour l'agent RL (4 canaux de 34) ---
ORACLE_DIM = 4

def _get_env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_shanten_discard_coef():
    return _get_env_float("SHANTEN_DISCARD_COEF", 0.0)


def _get_shanten_call_coef():
    return _get_env_float("SHANTEN_CALL_COEF", 0.0)


FORCE_CLOSED_HAND_PROB = _get_env_float("FORCE_CLOSED_HAND_PROB", 0.0)
OPEN_MELD_PENALTY = _get_env_float("OPEN_MELD_PENALTY", 0.0)
OPEN_KAN_PENALTY = _get_env_float("OPEN_KAN_PENALTY", 0.0)
ANKAN_PENALTY = _get_env_float("ANKAN_PENALTY", 0.0)
NO_YAKU_CALL_PENALTY = _get_env_float("NO_YAKU_CALL_PENALTY", 0.0)

_MELD_TYPE_MAP = {
    "chi": Meld.CHI,
    "pon": Meld.PON,
    "kan_closed": Meld.KAN,
    "shouminkan": Meld.SHOUMINKAN,
    "kan_open": Meld.KAN,
}

class MultiAgentMahjongEnv:
    def __init__(self, seed=None, frozen_sl_models=None, compute_lookahead=True):
        self.rng = np.random.RandomState(seed)
        # [RL] modeles SL figes pour riichi/chow/pong/kong.
        # Si None, ces tetes yield une request comme avant.
        # Si dict {'riichi': model, ...}, le modele SL decide directement.
        self.frozen_sl = frozen_sl_models
        # [DANGER] si True, les requetes DISCARD contiennent 'danger'
        # (34 flottants) : 1 si discarder cette tuile donnerait ron a un
        # adversaire non-furiten tenpai sur cette attente. Label calcule
        # depuis les mains cachees : feedback dense pour apprendre a
        # lire le danger (oracle en phase 1, rivieres en phase 2/3).
        self.compute_danger = False
        self._gen = None
        self.hands = None
        self.melds = None
        self.discards = None
        self.use_red = USE_RED_FIVES
        self.river_all = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
        self._clear_reds()
        self._compute_lookahead = compute_lookahead
        self.wall = None
        self.rinshan_pool = None
        self.dora_indicators = None
        self.ura_dora_indicators = None
        self.riichi = None
        self.riichi_deposits = 0
        # [FIX BATONS RIICHI] cout de baton deja paye CETTE manche, par
        # siege interne (unite reward, positif = paye). Applique dans _final.
        self.stick_paid = [0.0, 0.0, 0.0, 0.0]
        self._furiten_temp = [False, False, False, False]
        self._abortive_riichi_count = 0
        self._abortive_kan_count = 0
        self._ippatsu = [False] * 4
        self._double_riichi = [False] * 4
        self._discard_count = [0] * 4
        self._discard_hist = [[] for _ in range(4)]
        self._safe_after_riichi = [np.zeros(N_TILES, dtype=np.float32) for _ in range(4)]
        self._any_call = False
        self._last_discarder = None
        self._pending_oracle = np.zeros((ORACLE_DIM, N_TILES), dtype=np.float32)
        self.ctx = {"round_wind_idx": 0, "round_idx": 0, "scores": [0.0] * 4}
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats():
        return {
            "rounds": 0, "ankan": 0, "shouminkan": 0, "daiminkan": 0,
            "riichi": 0, "tsumo": 0, "ron": 0, "ron_refused": 0,
            "multi_ron": 0, "exhaustive": 0,
            "abortive_4riichi": 0, "abortive_4kan": 0, "abortive_3ron": 0,
            "wins_total": 0, "yaku_counts": {},
            "forced_closed_seat_hands": 0, "seat_hands": 0,
            "pon_calls": 0, "chi_calls": 0, "open_calls": 0,
            "wins_open": 0, "wins_closed": 0, "no_yaku_calls": 0,
            "sl_errors": 0,
        }

    def get_stats(self):
        return dict(self._stats)

    def reset_stats(self):
        self._stats = self._new_stats()

    def _record_win_yaku(self, seat, win_result):
        self._stats["wins_total"] += 1
        if self._is_hand_closed(seat):
            self._stats["wins_closed"] += 1
        else:
            self._stats["wins_open"] += 1
        if win_result.yaku:
            for y in win_result.yaku:
                self._stats["yaku_counts"][y.name] = \
                    self._stats["yaku_counts"].get(y.name, 0) + 1

    def set_game_context(self, round_wind_idx, round_idx, scores, honba=0,
                         kyoutaku=0):
        # kyoutaku : batons de riichi restes sur la table (manche nulle
        # precedente) ; ils sont deja payes, reportes ici et empoches par
        # le prochain gagnant.
        self.ctx = {"round_wind_idx": int(round_wind_idx),
                    "round_idx": int(round_idx),
                    "scores": [float(x) for x in scores],
                    "honba": int(honba),
                    "kyoutaku": int(kyoutaku)}

    def reset_round(self):
        while True:
            self._gen = self._round_flow()
            try:
                first = next(self._gen)
            except StopIteration:
                continue
            return first["next_request"]

    # def step(self, action):
    #     try:
    #         return self._gen.send(action)
    #     except StopIteration as e:
    #         return e.value


    def step(self, action):
        try:
            result = self._gen.send(action)
        except StopIteration as e:
            result = e.value
        # [pMCPA] Stocke le reward pour les rollouts
        if result.get('round_done'):
            self._last_round_rewards = result.get('final_rewards', {})
        return result

    @property
    def obs_dim(self):
        return N_CHANNELS * N_TILES

    @property
    def oracle_dim(self):
        return ORACLE_DIM

    def _deal(self):
        full_wall = []
        for tile in range(N_TILES):
            full_wall.extend([tile] * COPIES_PER_TILE)
        if self.use_red:
            # [ROUGES] une copie de chaque 5 devient un sentinel N_TILES+k
            # (34, 35, 36) ; decode par _type_of / _to_hand a la pioche.
            for k, t in enumerate(RED_TYPES):
                full_wall[full_wall.index(t)] = N_TILES + k
        self.rng.shuffle(full_wall)

        self.hands = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.discards = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
        self._clear_reds()
        # [FIX FURITEN] defausses COMPLETES (tuiles prises par chi/pon/kan
        # comprises) : `discards` les retire (feature), mais le furiten
        # permanent doit les compter.
        self.river_all = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
        self.riichi = [False, False, False, False]
        # [FIX BATONS RIICHI] on repart des batons reportes (0 par defaut).
        self.riichi_deposits = int(self.ctx.get("kyoutaku", 0))
        self.stick_paid = [0.0, 0.0, 0.0, 0.0]
        self._furiten_temp = [False, False, False, False]
        self._abortive_riichi_count = 0
        self._abortive_kan_count = 0
        self._ippatsu = [False] * 4
        self._double_riichi = [False] * 4
        self._discard_count = [0] * 4
        self._discard_hist = [[] for _ in range(4)]
        self._safe_after_riichi = [np.zeros(N_TILES, dtype=np.float32) for _ in range(4)]
        self._any_call = False
        self._last_discarder = None
        self.force_closed = [self.rng.random() < FORCE_CLOSED_HAND_PROB for _ in range(4)]
        self._stats["seat_hands"] += 4
        self._stats["forced_closed_seat_hands"] += sum(self.force_closed)

        idx = 0
        for seat in range(4):
            for tile in full_wall[idx:idx + HAND_SIZE]:
                self._to_hand(seat, tile)
            idx += HAND_SIZE

        dead_wall = full_wall[len(full_wall) - DEAD_WALL_SIZE:]
        self.wall = full_wall[idx:len(full_wall) - DEAD_WALL_SIZE]
        self._initial_wall_size = len(self.wall)

        self.rinshan_pool = dead_wall[:RINSHAN_POOL_SIZE]
        self._sacrificed_count = 0
        dora_pairs = dead_wall[RINSHAN_POOL_SIZE:]
        self._dora_pairs = [(dora_pairs[2 * i], dora_pairs[2 * i + 1])
                             for i in range(N_DORA_PAIRS)]
        self.dora_indicators = [self._reveal_indicator(self._dora_pairs[0][0])]
        self.ura_dora_indicators = [self._type_of(self._dora_pairs[0][1])]
        self._next_dora_pair_idx = 1

    # ------------------------------------------------------------------ #
    # [ROUGES] Suivi des 3 rouges (un 5 rouge par famille)
    # ------------------------------------------------------------------ #
    # Les tuiles restent comptees par TYPE (34). Pour chaque rouge k (0=5m,
    # 1=5p, 2=5s) on suit seulement OU il est :
    #   red_hand[seat][k]  : dans la main du siege (implique hands[t] >= 1)
    #   red_river[seat][k] : dans la riviere du siege (pas encore appele)
    #   melds[seat][i]["reds"] : liste des k dont le rouge est dans le meld
    # Regle de defausse (l'action est un TYPE) : on garde le rouge, on
    # defausse la copie normale d'abord ; le rouge part quand c'est la
    # derniere copie de la main (ou un tsumogiri force en riichi).
    def _clear_reds(self):
        self.red_hand = [np.zeros(N_RED, dtype=bool) for _ in range(4)]
        self.red_river = [np.zeros(N_RED, dtype=bool) for _ in range(4)]
        self.red_indicator = np.zeros(N_RED, dtype=bool)
        self._last_discard_red = -1     # k si la derniere tuile defaussee / kakan est rouge
        self._drawn_red = -1            # k si la tuile qu'on vient de piocher est rouge

    @staticmethod
    def _type_of(raw):
        return int(raw) if raw < N_TILES else RED_TYPES[int(raw) - N_TILES]

    def _reveal_indicator(self, raw):
        if raw >= N_TILES:
            self.red_indicator[int(raw) - N_TILES] = True
        return self._type_of(raw)

    def _to_hand(self, seat, raw):
        """Ajoute une tuile (type ou sentinel rouge) a la main ; retourne son type."""
        t = self._type_of(raw)
        self.hands[seat][t] += 1
        self._drawn_red = -1
        if raw >= N_TILES:
            self.red_hand[seat][int(raw) - N_TILES] = True
            self._drawn_red = int(raw) - N_TILES
        return t

    def _after_discard(self, seat, t, tsumogiri_red=-1):
        """A appeler APRES hands[seat][t] -= 1 : decide si la tuile defaussee est le rouge."""
        k = RED_INDEX.get(t, -1)
        out = -1
        if k >= 0 and self.red_hand[seat][k] and (
                self.hands[seat][t] == 0 or tsumogiri_red == k):
            self.red_hand[seat][k] = False
            self.red_river[seat][k] = True
            out = k
        self._last_discard_red = out

    def _claim_reds(self, seat, called_type, removed_types):
        """A appeler APRES le retrait des tuiles de la main pour un meld.
        Retourne la liste des rouges qui entrent dans le meld (tuile appelee
        + rouges de la main dont la derniere copie vient de partir)."""
        reds = []
        if called_type is not None:
            k = RED_INDEX.get(called_type, -1)
            if k >= 0 and self._last_discard_red == k:
                d = self._last_discarder
                if d is not None:
                    self.red_river[d][k] = False
                reds.append(k)
                self._last_discard_red = -1
        for t in set(removed_types):
            kk = RED_INDEX.get(t, -1)
            if kk >= 0 and self.red_hand[seat][kk] and self.hands[seat][t] == 0:
                self.red_hand[seat][kk] = False
                reds.append(kk)
        return reds

    def _seat_reds(self, seat):
        """Rouges possedes par le siege (main + melds), tableau bool (3,)."""
        out = np.zeros(N_RED, dtype=bool)
        for k in range(N_RED):
            if self.red_hand[seat][k] and self.hands[seat][RED_TYPES[k]] > 0:
                out[k] = True
        for m in self.melds[seat]:
            for k in m.get("reds", ()):
                out[k] = True
        return out

    def _red_state(self, seat, pending_claim_tile=None):
        hand = [np.array([bool(self.red_hand[s][k]
                               and self.hands[s][RED_TYPES[k]] > 0)
                          for k in range(N_RED)]) for s in range(4)]
        meld = []
        for s in range(4):
            a = np.zeros(N_RED, dtype=bool)
            for m in self.melds[s]:
                for k in m.get("reds", ()):
                    a[k] = True
            meld.append(a)
        pending = -1
        if pending_claim_tile is not None:
            k = RED_INDEX.get(int(pending_claim_tile), -1)
            if k >= 0 and self._last_discard_red == k:
                pending = k
        return {'hand': hand, 'meld': meld,
                'river': [r.copy() for r in self.red_river],
                'indicator': self.red_indicator.copy(),
                'pending': pending}

    def _reveal_kan_dora(self):
        if self._next_dora_pair_idx < N_DORA_PAIRS:
            ind, ura = self._dora_pairs[self._next_dora_pair_idx]
            self.dora_indicators.append(self._reveal_indicator(ind))
            self.ura_dora_indicators.append(self._type_of(ura))
            self._next_dora_pair_idx += 1

    def _should_declare_ron(self, seat, win_result):
        scores = list(self.ctx.get("scores", [0.0] * 4))
        round_idx = int(self.ctx.get("round_idx", 0))
        n_rounds = 4   # [NOTE] env tonpuu-only ; 8 pour un usage hanchan

        if round_idx < n_rounds - 1:
            return True

        cost = win_result.cost
        if isinstance(cost, dict):
            gain = cost.get("total", 0) / 1000.0
        else:
            gain = cost / 1000.0

        new_scores = scores.copy()
        new_scores[seat] += gain

        # [FIX] refuser seulement si on RESTE dernier apres le gain.
        # (l'ancien code refusait si on ne devenait pas 1er -- faux)
        others = [new_scores[o] for o in range(4) if o != seat]
        if new_scores[seat] <= max(others) and new_scores[seat] <= min(others):
            return False
        return True

    def _draw_replacement(self):
        if not self.rinshan_pool:
            return None
        tile = self.rinshan_pool.pop()
        if self.wall:
            self.wall.pop()
            self._sacrificed_count += 1
        return tile

    @staticmethod
    def _is_suited(tile):
        return tile < 27

    def _meld_counts(self, seat):
        total = np.zeros(N_TILES, dtype=np.int32)
        for m in self.melds[seat]:
            for tt in m["tiles"]:
                total[tt] += 1
        return total

    def _full_counts(self, seat):
        return self.hands[seat] + self._meld_counts(seat)

    def _shanten_counts(self, seat):
        counts = self._full_counts(seat).copy()
        for m in self.melds[seat]:
            if m["type"] in ("kan_closed", "shouminkan", "kan_open"):
                counts[m["tiles"][0]] -= 1
        return counts

    @staticmethod
    def _others(seat):
        return [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]

    def _is_hand_closed(self, seat):
        return all(m["type"] == "kan_closed" for m in self.melds[seat])

    def _has_yaku_potential(self, seat):
        hand = self.hands[seat]
        melds = self.melds[seat]
        has_terminal_or_honor = False
        suits_with_terminal = set()
        dragon_pair = False
        wind_pair = False
        suits_in_hand = set()
        hand_pair_exists = False
        for t in range(N_TILES):
            c = int(hand[t])
            if c == 0:
                continue
            if t >= 27:
                has_terminal_or_honor = True
                if t in (31, 32, 33) and c >= 2:
                    dragon_pair = True
                if t in (27, 28, 29, 30) and c >= 2:
                    wind_pair = True
            else:
                suits_in_hand.add(t // 9)
                if t % 9 in (0, 8):
                    has_terminal_or_honor = True
                    suits_with_terminal.add(t // 9)
            if c >= 2:
                hand_pair_exists = True

        has_chi = False
        has_yakuhai_meld = False
        suits_in_melds = set()
        melds_only_terminals_or_honors = True
        for m in melds:
            mt = m["type"]
            tiles = m["tiles"]
            if mt == "chi":
                has_chi = True
                melds_only_terminals_or_honors = False
            elif mt in ("pon", "kan_open", "kan_closed", "shouminkan"):
                t0 = tiles[0]
                if t0 >= 27:
                    has_terminal_or_honor = True
                    has_yakuhai_meld = True
                elif t0 % 9 in (0, 8):
                    has_terminal_or_honor = True
                    suits_with_terminal.add(t0 // 9)
                else:
                    melds_only_terminals_or_honors = False
            for tt in tiles:
                if tt < 27:
                    suits_in_melds.add(tt // 9)

        if not has_terminal_or_honor:
            return True
        if has_yakuhai_meld or dragon_pair or wind_pair:
            return True
        if not has_chi and hand_pair_exists:
            return True
        if not has_chi:
            for m in melds:
                if m["type"] in ("pon", "kan_open", "kan_closed", "shouminkan"):
                    return True
        if len(suits_in_hand | suits_in_melds) <= 1:
            return True
        chantai_ok = True
        for t in range(27):
            if hand[t] > 0 and t % 9 not in (0, 8):
                chantai_ok = False
                break
        if chantai_ok and melds_only_terminals_or_honors:
            return True
        return False

    def _build_136_representation(self, seat):
        # [ROUGES] ids 136 : le rouge d'un 5 a l'id 4*t (16 / 52 / 88 =
        # AKA_DORA_LIST de la lib). Sans rouge on demarre a 4*t+1, sinon
        # tout 5 compterait comme aka. Les ids d'un type restent contigus.
        next_id = [4 * t for t in range(N_TILES)]
        if self.use_red:
            reds = self._seat_reds(seat)
            total = self._full_counts(seat)
            for k, t in enumerate(RED_TYPES):
                # 4 copies d'un 5 => le rouge y est forcement (evite de
                # deborder sur l'id du type suivant si l'etat est injecte)
                if not reds[k] and total[t] < 4:
                    next_id[t] += 1
        used = []

        def take_id(t):
            tid = next_id[t]
            next_id[t] += 1
            used.append(tid)
            return tid

        meld_objs = []
        for m in self.melds[seat]:
            ids = [take_id(tt) for tt in m["tiles"]]
            opened = m["type"] != "kan_closed"
            meld_objs.append(Meld(meld_type=_MELD_TYPE_MAP[m["type"]],
                                   tiles=ids, opened=opened))
        for t in range(N_TILES):
            for _ in range(int(self.hands[seat][t])):
                take_id(t)
        tiles_136 = sorted(used)
        last_id = [x - 1 for x in next_id]       # dernier id attribue par type
        return tiles_136, meld_objs, last_id

    def _evaluate_hand(self, seat, win_tile_type, is_tsumo, already_in_hand,
                        is_rinshan=False, is_chankan=False,
                        is_haitei=False, is_houtei=False):
        win_red_k = RED_INDEX.get(win_tile_type, -1)
        win_red_added = (not already_in_hand and self.use_red and win_red_k >= 0
                         and self._last_discard_red == win_red_k
                         and not self.red_hand[seat][win_red_k])
        if not already_in_hand:
            self.hands[seat][win_tile_type] += 1
            if win_red_added:
                # [ROUGES] la tuile du ron / chankan est le rouge : il compte
                self.red_hand[seat][win_red_k] = True
        try:
            if shanten_of(self._shanten_counts(seat).tolist()) != -1:
                return None
            tiles_136, melds_136, last_id = self._build_136_representation(seat)
            win_tile_136 = last_id[win_tile_type]
            dora_ind_136 = [t * 4 for t in self.dora_indicators]
            ura_ind_136 = ([t * 4 for t in self.ura_dora_indicators]
                            if self.riichi[seat] else None)
            first_draw = (is_tsumo and not self._any_call
                           and self._discard_count[seat] == 0)
            config = HandConfig(
                is_tsumo=is_tsumo, is_riichi=self.riichi[seat],
                is_ippatsu=bool(self.riichi[seat] and self._ippatsu[seat]),
                is_daburu_riichi=bool(self.riichi[seat] and self._double_riichi[seat]),
                is_rinshan=is_rinshan, is_chankan=is_chankan,
                is_haitei=is_haitei, is_houtei=is_houtei,
                is_tenhou=bool(first_draw and seat == 0),
                is_chiihou=bool(first_draw and seat != 0),
                player_wind=SEAT_WINDS[seat],
                round_wind=SEAT_WINDS[min(self.ctx["round_wind_idx"], 3)],
                kyoutaku_number=self.riichi_deposits,
                options=OptionalRules(has_open_tanyao=True,
                                      has_aka_dora=self.use_red,
                                      has_double_yakuman=True),
            )
            result = HandCalculator.estimate_hand_value(
                tiles_136, win_tile_136, melds=melds_136,
                dora_indicators=dora_ind_136,
                ura_dora_indicators=ura_ind_136, config=config,
            )
            return result if result.error is None else None
        finally:
            if not already_in_hand:
                self.hands[seat][win_tile_type] -= 1
                if win_red_added:
                    self.red_hand[seat][win_red_k] = False

    def _can_declare_win(self, seat, win_tile_type, is_tsumo, already_in_hand, **flags):
        return self._evaluate_hand(seat, win_tile_type, is_tsumo,
                                    already_in_hand, **flags)

    def _is_furiten(self, seat):
        if self._furiten_temp[seat]:
            return True
        base_counts = self._shanten_counts(seat)
        if int(base_counts.sum()) >= 14:
            return False
        if shanten_of(base_counts.tolist()) != 0:
            return False
        for t in range(N_TILES):
            if max(self.discards[seat][t], self.river_all[seat][t]) > 0:
                trial = base_counts.copy()
                trial[t] += 1
                if shanten_of(trial.tolist()) == -1:
                    return True
        return False

    def _check_abortive_draw(self):
        if self._abortive_riichi_count >= 4:
            return "abortive_4riichi"
        if self._abortive_kan_count >= 4:
            return "abortive_4kan"
        return None

    def _abortive_draw_rewards(self, exclude=None):
        # [FIX BATONS RIICHI] Avant : +1.0 de "remboursement" a chaque
        # riichi, pour un baton jamais debite -> argent cree de rien. Les
        # regles reelles ne remboursent pas : les batons restent sur la
        # table (reportes a la manche suivante). Le cout est deja dans
        # stick_paid, applique par _final. `exclude` garde pour compat.
        return self._zero_rewards()

    def _chow_variants(self, seat, tile):
        if not self._is_suited(tile):
            return None
        hand = self.hands[seat]
        col = tile % 9
        lower = col >= 2 and hand[tile - 2] >= 1 and hand[tile - 1] >= 1
        mid = 1 <= col <= 7 and hand[tile - 1] >= 1 and hand[tile + 1] >= 1
        upper = col <= 6 and hand[tile + 1] >= 1 and hand[tile + 2] >= 1
        if not (lower or mid or upper):
            return None
        return np.array([1.0, float(lower), float(mid), float(upper)],
                         dtype=np.float32)

    def _apply_pong(self, seat, tile):
        self.hands[seat][tile] -= 2
        reds = self._claim_reds(seat, tile, [tile])
        self.melds[seat].append({"type": "pon", "tiles": [tile, tile, tile],
                                 "reds": reds})
        self._stats["pon_calls"] += 1
        self._any_call = True
        self._ippatsu = [False] * 4
        self._stats["open_calls"] += 1

    def _apply_chow(self, seat, tile, variant):
        if variant == CHOW_LOWER:
            used = (tile - 2, tile - 1)
        elif variant == CHOW_MID:
            used = (tile - 1, tile + 1)
        else:
            used = (tile + 1, tile + 2)
        for t in used:
            self.hands[seat][t] -= 1
        reds = self._claim_reds(seat, tile, list(used))
        self.melds[seat].append({"type": "chi",
                                 "tiles": sorted([tile, used[0], used[1]]),
                                 "reds": reds})
        self._stats["chi_calls"] += 1
        self._any_call = True
        self._ippatsu = [False] * 4
        self._stats["open_calls"] += 1

    def _apply_daiminkan(self, seat, tile):
        self.hands[seat][tile] -= 3
        reds = self._claim_reds(seat, tile, [tile])
        self.melds[seat].append({"type": "kan_open", "tiles": [tile] * 4,
                                 "reds": reds})
        self._any_call = True
        self._ippatsu = [False] * 4

    def _find_closed_kong(self, seat):
        for t in range(N_TILES):
            if self.hands[seat][t] >= 4:
                return t
        return None

    def _find_shouminkan(self, seat):
        for t in range(N_TILES):
            if self.hands[seat][t] >= 1:
                for m in self.melds[seat]:
                    if m["type"] == "pon" and m["tiles"][0] == t:
                        return t
        return None

    def _apply_closed_kong(self, seat, tile):
        self.hands[seat][tile] -= 4
        reds = self._claim_reds(seat, None, [tile])
        self.melds[seat].append({"type": "kan_closed", "tiles": [tile] * 4,
                                 "reds": reds})
        self._any_call = True
        self._ippatsu = [False] * 4
        self._reveal_kan_dora()
        self._abortive_kan_count += 1

    def _apply_shouminkan(self, seat, tile):
        self.hands[seat][tile] -= 1
        added = self._claim_reds(seat, None, [tile])
        # la tuile ajoutee est celle qu'un adversaire peut chankan
        self._last_discard_red = added[0] if added else -1
        for m in self.melds[seat]:
            if m["type"] == "pon" and m["tiles"][0] == tile:
                m["type"] = "shouminkan"
                m["tiles"].append(tile)
                m.setdefault("reds", []).extend(added)
                break
        self._any_call = True
        self._ippatsu = [False] * 4
        self._reveal_kan_dora()
        self._abortive_kan_count += 1

    def _riichi_discard_options(self, seat):
        options = []
        for t in range(N_TILES):
            if self.hands[seat][t] > 0:
                trial = self._shanten_counts(seat).copy()
                trial[t] -= 1
                if shanten_of(trial.tolist()) == 0:
                    options.append(t)
        return options

    def _can_declare_riichi(self, seat):
        if self.riichi[seat] or not self._is_hand_closed(seat) or not self.wall:
            return None
        options = self._riichi_discard_options(seat)
        return options if options else None

    def _unseen_counts(self, seat):
        seen = self.hands[seat].astype(np.int32).copy()
        for p in range(4):
            seen += self._meld_counts(p)
            seen += self.discards[p]
        for ind in self.dora_indicators:
            seen[ind] += 1
        return np.clip(COPIES_PER_TILE - seen, 0, COPIES_PER_TILE)

    def _build_oracle(self, seat):
        """Retourne l'obs oracle (ORACLE_DIM=4, 34) :
            - canal 0 : main adversaire 1 (normalise /4)
            - canal 1 : main adversaire 2
            - canal 2 : main adversaire 3
            - canal 3 : mur cache (normalise /4)
        Conforme a Suphx §3.3 : l'oracle voit les mains adverses + le mur.
        """
        opps = self._others(seat)
        opp_hands = [self.hands[o].astype(np.float32) / COPIES_PER_TILE
                     for o in opps]
        unseen = self._unseen_counts(seat).astype(np.float32)
        hidden_wall = np.clip(
            unseen - sum(self.hands[o] for o in opps),
            0, COPIES_PER_TILE
        ) / COPIES_PER_TILE
        return np.stack(opp_hands + [hidden_wall], axis=0)  # (4, 34)

    def _build_obs(self, seat, pending_claim_tile=None):
        """Retourne (N_CHANNELS, 34) float32 via envs.suphx_features.
        [MODIFIE] Suppression game_format et round_progress.
        rounds_remaining devient absolu.
        """
        scores = list(self.ctx.get("scores", [0.0] * 4))
        round_wind = int(self.ctx.get("round_wind_idx", 0))
        round_index = int(self.ctx.get("round_idx", 0))
        round_id = (round_index % 4) + 1
        normal_round_count = 4  # tonpuu (cet env est tonpuu-only)
        is_extension = round_index >= normal_round_count
        extension_index = max(0, round_index - normal_round_count + 1)
        is_last_normal_round = round_index == normal_round_count - 1

        # [MODIFIE] rounds_remaining ABSOLU
        rounds_remaining = max(normal_round_count - (round_index + 1), 0)

        obs = build_suphx_obs(
            hands=self.hands,
            melds=self.melds,
            discards=self.discards,
            discard_hist=self._discard_hist,
            riichi_flags=self.riichi,
            compute_lookahead_flag=self._compute_lookahead,
            furiten_flag=self._is_furiten(seat),
            dora_indicators=self.dora_indicators,
            wall_remaining=len(self.wall),
            scores=scores,
            dealer_seat=0,
            round_wind=round_wind,
            round_id=round_id,
            honba=int(self.ctx.get("honba", 0)),
            kyoutaku=self.riichi_deposits,
            seat=seat,
            pending_claim_tile=pending_claim_tile,
            # [SUPPRIME] game_format="tonpuu",
            is_last_normal_round=is_last_normal_round,
            is_extension=is_extension,
            extension_index=extension_index,
            normal_round_count=normal_round_count,
            # [SUPPRIME] round_progress=round_index + 1,
            rounds_remaining=rounds_remaining,
            red_state=(self._red_state(seat, pending_claim_tile)
                       if self.use_red else None),
        )
        self._pending_oracle = self._build_oracle(seat)
        return obs
    def _sl_decide(self, seat, head, mask, pending_claim_tile=None):
        """Appelle le modele SL fige pour cette tete.

        Retourne l'action (int) ou None si pas de modele fige pour cette
        tete (fallback : yield dans _round_flow).
        """
        if self.frozen_sl is None or head not in self.frozen_sl:
            return None
        try:
            import torch
            obs = self._build_obs(seat, pending_claim_tile=pending_claim_tile)
            model = self.frozen_sl[head]
            # [FIX DEVICE] Les tetes gelees vivent sur le device du worker
            # (cuda:0 en entrainement GPU) ; l'obs etait creee sur CPU ->
            # "Expected all tensors to be on the same device". L'ancien
            # `except Exception: return None` avalait l'erreur : AUCUNE tete
            # SL (riichi/pon/chow/kong) ne fonctionnait en entrainement GPU.
            dev = next(model.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=dev).unsqueeze(0)
            with torch.no_grad():
                logits = model(obs_t)[0].cpu().numpy()
            # [FIX MASQUE] logits * mask ne masque pas : une action illegale
            # vaut 0 et gagne l'argmax des que tous les logits legaux sont
            # negatifs (ex. chow : variante illegale choisie -> main
            # corrompue). On masque a -inf.
            legal = np.asarray(mask) > 0
            if not legal.any():
                return 0
            masked = np.where(legal, logits, -np.inf)
            return int(masked.argmax())
        except Exception:
            # [FIX ECHEC SILENCIEUX] Avant : return None -> l'env yield une
            # requete que le worker renvoyait avec action=None, et pour CHOW
            # None != CHOW_PASS appliquait un chow illegal. Maintenant :
            # compteur (visible dans la ligne GAME), traceback une seule
            # fois, et action SURE (0 = passer / non).
            import traceback
            self._stats["sl_errors"] = self._stats.get("sl_errors", 0) + 1
            if not getattr(self, "_sl_error_logged", False):
                self._sl_error_logged = True
                print(f"[env] !!! erreur tete SL figee '{head}' "
                      f"(action sure 0 utilisee ; compteur n_sl_errors) :",
                      flush=True)
                traceback.print_exc()
            return 0
        
    @staticmethod
    def _zero_rewards():
        return {s: 0.0 for s in range(4)}

    def _distribute_win_payment(self, winner, discarder, result, give_kyoutaku=True):
        r = self._zero_rewards()
        cost = result.cost
        kb = cost.get("kyoutaku_bonus", 0) or 0
        gain_bonus = kb if give_kyoutaku else 0
        if discarder is not None:
            r[winner] += (cost["total"] - kb + gain_bonus) * REWARD_SCALE
            r[discarder] -= (cost["total"] - kb) * REWARD_SCALE
            return r
        winner_is_dealer = (winner == 0)
        for o in self._others(winner):
            amount = cost["main"] if (winner_is_dealer or o == 0) else cost["additional"]
            r[o] -= amount * REWARD_SCALE
        r[winner] += (cost["total"] - kb + gain_bonus) * REWARD_SCALE
        return r

    def _noten_payment_rewards(self):
        r = self._zero_rewards()
        tenpai = [s for s in range(4)
                  if shanten_of(self._shanten_counts(s).tolist()) == 0]
        noten = [s for s in range(4) if s not in tenpai]
        if 0 < len(tenpai) < 4:
            gain_each = NOTEN_PAYMENT_POOL / len(tenpai)
            loss_each = NOTEN_PAYMENT_POOL / len(noten)
            for s in tenpai:
                r[s] += gain_each * REWARD_SCALE
            for s in noten:
                r[s] -= loss_each * REWARD_SCALE
        # [RETRAIT SHAPING 2026-11] l'ancien terme -0.1*shanten par joueur
        # (EXHAUSTIVE_SHANTEN_COEF) a ete supprime : paiements de noten
        # exacts, game_scores conservatifs (somme nulle), plus de derive
        # sur les seuils 30k/tobi/rangs. Voir remove_exhaustive_shanten_
        # penalty.patch.
        return r

    def _danger_labels(self, seat):
        """[DANGER] danger[t] = 1 si un adversaire tenpai (hors furiten)
        attend la tuile t.

        Utilise les mains cachees : la penalite RL (--danger-penalty)
        s'applique a la decision qui defausse une tuile dangereuse.
        Ignore le yaku (approximation : certains ron sont impossibles
        faute de yaku, la penalite reste du bon cote).
        Cout : ~3 shanten par requete + 34 par adversaire tenpai.
        """
        danger = np.zeros(N_TILES, dtype=np.float32)
        for o in self._others(seat):
            base = self._shanten_counts(o)
            if int(base.sum()) != 13:
                continue
            if shanten_of(base.tolist()) != 0:
                continue
            if self._is_furiten(o):
                continue
            for t in range(N_TILES):
                if base[t] >= 4:
                    continue
                trial = base.copy()
                trial[t] += 1
                if shanten_of(trial.tolist()) == -1:
                    danger[t] = 1.0
        return danger

    def _request(self, seat, action_type, obs, mask):
        req = {"seat": seat, "action_type": action_type, "obs": obs, "mask": mask,
               "oracle": self._pending_oracle}
        if self.compute_danger and action_type == DISCARD:
            req["danger"] = self._danger_labels(seat)
        return req

    @staticmethod
    def _combine(imm_reward, acting_seat, next_request):
        return {"imm_reward": imm_reward, "acting_seat": acting_seat,
                "round_done": False, "final_rewards": None,
                "next_request": next_request}

    def _final(self, imm_reward, acting_seat, final_rewards, outcome):
        if acting_seat is not None:
            final_rewards[acting_seat] = final_rewards.get(acting_seat, 0.0) + imm_reward
        # [FIX STATS ROUNDS] L'increment etait en code mort dans _red_state
        # (place apres le return) : get_stats()["rounds"] restait bloque a 0.
        # _final est l'unique point de passage de toute fin de manche
        # (tsumo, ron, multi_ron, chankan, exhaustive, abortives), c'est donc
        # ici que se compte une manche jouee.
        self._stats["rounds"] += 1
        # [FIX BATONS RIICHI] Le cout du baton (-1000 a la declaration) etait
        # range dans `pending`, ecrase des le discard suivant, donc jamais
        # preleve : un riichi etait GRATUIT et les batons empoches par le
        # gagnant n'avaient jamais ete payes (somme des scores non nulle).
        # Preleve ici, pour TOUTES les issues de manche (tsumo, ron, chankan,
        # exhaustive, abortives), independamment de `pending`.
        for s in range(4):
            if self.stick_paid[s]:
                final_rewards[s] = final_rewards.get(s, 0.0) - self.stick_paid[s]
        return {"imm_reward": None, "acting_seat": None,
                "round_done": True, "final_rewards": final_rewards,
                "next_request": None, "outcome": outcome}

    # ------------------------------------------------------------------ #
    def _round_flow(self):
        self._deal()
        seat = 0
        pending = (0.0, None)
        just_called = False
        replacement_tile = None
        kuikae_forbidden = None

        shanten_discard_coef = _get_shanten_discard_coef()
        shanten_call_coef = _get_shanten_call_coef()

        while True:
            forced_discard_mask = None
            drawn_tile = None

            if not just_called:
                if replacement_tile is not None:
                    tile = self._to_hand(seat, replacement_tile)
                    replacement_tile = None
                    is_rinshan = True
                else:
                    if not self.wall:
                        self._stats["exhaustive"] += 1
                        final_rewards = self._noten_payment_rewards()
                        return self._final(pending[0], pending[1], final_rewards,
                                            "exhaustive")
                    tile = self._to_hand(seat, self.wall.pop())
                    is_rinshan = False

                drawn_tile = tile
                if not self.riichi[seat]:
                    self._furiten_temp[seat] = False
                is_haitei = (len(self.wall) == 0)

                win_result = self._can_declare_win(
                    seat, tile, is_tsumo=True, already_in_hand=True,
                    is_rinshan=is_rinshan, is_haitei=is_haitei)
                if win_result is not None:
                    self._stats["tsumo"] += 1
                    self._record_win_yaku(seat, win_result)
                    final_rewards = self._distribute_win_payment(seat, None, win_result)
                    return self._final(pending[0], pending[1], final_rewards, "tsumo")

                self._ippatsu[seat] = False

                if not self.riichi[seat]:
                    closed_kong_tile = self._find_closed_kong(seat)
                    shouminkan_tile = (None if closed_kong_tile is not None
                                        else self._find_shouminkan(seat))
                    kong_tile = (closed_kong_tile if closed_kong_tile is not None
                                  else shouminkan_tile)
                    kong_kind = ("closed" if closed_kong_tile is not None
                                  else ("shouminkan" if shouminkan_tile is not None
                                        else None))

                    if kong_tile is not None and self.rinshan_pool:
                        mask_k = np.array([1.0, 1.0], dtype=np.float32)
                        # [RL] Decide par SL fige si disponible
                        kong_action = self._sl_decide(seat, 'kong', mask_k,
                                                       pending_claim_tile=None)
                        if kong_action is None:
                            obs_k = self._build_obs(seat)
                            request = self._request(seat, KONG, obs_k, mask_k)
                            kong_action = yield self._combine(pending[0], pending[1], request)


                        if kong_action == KONG_YES:
                            if kong_kind == "closed":
                                self._stats["ankan"] += 1
                            else:
                                self._stats["shouminkan"] += 1

                            sh_before = shanten_of(self._shanten_counts(seat).tolist())
                            if kong_kind == "closed":
                                self._apply_closed_kong(seat, kong_tile)
                                pending = (
                                    shanten_call_coef *
                                    float(sh_before - shanten_of(self._shanten_counts(seat).tolist()))
                                    - ANKAN_PENALTY, seat)
                            else:
                                self._apply_shouminkan(seat, kong_tile)
                                pending = (
                                    shanten_call_coef *
                                    float(sh_before - shanten_of(self._shanten_counts(seat).tolist()))
                                    - OPEN_KAN_PENALTY, seat)
                                self._last_discarder = seat
                                chankan_results = {
                                    o: self._can_declare_win(
                                        o, kong_tile, is_tsumo=False,
                                        already_in_hand=False, is_chankan=True)
                                    for o in self._others(seat)
                                }
                                chankan_candidates = {
                                    o: res for o, res in chankan_results.items()
                                    if res is not None and not self._is_furiten(o)
                                }
                                if chankan_candidates:
                                    chankan_ronners = {}
                                    prev_pending = pending
                                    for o in list(chankan_candidates.keys()):
                                        if self.frozen_sl is not None:
                                            if self._should_declare_ron(o, chankan_candidates[o]):
                                                chankan_ronners[o] = chankan_candidates[o]
                                                prev_pending = (0.0, o)
                                            else:
                                                self._furiten_temp[o] = True
                                                self._stats["ron_refused"] += 1
                                                prev_pending = (-RON_REFUSE_PENALTY, o)
                                        else:
                                            obs_r = self._build_obs(o, pending_claim_tile=kong_tile)
                                            mask_r = np.array([1.0, 1.0], dtype=np.float32)
                                            req_r = self._request(o, RON, obs_r, mask_r)
                                            ron_action = yield self._combine(
                                                prev_pending[0], prev_pending[1], req_r)
                                            if ron_action == RON_YES:
                                                chankan_ronners[o] = chankan_candidates[o]
                                                prev_pending = (0.0, o)
                                            else:
                                                self._furiten_temp[o] = True
                                                self._stats["ron_refused"] += 1
                                                prev_pending = (-RON_REFUSE_PENALTY, o)

                                    # [FIX CHANKAN] Resolution du ron sur kakan :
                                    # bloquait ici -- le kan-declareur PAYE le ron,
                                    # la manche se termine.
                                    if chankan_ronners:
                                        if len(chankan_ronners) > 1:
                                            self._stats["multi_ron"] += 1
                                        else:
                                            self._stats["ron"] += 1
                                        final_rewards = self._zero_rewards()
                                        for k_o, (o, res) in enumerate(chankan_ronners.items()):
                                            payment = self._distribute_win_payment(
                                                o, seat, res, give_kyoutaku=(k_o == 0))
                                            self._record_win_yaku(o, res)
                                            for s in range(4):
                                                final_rewards[s] += payment[s]
                                        outcome = "multi_ron" if len(chankan_ronners) > 1 else "ron"
                                        return self._final(prev_pending[0], prev_pending[1],
                                                           final_rewards, outcome)

                            abort = self._check_abortive_draw()

                            abort = self._check_abortive_draw()
                            if abort:
                                self._stats[abort] += 1
                                return self._final(pending[0], pending[1],
                                                    self._abortive_draw_rewards(), abort)

                            replacement_tile = self._draw_replacement()
                            if replacement_tile is None:
                                self._stats["exhaustive"] += 1
                                final_rewards = self._noten_payment_rewards()
                                return self._final(pending[0], pending[1],
                                                    final_rewards, "exhaustive")
                            continue
                        else:
                            pending = (0.0, seat)

                riichi_options = self._can_declare_riichi(seat)
                if riichi_options is not None:
                    mask_r = np.array([1.0, 1.0], dtype=np.float32)
                    # [RL] Decide par SL fige si disponible
                    riichi_action = self._sl_decide(seat, 'riichi', mask_r,
                                                     pending_claim_tile=None)
                    if riichi_action is None:
                        obs_r = self._build_obs(seat)
                        request = self._request(seat, RIICHI, obs_r, mask_r)
                        riichi_action = yield self._combine(pending[0], pending[1], request)

                    if riichi_action == RIICHI_YES:
                        self._stats["riichi"] += 1
                        self._double_riichi[seat] = (
                            self._discard_count[seat] == 0 and not self._any_call)
                        self._ippatsu[seat] = True
                        self.riichi[seat] = True
                        self.riichi_deposits += 1
                        # [FIX BATONS RIICHI] cout enregistre ici, preleve
                        # en fin de manche par _final (et non plus dans
                        # `pending`, ecrase au discard suivant).
                        self.stick_paid[seat] += RIICHI_STICK_NET_COST * REWARD_SCALE
                        self._abortive_riichi_count += 1
                        forced_discard_mask = np.zeros(N_TILES, dtype=np.float32)
                        for t in riichi_options:
                            forced_discard_mask[t] = 1.0
                        pending = (RIICHI_DECLARE_BONUS, seat)
                        abort = self._check_abortive_draw()
                        if abort:
                            self._stats[abort] += 1
                            return self._final(0.0, None,
                                                self._abortive_draw_rewards(exclude=seat),
                                                abort)
                    else:
                        pending = (0.0, seat)

            just_called = False

            # Discard
            if self.riichi[seat] and forced_discard_mask is None:
                action = tile
                self.hands[seat][action] -= 1
                self.discards[seat][action] += 1
                self.river_all[seat][action] += 1
                # [ROUGES] tsumogiri force en riichi : si la tuile piochee
                # est le rouge, c'est lui qui part.
                self._after_discard(seat, action, tsumogiri_red=self._drawn_red)
                discarded_tile, discarder = action, seat
            else:
                mask_d = (forced_discard_mask if forced_discard_mask is not None
                          else (self.hands[seat] > 0).astype(np.float32))
                if kuikae_forbidden:
                    mask_k2 = mask_d.copy()
                    for kt in kuikae_forbidden:
                        mask_k2[kt] = 0.0
                    if mask_k2.sum() > 0:
                        mask_d = mask_k2
                obs_d = self._build_obs(seat)
                request = self._request(seat, DISCARD, obs_d, mask_d)
                action = yield self._combine(pending[0], pending[1], request)

                shanten_before = shanten_of(self._shanten_counts(seat).tolist())
                self.hands[seat][action] -= 1
                self.discards[seat][action] += 1
                self.river_all[seat][action] += 1
                self._after_discard(seat, action)
                shanten_after = shanten_of(self._shanten_counts(seat).tolist())
                pending = (shanten_discard_coef *
                            float(shanten_before - shanten_after) - STEP_PENALTY, seat)
                discarded_tile, discarder = action, seat

            kuikae_forbidden = None
            self._discard_hist[discarder].append(
                (int(discarded_tile),
                 bool(drawn_tile is not None and discarded_tile == drawn_tile)))
            self._discard_count[discarder] += 1
            self._last_discarder = discarder
            for o in range(4):
                if self.riichi[o] and o != discarder:
                    self._safe_after_riichi[o][discarded_tile] = 1.0

            # Ron
            is_houtei = (len(self.wall) == 0)
            ron_results = {
                o: self._can_declare_win(o, discarded_tile, is_tsumo=False,
                                          already_in_hand=False, is_houtei=is_houtei)
                for o in self._others(discarder)
            }
            ron_candidates = {
                o: res for o, res in ron_results.items()
                if res is not None and not self._is_furiten(o)
            }

            if len(ron_candidates) >= 3:
                self._stats["abortive_3ron"] += 1
                return self._final(pending[0], pending[1],
                                    self._abortive_draw_rewards(), "abortive_3ron")

            if ron_candidates:
                ronners = {}
                prev_pending = pending
                for o in list(ron_candidates.keys()):
                    # [FIX] RL : decide en interne via regle rule-based,
                    # sinon yield (SL/entrainement manuel).
                    if self.frozen_sl is not None:
                        if self._should_declare_ron(o, ron_candidates[o]):
                            ronners[o] = ron_candidates[o]
                            prev_pending = (0.0, o)
                        else:
                            self._furiten_temp[o] = True
                            self._stats["ron_refused"] += 1
                            prev_pending = (-RON_REFUSE_PENALTY, o)
                    else:
                        obs_ron = self._build_obs(o, pending_claim_tile=discarded_tile)
                        mask_ron = np.array([1.0, 1.0], dtype=np.float32)
                        request = self._request(o, RON, obs_ron, mask_ron)
                        ron_action = yield self._combine(prev_pending[0], prev_pending[1], request)
                        if ron_action == RON_YES:
                            ronners[o] = ron_candidates[o]
                            prev_pending = (0.0, o)
                        else:
                            self._furiten_temp[o] = True
                            self._stats["ron_refused"] += 1
                            prev_pending = (-RON_REFUSE_PENALTY, o)

                if ronners:
                    if len(ronners) > 1:
                        self._stats["multi_ron"] += 1
                    else:
                        self._stats["ron"] += 1
                    final_rewards = self._zero_rewards()
                    for k_o, (o, res) in enumerate(ronners.items()):
                        payment = self._distribute_win_payment(
                            o, discarder, res, give_kyoutaku=(k_o == 0))
                        self._record_win_yaku(o, res)
                        for s in range(4):
                            final_rewards[s] += payment[s]
                    outcome = "multi_ron" if len(ronners) > 1 else "ron"
                    return self._final(prev_pending[0], prev_pending[1],
                                        final_rewards, outcome)
                pending = prev_pending

                
            # Daiminkan
            daiminkan_seat = next(
                (o for o in self._others(discarder)
                 if not self.riichi[o] and self.hands[o][discarded_tile] >= 3), None)
            claimed = False
            if daiminkan_seat is not None and self.rinshan_pool:
                mask_k = np.array([1.0, 1.0], dtype=np.float32)
                # [RL] Decide par SL fige si disponible
                kong_action = self._sl_decide(daiminkan_seat, 'kong', mask_k,
                                               pending_claim_tile=discarded_tile)
                if kong_action is None:
                    obs_k = self._build_obs(daiminkan_seat,
                                             pending_claim_tile=discarded_tile)
                    request = self._request(daiminkan_seat, KONG, obs_k, mask_k)
                    kong_action = yield self._combine(pending[0], pending[1], request)

                if kong_action == KONG_YES:
                    self._stats["daiminkan"] += 1
                    sh_before = shanten_of(self._shanten_counts(daiminkan_seat).tolist())
                    self.discards[discarder][discarded_tile] -= 1
                    self._apply_daiminkan(daiminkan_seat, discarded_tile)
                    sh_after = shanten_of(self._shanten_counts(daiminkan_seat).tolist())
                    pending = (shanten_call_coef * float(sh_before - sh_after),
                                daiminkan_seat)
                    self._reveal_kan_dora()
                    self._abortive_kan_count += 1
                    pending = (pending[0] - OPEN_MELD_PENALTY - OPEN_KAN_PENALTY,
                                pending[1])
                    if not self._has_yaku_potential(daiminkan_seat):
                        self._stats["no_yaku_calls"] += 1
                        pending = (pending[0] - NO_YAKU_CALL_PENALTY, pending[1])
                    abort = self._check_abortive_draw()
                    if abort:
                        self._stats[abort] += 1
                        return self._final(pending[0], pending[1],
                                            self._abortive_draw_rewards(), abort)
                    replacement_tile = self._draw_replacement()
                    if replacement_tile is None:
                        self._stats["exhaustive"] += 1
                        final_rewards = self._noten_payment_rewards()
                        return self._final(pending[0], pending[1],
                                            final_rewards, "exhaustive")
                    seat = daiminkan_seat
                    just_called = False
                    claimed = True
                    continue
                else:
                    pending = (0.0, daiminkan_seat)

            # Pong
            if not claimed:
                pong_seat = next(
                    (o for o in self._others(discarder)
                     if not self.riichi[o] and not self.force_closed[o]
                     and self.hands[o][discarded_tile] >= 2), None)
                if pong_seat is not None:
                    mask_p = np.array([1.0, 1.0], dtype=np.float32)
                    pong_action = self._sl_decide(pong_seat, 'pong', mask_p,
                                                   pending_claim_tile=discarded_tile)
                    if pong_action is None:
                        obs_p = self._build_obs(pong_seat,
                                                 pending_claim_tile=discarded_tile)
                        request = self._request(pong_seat, PONG, obs_p, mask_p)
                        pong_action = yield self._combine(pending[0], pending[1], request)

                    if pong_action == 1:
                        sh_before = shanten_of(self._shanten_counts(pong_seat).tolist())
                        self.discards[discarder][discarded_tile] -= 1
                        self._apply_pong(pong_seat, discarded_tile)
                        sh_after = shanten_of(self._shanten_counts(pong_seat).tolist())
                        pending = (shanten_call_coef *
                                    float(sh_before - sh_after) - OPEN_MELD_PENALTY,
                                    pong_seat)
                        if not self._has_yaku_potential(pong_seat):
                            self._stats["no_yaku_calls"] += 1
                            pending = (pending[0] - NO_YAKU_CALL_PENALTY, pending[1])
                        seat, claimed, just_called = pong_seat, True, True
                        kuikae_forbidden = {discarded_tile}
                    else:
                        pending = (0.0, pong_seat)

            # Chow
            if not claimed:
                chow_seat = (discarder + 1) % 4
                chow_mask = (self._chow_variants(chow_seat, discarded_tile)
                             if not self.riichi[chow_seat] and not self.force_closed[chow_seat]
                             else None)
                if chow_mask is not None:
                    chow_action = self._sl_decide(chow_seat, 'chow', chow_mask,
                                                   pending_claim_tile=discarded_tile)
                    if chow_action is None:
                        obs_c = self._build_obs(chow_seat,
                                                 pending_claim_tile=discarded_tile)
                        request = self._request(chow_seat, CHOW, obs_c, chow_mask)
                        chow_action = yield self._combine(pending[0], pending[1], request)

                    if chow_action != CHOW_PASS:
                        sh_before = shanten_of(self._shanten_counts(chow_seat).tolist())
                        self.discards[discarder][discarded_tile] -= 1
                        self._apply_chow(chow_seat, discarded_tile, chow_action)
                        sh_after = shanten_of(self._shanten_counts(chow_seat).tolist())
                        pending = (shanten_call_coef *
                                    float(sh_before - sh_after) - OPEN_MELD_PENALTY,
                                    chow_seat)
                        if not self._has_yaku_potential(chow_seat):
                            self._stats["no_yaku_calls"] += 1
                            pending = (pending[0] - NO_YAKU_CALL_PENALTY, pending[1])
                        seat, claimed, just_called = chow_seat, True, True
                        kuikae_forbidden = {discarded_tile}
                        col_c = discarded_tile % 9
                        if chow_action == CHOW_LOWER and col_c - 3 >= 0:
                            kuikae_forbidden.add(discarded_tile - 3)
                        elif chow_action == CHOW_UPPER and col_c + 3 <= 8:
                            kuikae_forbidden.add(discarded_tile + 3)
                    else:
                        pending = (0.0, chow_seat)

            if not claimed:
                seat = (discarder + 1) % 4