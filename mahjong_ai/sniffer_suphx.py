"""
Sniffer Mahjong Soul - compatible avec les 5 modeles SL Suphx + pMCPA.

[FIX 2025-XX] Ajout de _an_gang_add_gang manquante (bug shouminkan qui
laissait la 4eme tuile en main).

[pMCPA] Integration de Parametric Monte-Carlo Policy Adaptation (§3.4) :
    - Au debut de chaque manche : reset sur poids offline + K simulations
      + finetune pondere par R(tau)
    - Pendant la manche : utilise theta_a (policy adaptee) au lieu de theta_o
    - A la manche suivante : reset sur theta_o

Variables d'environnement :
    SL_CHECKPOINT_DIR : dossier contenant discard.pt, riichi.pt, ...
                        (defaut : rl_final)
    SL_ENABLE         : "1" (defaut) ou "0"
    SL_USE_LOOKAHEAD  : "0" (defaut) pour skip le DFS look-ahead
    SNIFFER_TRACE     : "1" (defaut)
    PMCPA_ENABLE      : "1" pour activer pMCPA (defaut : "0")
    PMCPA_K           : nombre de simulations (defaut : 20)
    PMCPA_LR          : learning rate d'adaptation (defaut : 1e-4)
    PMCPA_GRAD_STEPS  : nombre de gradient updates (defaut : 3)
    PMCPA_TIMEOUT     : timeout adaptation en secondes (defaut : 60)
"""
import os
import struct
import time
from datetime import datetime

import numpy as np
import torch
from google.protobuf.json_format import MessageToDict
from ms import protocol_pb2 as pb

from mahjong_ai.envs.suphx_features import (
    build_suphx_obs, N_CHANNELS, N_TILES as _N_TILES_SL,
    RED_TYPES as _RED_TYPES,
)
from mahjong_ai.envs.multi_agent import (
    DISCARD, PONG, CHOW, KONG, RIICHI,
    CHOW_PASS, CHOW_LOWER, CHOW_MID, CHOW_UPPER,
)

N_TILES = 34
HAND_SIZE = 13
INITIAL_WALL_SIZE = 70
COPIES_PER_TILE = 4

CHECKPOINT_DIR = os.environ.get("SL_CHECKPOINT_DIR", "checkpoints/")
USE_SL = os.environ.get("SL_ENABLE", "1") == "1"
USE_LOOKAHEAD = os.environ.get("SL_USE_LOOKAHEAD", "1") == "1"

# [pMCPA] Config via variables d'environnement
PMCPA_ENABLE = os.environ.get("PMCPA_ENABLE", "1") == "1"
PMCPA_K = int(os.environ.get("PMCPA_K", "30"))
PMCPA_LR = float(os.environ.get("PMCPA_LR", "1e-4"))
PMCPA_GRAD_STEPS = int(os.environ.get("PMCPA_GRAD_STEPS", "3"))
PMCPA_TIMEOUT = float(os.environ.get("PMCPA_TIMEOUT", "60"))

KEYS = [0x84, 0x5E, 0x4E, 0x42, 0x39, 0xA2, 0x1F, 0x60, 0x1C]

HONOR_MAP = {
    "1z": 27, "2z": 28, "3z": 29, "4z": 30, "5z": 31, "6z": 32, "7z": 33,
    "E": 27, "S": 28, "W": 29, "N": 30,
    "Wh": 31, "Gr": 32, "Rd": 33,
    "P": 31, "F": 32, "C": 33,
}

OP_CHI = 2
OP_PON = 3
OP_KAN = 4
OP_RIICHI = 5
OP_RON = 8

SCORE_BASELINE = 25000.0
TRACE_ALL_ACTIONS = os.environ.get("SNIFFER_TRACE", "1") == "1"


# ═══════════════════════════════════════════════════════════════════════
# CONVERSION DES TUILES
# ═══════════════════════════════════════════════════════════════════════

def tile_to_34(t):
    if t is None:
        return -1
    if isinstance(t, int):
        return t // 4 if t >= 34 else t
    s = str(t).strip()
    if not s:
        return -1
    if len(s) >= 2 and s[-1] in ("m", "p", "s") and s[:-1].isdigit():
        n = int(s[:-1])
        if n == 0:
            n = 5
        if not (1 <= n <= 9):
            return -1
        offset = {"m": 0, "p": 9, "s": 18}[s[-1]]
        return offset + (n - 1)
    if s in HONOR_MAP:
        return HONOR_MAP[s]
    s_up = s.upper()
    for k, v in HONOR_MAP.items():
        if k.upper() == s_up:
            return v
    if s.isdigit():
        n = int(s)
        return n // 4 if n >= 34 else n
    return -1


# [ROUGES] Mahjong Soul envoie les 5 rouges sous la forme "0m" / "0p" / "0s"
_RED_STRS = {"0m": 0, "0p": 1, "0s": 2, "5mr": 0, "5pr": 1, "5sr": 2}


def tile_red_index(t):
    """0/1/2 si `t` est un 5 rouge (5m/5p/5s), sinon -1."""
    if t is None or isinstance(t, int):
        return -1
    return _RED_STRS.get(str(t).strip().lower(), -1)


def _raw_tile_of(data):
    """Chaine brute de la tuile d'un message (pour retrouver le rouge)."""
    for key in ("tile", "pai", "tiles"):
        v = data.get(key)
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, str) and v.strip():
            return v
    return None


def _extract_single_tile(data):
    """Extrait une seule tuile depuis un dict MJAI."""
    for key in ("tile", "pai", "tiles", "consumed"):
        if key not in data:
            continue
        v = data[key]
        if isinstance(v, list):
            for x in v:
                idx = tile_to_34(x)
                if idx >= 0:
                    return idx, key
        else:
            idx = tile_to_34(v)
            if idx >= 0:
                return idx, key
    return -1, None


def disp_hand(hand, red_hand):
    """Main triee ; la copie rouge d'un 5 s'affiche 5mr / 5pr / 5sr."""
    out, shown = [], set()
    for t in sorted(hand):
        k = _RED_TYPES.index(t) if t in _RED_TYPES else -1
        if k >= 0 and red_hand[k] and k not in shown:
            shown.add(k)
            out.append(tile_str(t) + "r")
        else:
            out.append(tile_str(t))
    return " ".join(out)


def disp_tile(state, t):
    """5m -> '5mr' (ou '5mr/5m' si j'ai aussi une copie normale) quand
    j'ai le rouge en main ; l'IA choisit un TYPE de tuile, pas la copie."""
    s = tile_str(t)
    if t in _RED_TYPES and state.red_hand[_RED_TYPES.index(t)]:
        return s + "r" if state.hand.count(t) <= 1 else s + "r/" + s
    return s


def tile_str(idx):
    if idx < 0:
        return "?"
    if idx < 9:   return f"{idx+1}m"
    if idx < 18:  return f"{idx-9+1}p"
    if idx < 27:  return f"{idx-18+1}s"
    return ["Est", "Sud", "Ouest", "Nord", "Blanc", "Vert", "Rouge"][idx - 27]


def decode_payload(data):
    data = bytearray(data)
    for i in range(len(data)):
        u = (23 ^ len(data)) + 5 * i + KEYS[i % len(KEYS)] & 255
        data[i] ^= u
    return bytes(data)


def msg_to_dict(obj):
    for kwargs in (
        {"always_print_fields_with_no_presence": True},
        {"including_default_value_fields": True},
        {},
    ):
        try:
            return MessageToDict(obj, preserving_proto_field_name=True, **kwargs)
        except TypeError:
            continue
    return {}


_req_tracker = {}


def parse_message(content):
    if len(content) < 2:
        return None, None, None
    msg_type = content[0]

    if msg_type == 1:
        wrapper = pb.Wrapper()
        try:
            wrapper.ParseFromString(content[1:])
        except Exception:
            return None, None, None
        method = wrapper.name
        if not method:
            return None, None, None
        name = method.split(".")[-1]
        cls = getattr(pb, name, None)
        if cls is None:
            return method, None, None
        obj = cls()
        try:
            obj.ParseFromString(wrapper.data)
        except Exception:
            return method, None, None

        if name == "ActionPrototype":
            action_name = getattr(obj, "name", None)
            action_data = getattr(obj, "data", None)
            if not action_name or not action_data:
                return method, None, action_name
            action_cls = getattr(pb, action_name, None)
            if action_cls is None:
                return method, None, action_name
            try:
                act = action_cls()
                act.ParseFromString(decode_payload(action_data))
                return method, msg_to_dict(act), action_name
            except Exception:
                return method, None, action_name
        else:
            try:
                return method, msg_to_dict(obj), None
            except Exception:
                return method, None, None

    elif msg_type == 2:
        if len(content) < 3:
            return None, None, None
        req_id = struct.unpack("<H", content[1:3])[0]
        wrapper = pb.Wrapper()
        try:
            wrapper.ParseFromString(content[3:])
        except Exception:
            return None, None, None
        method = wrapper.name
        if method:
            _req_tracker[req_id] = method
        return method, None, None

    elif msg_type == 3:
        if len(content) < 3:
            return None, None, None
        req_id = struct.unpack("<H", content[1:3])[0]
        return _req_tracker.pop(req_id, None), None, None

    return None, None, None


# ═══════════════════════════════════════════════════════════════════════
# FURITEN : helpers
# ═══════════════════════════════════════════════════════════════════════

_SHANTEN = None


def _is_agari(counts):
    """True si `counts` (14 tuiles equivalentes) est une main complete :
    standard + chiitoitsu + KOKUSHI (la lib `mahjong` les gere par defaut).
    [FIX FURITEN] update_furiten utilisait shanten_of() qui a
    use_kokushi=False : un kokushi n'etait jamais detecte comme attente."""
    global _SHANTEN
    try:
        if _SHANTEN is None:
            from mahjong.shanten import Shanten
            _SHANTEN = Shanten()
        return _SHANTEN.calculate_shanten([int(x) for x in counts]) == -1
    except Exception:
        from mahjong_ai.envs.tiles import shanten_of
        return shanten_of([int(x) for x in counts]) == -1


# ═══════════════════════════════════════════════════════════════════════
# ETAT DE JEU
# ═══════════════════════════════════════════════════════════════════════

class GameState:
    def __init__(self):
        self._pmcpa_pending_adapt = False
        self.reset()

    def reset(self):
        self.my_seat = None
        self.hand = []
        self.hand_counts = np.zeros(N_TILES, dtype=np.int32)
        self.melds = []
        self.discards = [[], [], [], []]
        self.opp_melds = [[], [], []]
        self.opp_hand_sizes = [13, 13, 13]
        self.own_riichi = False
        self.opp_riichi = [False, False, False]
        self.dora_indicators = []
        self.wall_remaining = INITIAL_WALL_SIZE
        self.pending_action = None
        self.last_claimed_tile = None
        self.last_chi_tiles = None
        self.last_pending_tile = None
        self.last_ops = None
        # [FIX FURITEN] trois sources, `furiten` = leur OU :
        #   furiten_perm   : une attente est dans MA riviere (sutehai),
        #                    tuiles prises par un adversaire comprises
        #   furiten_temp   : ron manque -> jusqu'a MA prochaine defausse
        #   furiten_riichi : ron manque en riichi -> jusqu'a fin de manche
        self.furiten = False
        self.furiten_perm = False
        self.furiten_temp = False
        self.furiten_riichi = False
        self._river_all = [[] for _ in range(4)]   # defausses COMPLETES
        self._ron_watch = None                     # tuile gagnante a resoudre
        self._clear_reds()
        self.last_discarder_seat = None
        self.last_discarded_tile = None
        self.round_wind_idx = 0
        self.dealer_seat = 0
        self.honba = 0
        self.kyoutaku = 0

        self.round_idx = 0
        self.game_format = os.environ.get("SUPHX_GAME_FORMAT", "tonpuu")
        if self.game_format not in ("hanchan", "tonpuu"):
            raise ValueError(
                "SUPHX_GAME_FORMAT doit valoir 'hanchan' ou 'tonpuu'"
            )
        self.scores = [0.0, 0.0, 0.0, 0.0]
        self._score_baseline = None
        self._initial_wall_size = INITIAL_WALL_SIZE
        self._safe_after_riichi = [np.zeros(N_TILES, dtype=np.float32) for _ in range(4)]
        self._discard_hist = [[] for _ in range(4)]
        self._last_draw_per_seat = [None, None, None, None]
        self._any_call = False
        self._last_discarder = None
        self._has_drawn_since_call = True  # True au debut de manche


    # ------------------------------------------------------------------ #
    # [ROUGES] suivi des 3 rouges (0m, 0p, 0s)
    # ------------------------------------------------------------------ #
    def _clear_reds(self):
        self.red_hand = np.zeros(3, dtype=bool)                 # MA main
        self.red_river = [np.zeros(3, dtype=bool) for _ in range(4)]
        self.red_meld = [np.zeros(3, dtype=bool) for _ in range(4)]
        self.red_indicator = np.zeros(3, dtype=bool)
        self.last_discard_red = -1

    def _set_indicators(self, raw_list):
        """Met a jour les indicateurs de dora (et leur eventuel rouge)."""
        inds = []
        for x in raw_list:
            t = tile_to_34(x)
            if t >= 0:
                inds.append(t)
            k = tile_red_index(x)
            if k >= 0:
                self.red_indicator[k] = True
        if inds:
            self.dora_indicators = inds

    def _red_into_meld(self, seat, k):
        for s in range(4):
            self.red_river[s][k] = False
        self.red_meld[seat][k] = True
        if seat == self.my_seat:
            self.red_hand[k] = False

    def red_state(self):
        hand = [np.zeros(3, dtype=bool) for _ in range(4)]
        if self.my_seat is not None:
            hand[self.my_seat] = self.red_hand.copy()
        pending = -1
        t = self.last_pending_tile
        k = self.last_discard_red
        if (t is not None and 0 <= k < 3 and _RED_TYPES[k] == t):
            pending = k
        return {'hand': hand, 'meld': [m.copy() for m in self.red_meld],
                'river': [r.copy() for r in self.red_river],
                'indicator': self.red_indicator.copy(),
                'pending': pending}

    def update_counts(self):
        self.hand_counts = np.zeros(N_TILES, dtype=np.int32)
        for t in self.hand:
            if 0 <= t < N_TILES:
                self.hand_counts[t] += 1

    def opp_index(self, seat):
        if self.my_seat is None:
            return None
        idx = (seat - self.my_seat - 1) % 4
        return idx if 0 <= idx < 3 else None

    def riichi_of(self, real_seat):
        if real_seat == self.my_seat:
            return self.own_riichi
        idx = self.opp_index(real_seat)
        if idx is None:
            return False
        return self.opp_riichi[idx]

    def _update_opp_hand_sizes(self):
        for i in range(3):
            removed = 0
            for m in self.opp_melds[i]:
                t = m["type"] if isinstance(m, dict) else None
                if t in ("chi", "pon", "kan_open"):
                    removed += 3
                elif t in ("kan_closed", "shouminkan"):
                    removed += 4
                else:
                    removed += 3
            self.opp_hand_sizes[i] = max(0, HAND_SIZE - removed)

    def _tenpai_counts(self):
        """Compte 34 de MA main (13 tuiles equivalentes hors pioche) :
        main + melds, un kan compte pour 3."""
        counts = np.zeros(34, dtype=np.int32)
        for t in self.hand:
            if 0 <= t < 34:
                counts[t] += 1
        for m in self.melds:
            for t in m["tiles"]:
                counts[t] += 1
        for m in self.melds:
            if m["type"] in ("kan_closed", "shouminkan", "kan_open"):
                counts[m["tiles"][0]] -= 1
        return counts

    def my_waits(self):
        """Tuiles qui completent ma main de 13 (standard, chiitoi, kokushi)."""
        counts = self._tenpai_counts()
        if int(counts.sum()) >= 14:
            return []
        waits = []
        for t in range(34):
            if counts[t] >= 4:
                continue
            trial = counts.copy()
            trial[t] += 1
            if _is_agari(trial):
                waits.append(t)
        return waits

    def update_furiten(self):
        """Furiten = permanent (sutehai) OU temporaire OU riichi.

        Permanent : calcule sur 13 tuiles ; a 14 tuiles (apres pioche) il
        vaut False, comme dans l'env d'entrainement (_is_furiten). Les
        furiten temporaire / riichi, eux, persistent a 14 tuiles.
        [FIX FURITEN] La riviere utilisee est `_river_all` : une tuile
        defaussee puis prise par un adversaire (chi/pon/kan) compte encore."""
        if self.my_seat is None:
            self.furiten = self.furiten_perm = False
            return
        perm = False
        try:
            counts = self._tenpai_counts()
            if int(counts.sum()) < 14:
                for t in set(self._river_all[self.my_seat]):
                    if 0 <= t < 34:
                        trial = counts.copy()
                        trial[t] += 1
                        if _is_agari(trial):
                            perm = True
                            break
        except Exception:
            perm = False
        self.furiten_perm = perm
        self.furiten = bool(perm or self.furiten_temp or self.furiten_riichi)

    def _arm_ron_watch(self, tile):
        """Un adversaire vient de defausser (ou d'ajouter en kakan) `tile`.
        Si j'aurais pu faire ron (attente, pas deja furiten, yaku), je
        note la tuile : le message SUIVANT dira si j'ai pris le ron
        (ActionHule) ou passe (la partie continue)."""
        self._ron_watch = None
        if self.my_seat is None or self.furiten:
            return
        try:
            if tile in self.my_waits() and (
                    self.own_riichi or has_yaku(self, tile)):
                self._ron_watch = tile
        except Exception as e:
            _say(f"[WARN] arm_ron_watch: {e}")

    def _resolve_ron_watch(self, action_name):
        if self._ron_watch is None:
            return
        if action_name in ("ActionDealTile", "ActionChiPengGang",
                           "ActionAnGangAddGang"):
            tile = self._ron_watch
            self._ron_watch = None
            self.furiten_temp = True
            if self.own_riichi:
                self.furiten_riichi = True
            _say(f"[FURITEN] ron manque sur {tile_str(tile)} -> "
                 f"{'riichi (jusqu a la fin de la manche)' if self.own_riichi else 'temporaire (jusqu a ta defausse)'}")
        else:
            # ActionHule / ActionNoTile / ActionLiuJu / ActionNewRound :
            # la manche s'arrete, rien a marquer.
            self._ron_watch = None

    def _force_kong_coherence(self):
        for t in range(N_TILES):
            if self.hand_counts[t] >= 4:
                for m in self.melds:
                    if m["type"] == "kan_closed" and m["tiles"][0] == t:
                        removed = 0
                        while t in self.hand and removed < 4:
                            self.hand.remove(t)
                            removed += 1
                        _say(f"[FIX] Corrige doublon kan {tile_str(t)} "
                             f"({removed} tuiles retirees)")
                        self.update_counts()
                        break

    def handle(self, method, data, action_name):
        if action_name is None or not isinstance(data, dict):
            return
        seat = data.get("seat")

        # [FIX FURITEN] resout le ron potentiel de l'action precedente
        self._resolve_ron_watch(action_name)

        if TRACE_ALL_ACTIONS and action_name and action_name.startswith("Action"):
            keys = list(data.keys())[:12]
            _say(f"[TRACE] {action_name} keys={keys}")

        if (self.my_seat is None
                and action_name == "ActionDealTile"
                and "operation" in data
                and seat is not None):
            self.my_seat = seat
            _say(f"Vous etes le joueur {seat + 1}")

        # [pMCPA] Reset sur poids offline AVANT la nouvelle manche
        if action_name == "ActionNewRound":
            if _pmcpa is not None:
                _pmcpa.reset()
            self._new_round(data)
        elif action_name == "ActionDealTile":
            self._deal(data)
            if seat == self.my_seat:
                _say(f"[DBG-DEAL] hand={sorted(self.hand)} "
                     f"melds={[m['type'] for m in self.melds]}")

            # [pMCPA] Lance l'adaptation des qu'on connait my_seat
            # (au premier DealTile avec operation).
            if (getattr(self, "_pmcpa_pending_adapt", False)
                    and self.my_seat is not None
                    and _pmcpa is not None):
                self._pmcpa_pending_adapt = False
                try:
                    _say(f"[pMCPA] Lancement adaptation "
                         f"(main: {len(self.hand)} tuiles, "
                         f"seat={self.my_seat})")
                    _pmcpa.adapt_from_sniffer_state(self)
                except Exception as e:
                    _say(f"[pMCPA] Erreur lancement : {e}")
                    import traceback
                    traceback.print_exc()

            
            if seat == self.my_seat:
                _say(f"[DBG-DEAL] hand={sorted(self.hand)} "
                     f"melds={[m['type'] for m in self.melds]}")
        elif action_name == "ActionDiscardTile":
            self._discard(data)
        elif action_name == "ActionChiPengGang":
            self._call(data)
        elif action_name == "ActionAnGangAddGang":
            self._an_gang_add_gang(data)
        elif action_name == "ActionLiqi":
            self._riichi(data)

        # [ROUGES/DORA] un kan revele un nouvel indicateur : Mahjong Soul
        # renvoie la liste complete dans `doras` (jamais relue avant).
        if action_name != "ActionNewRound" and data.get("doras"):
            self._set_indicators(data["doras"])

        self._update_opp_hand_sizes()
        self._force_kong_coherence()
        self.update_furiten()

        # [pMCPA] Lance l'adaptation apres avoir recu la main initiale
        # (13 tuiles pour les non-dealers, 14 pour le dealer apres pioche)
        # [pMCPA] Marque qu'on veut lancer l'adaptation des que my_seat
        # sera connu (au premier DealTile avec operation).
        if (action_name == "ActionNewRound" and len(self.hand) >= 13
                and _pmcpa is not None):
            self._pmcpa_pending_adapt = True

        if action_name == "ActionNewRound" and len(self.hand) >= 14:
            if self.my_seat is None:
                self.my_seat = 0
                _say(f"Vous etes le joueur 1 (dealer, {len(self.hand)} tuiles)")
            self.last_pending_tile = None
            self.last_ops = None
            self.pending_action = {"name": "DISCARD"}
            return

        if (action_name == "ActionDealTile"
                and "operation" in data
                and seat == self.my_seat):
            self.last_pending_tile = None
            self.last_ops = None
            self.pending_action = {"name": "DISCARD"}
            return

        if (action_name == "ActionChiPengGang"
                and seat == self.my_seat
                and "operation" in data):
            self.last_pending_tile = None
            self.last_ops = None
            self.pending_action = {"name": "DISCARD"}
            return

        if (action_name == "ActionDiscardTile" and seat != self.my_seat):
            self.last_pending_tile = tile_to_34(data.get("tile"))
            self.last_ops = data.get("operation", {})
            self.pending_action = {"name": "CALL", "discard_seat": seat}
            return

    def _new_round(self, data):
        self.melds = []
        self.discards = [[], [], [], []]
        self.opp_melds = [[], [], []]
        self.opp_hand_sizes = [13, 13, 13]
        self.own_riichi = False
        self.opp_riichi = [False, False, False]
        self.hand = []
        self.pending_action = None
        self.last_claimed_tile = None
        self.last_chi_tiles = None
        self.last_pending_tile = None
        self.last_ops = None
        self.furiten = False
        self.furiten_perm = False
        self.furiten_temp = False
        self.furiten_riichi = False
        self._river_all = [[] for _ in range(4)]
        self._ron_watch = None
        self._clear_reds()
        self.last_discarder_seat = None
        self.last_discarded_tile = None
        self.wall_remaining = INITIAL_WALL_SIZE
        self._initial_wall_size = INITIAL_WALL_SIZE
        self._safe_after_riichi = [np.zeros(N_TILES, dtype=np.float32) for _ in range(4)]
        self._discard_hist = [[] for _ in range(4)]
        self._last_draw_per_seat = [None, None, None, None]
        self._any_call = False
        self._last_discarder = None
        self._pmcpa_pending_adapt = False

        self.round_wind_idx = int(data.get("chang", 0) or 0)
        self.dealer_seat = int(data.get("ju", 0) or 0)
        self.honba = int(data.get("ben", 0) or 0)

        raw_scores = data.get("scores")
        if isinstance(raw_scores, list) and len(raw_scores) == 4:
            if self._score_baseline is None:
                self._score_baseline = [float(s) for s in raw_scores]
            self.scores = [
                (float(raw_scores[p]) - self._score_baseline[p]) / 1000.0
                for p in range(4)
            ]

        if getattr(self, "_round_seen", False):
            self.round_idx += 1
        self._round_seen = True

        if data.get("doras"):
            self._set_indicators(data["doras"])
        if data.get("tiles"):
            raw = data["tiles"]
            self.hand = [t for t in (tile_to_34(x) for x in raw) if t >= 0]
            for x in raw:
                k = tile_red_index(x)
                if k >= 0:
                    self.red_hand[k] = True
            self.update_counts()
            hand_display = disp_hand(self.hand, self.red_hand)
            _say(f"Main initiale ({len(self.hand)} tuiles) : {hand_display}")
        if "left_tile_count" in data:
            self.wall_remaining = data["left_tile_count"]
            self._initial_wall_size = max(self.wall_remaining, 1)

    def _deal(self, data):
        seat = data.get("seat")
        tile = data.get("tile")
        if "left_tile_count" in data:
            self.wall_remaining = data["left_tile_count"]
        if seat is None or tile is None:
            return
        idx = tile_to_34(tile)
        if idx < 0:
            return
        self._last_draw_per_seat[seat] = idx
        if seat == self.my_seat:
            k = tile_red_index(tile)
            if k >= 0:
                self.red_hand[k] = True
            self.hand.append(idx)
            self.update_counts()
            # [FIX KAN-TIMING] On vient de piocher -> kan a nouveau autorise
            self._has_drawn_since_call = True

    def _discard(self, data):
        seat = data.get("seat")
        tile = data.get("tile")
        if seat is None or tile is None:
            return
        idx = tile_to_34(tile)
        if idx < 0:
            return

        tsumogiri = (self._last_draw_per_seat[seat] == idx)
        self._last_draw_per_seat[seat] = None

        k_red = tile_red_index(tile)
        self.last_discard_red = k_red
        if k_red >= 0:
            if seat == self.my_seat:
                self.red_hand[k_red] = False
            if 0 <= seat < 4:
                self.red_river[seat][k_red] = True

        if seat == self.my_seat and idx in self.hand:
            self.hand.remove(idx)
            self.update_counts()
        if 0 <= seat < len(self.discards):
            self.discards[seat].append(idx)

        self._discard_hist[seat].append((idx, bool(tsumogiri)))
        self._river_all[seat].append(idx)
        self._last_discarder = seat
        for other in range(4):
            if other == seat:
                continue
            if self.riichi_of(other):
                self._safe_after_riichi[other][idx] = 1.0

        self.last_discarder_seat = seat
        self.last_discarded_tile = idx
        if data.get("is_liqi"):
            if seat == self.my_seat:
                self.own_riichi = True
            else:
                oi = self.opp_index(seat)
                if oi is not None:
                    self.opp_riichi[oi] = True

        # [FIX FURITEN] ma defausse met fin au furiten temporaire (pas au
        # furiten riichi) ; la defausse d'un adversaire est un ron potentiel.
        if seat == self.my_seat:
            self.furiten_temp = False
            self._ron_watch = None
        else:
            self._arm_ron_watch(idx)

    def _call(self, data):
        seat = data.get("seat")
        raw = data.get("tiles", [])
        if seat is None or not raw:
            return

        meld = [t for t in (tile_to_34(x) for x in raw) if t >= 0]
        if not meld:
            return

        ms = sorted(meld)

        if len(meld) == 3 and ms[0] == ms[1] == ms[2]:
            meld_type = "pon"
        elif len(meld) == 3 and ms[1] == ms[0] + 1 and ms[2] == ms[1] + 1:
            meld_type = "chi"
        elif len(meld) == 4 and ms[0] == ms[1] == ms[2] == ms[3]:
            if seat == self.my_seat and self.hand_counts[ms[0]] >= 4:
                meld_type = "kan_closed"
                _say(f"[INFO] Ankan detecte via ChiPengGang "
                     f"({tile_str(ms[0])})")
            else:
                meld_type = "kan_open"
        else:
            meld_type = "unknown"

        if meld_type in ("pon", "chi", "kan_open"):
            if (self.last_discarder_seat is not None
                    and self.last_discarded_tile is not None):
                d = self.discards[self.last_discarder_seat]
                if self.last_discarded_tile in d:
                    d.remove(self.last_discarded_tile)

        self._any_call = True
        self._last_draw_per_seat[seat] = None

        # [ROUGES] le rouge de la tuile appelee (riviere -> meld) ou venant
        # de la main (consume) entre dans le meld
        if 0 <= seat < 4:
            for x in raw:
                k = tile_red_index(x)
                if k >= 0:
                    self._red_into_meld(seat, k)

        if seat == self.my_seat:
            self.melds.append({"type": meld_type, "tiles": ms})
            self.last_chi_tiles = ms if meld_type == "chi" else None

            if meld_type == "kan_closed":
                n_to_remove = 4
            else:
                n_to_remove = len(meld) - 1

            removed = 0
            for m_tile in list(meld):
                if removed >= n_to_remove:
                    break
                if m_tile in self.hand:
                    self.hand.remove(m_tile)
                    removed += 1

            if meld_type == "kan_closed":
                self.last_claimed_tile = None
            else:
                remaining = list(meld)
                for m_tile in list(remaining):
                    if m_tile in self.hand:
                        remaining.remove(m_tile)
                self.last_claimed_tile = remaining[-1] if remaining else meld[-1]

            self.update_counts()
            # [FIX KAN-TIMING] On vient d'appeler -> kan interdit jusqu'a
            # la prochaine pioche. Un kan (open/closed) compte aussi comme
            # un "appel" : la tuile rinshan qui suit remettra le flag a True.
            self._has_drawn_since_call = False
            _say(f"[INFO] Meld {meld_type} : "
                 f"{[tile_str(t) for t in ms]} ({removed} retires)")
        else:
            oi = self.opp_index(seat)
            if oi is not None:
                self.opp_melds[oi].append({"type": meld_type, "tiles": ms})

    def _an_gang_add_gang(self, data):
        """Ankan et shouminkan robustes."""
        seat = data.get("seat")
        gang_type = data.get("type")

        idx, src = _extract_single_tile(data)
        if seat is None or idx < 0:
            _say(f"[WARN] AnGangAddGang sans seat ou tuile: {data}")
            return

        _say(f"[DBG-KAN-RAW] seat={seat} type={gang_type!r} "
             f"src={src!r} idx={idx}({tile_str(idx)}) "
             f"hand_count={self.hand_counts[idx]}")

        if gang_type is None:
            gt = ""
        else:
            try:
                gt = str(int(gang_type)).lower()
            except (ValueError, TypeError):
                gt = str(gang_type).strip().lower()

        is_ankan = False
        is_shouminkan = False

        if gt in ("kan", "ankan", "closed", "closed_kan"):
            is_ankan = True
        elif gt in ("add_gang", "kakan", "shouminkan",
                    "added", "added_kan"):
            is_shouminkan = True
        elif gt in ("1", "2"):
            has_pon = any(m["type"] == "pon" and m["tiles"][0] == idx
                          for m in self.melds)
            if has_pon and self.hand_counts[idx] >= 1:
                is_shouminkan = True
            elif self.hand_counts[idx] >= 4:
                is_ankan = True
            else:
                is_ankan = True
        else:
            has_pon = any(m["type"] == "pon" and m["tiles"][0] == idx
                          for m in self.melds)
            if has_pon and self.hand_counts[idx] >= 1:
                is_shouminkan = True
                _say(f"[FIX] Type inconnu ({gang_type!r}) + pon existant "
                     f"-> SHouminkan {tile_str(idx)}")
            elif self.hand_counts[idx] >= 4:
                is_ankan = True
                _say(f"[FIX] Type inconnu ({gang_type!r}) + 4 en main "
                     f"-> ankan {tile_str(idx)}")
            else:
                _say(f"[WARN] AnGangAddGang type inconnu ({gang_type!r}) "
                     f"et etat ambigu (hand_count={self.hand_counts[idx]})")
                return

        _say(f"[DBG-KAN] seat={seat} type={gang_type!r} tile={tile_str(idx)} "
             f"-> ankan={is_ankan} shouminkan={is_shouminkan}")

        self._any_call = True

        # [ROUGES] ankan : 4 copies d'un 5 => le rouge y est ; kakan : la
        # tuile ajoutee est rouge si le message le dit (sinon, pour moi, si
        # ma derniere copie en main etait le rouge).
        k_five = {t: i for i, t in enumerate(_RED_TYPES)}.get(idx, -1)
        if 0 <= seat < 4 and k_five >= 0:
            k_raw = tile_red_index(_raw_tile_of(data))
            if is_ankan:
                self._red_into_meld(seat, k_five)
            elif is_shouminkan:
                if k_raw == k_five:
                    self._red_into_meld(seat, k_five)
                elif (seat == self.my_seat and self.red_hand[k_five]
                        and self.hand.count(idx) <= 1):
                    self._red_into_meld(seat, k_five)
                if k_raw == k_five or self.red_meld[seat][k_five]:
                    self.last_discard_red = k_five

        if seat == self.my_seat:
            if is_ankan:
                removed = 0
                for _ in range(4):
                    if idx in self.hand:
                        self.hand.remove(idx)
                        removed += 1
                if removed < 4:
                    _say(f"[WARN] Ankan: seulement {removed}/4 tuiles "
                         f"trouvees en main, etat desynchronise ?")
                existing_kan = any(
                    m["type"] == "kan_closed" and m["tiles"][0] == idx
                    for m in self.melds)
                if not existing_kan:
                    self.melds.append({"type": "kan_closed",
                                        "tiles": [idx] * 4})
                    _say(f"[INFO] Ankan : {tile_str(idx)}")
                else:
                    _say(f"[INFO] Ankan deja existant pour {tile_str(idx)}, "
                         f"skip meld add")

            elif is_shouminkan:
                if idx in self.hand:
                    self.hand.remove(idx)
                    _say(f"[DBG] retrait {tile_str(idx)} de la main "
                         f"(reste {self.hand.count(idx)} copies)")
                else:
                    _say(f"[WARN] Shouminkan: {tile_str(idx)} pas en main "
                         f"(hand_count={self.hand_counts[idx]})")
                found = False
                for m in self.melds:
                    if m["type"] == "pon" and m["tiles"][0] == idx:
                        m["type"] = "shouminkan"
                        m["tiles"].append(idx)
                        found = True
                        break
                if found:
                    _say(f"[INFO] Shouminkan : {tile_str(idx)}")
                else:
                    _say(f"[WARN] Shouminkan sans pon existant pour "
                          f"{tile_str(idx)}")

            self._last_draw_per_seat[seat] = None
            self.update_counts()
            # [FIX KAN-TIMING] Un kan consomme la pioche courante (on va
            # piocher en rinshan). Jusqu'a cette rinshan -> pas d'autre kan.
            self._has_drawn_since_call = False
            _say(f"[DBG-KAN-AFTER] hand_size={len(self.hand)} "
                 f"melds={[m['type'] for m in self.melds]}")

        else:
            oi = self.opp_index(seat)
            if oi is None:
                return
            if is_ankan:
                self.opp_melds[oi].append(
                    {"type": "kan_closed", "tiles": [idx] * 4})
            elif is_shouminkan:
                for m in self.opp_melds[oi]:
                    if m["type"] == "pon" and m["tiles"][0] == idx:
                        m["type"] = "shouminkan"
                        m["tiles"].append(idx)
                        break
                # [FIX FURITEN] chankan possible : passer = ron manque
                self._arm_ron_watch(idx)

    def _riichi(self, data):
        seat = data.get("seat")
        if seat is None:
            return
        if seat == self.my_seat:
            self.own_riichi = True
        else:
            oi = self.opp_index(seat)
            if oi is not None:
                self.opp_riichi[oi] = True


# ═══════════════════════════════════════════════════════════════════════
# OBSERVATION
# ═══════════════════════════════════════════════════════════════════════

def _build_hands_list(state):
    hands = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
    if state.my_seat is not None:
        hands[state.my_seat] = state.hand_counts.copy()
    return hands


def _build_melds_list(state):
    melds = [[] for _ in range(4)]
    if state.my_seat is not None:
        melds[state.my_seat] = state.melds
    for i in range(3):
        real_seat = ((state.my_seat or 0) + 1 + i) % 4
        melds[real_seat] = state.opp_melds[i]
    return melds


def _build_discards_list(state):
    discards = [np.zeros(N_TILES, dtype=np.int32) for _ in range(4)]
    for p in range(4):
        for t in state.discards[p]:
            if 0 <= t < N_TILES:
                discards[p][t] += 1
    return discards


def _build_discard_hist(state):
    return state._discard_hist


def _build_riichi_flags(state):
    flags = [False] * 4
    if state.my_seat is not None:
        flags[state.my_seat] = state.own_riichi
    for i in range(3):
        real_seat = ((state.my_seat or 0) + 1 + i) % 4
        flags[real_seat] = state.opp_riichi[i]
    return flags


def build_observation(state):
    if state.my_seat is None:
        return np.zeros((N_CHANNELS, N_TILES), dtype=np.float32)

    round_id = (state.round_idx % 4) + 1
    normal_round_count = 4 if state.game_format == "tonpuu" else 8
    rounds_remaining = max(
        normal_round_count - (state.round_idx + 1), 0
    )
    is_last_normal_round = (
        state.round_idx == normal_round_count - 1
    )
    is_extension = state.round_idx >= normal_round_count
    extension_index = max(
        0, state.round_idx - normal_round_count + 1
    )

    return build_suphx_obs(
        hands=_build_hands_list(state),
        melds=_build_melds_list(state),
        discards=_build_discards_list(state),
        discard_hist=_build_discard_hist(state),
        riichi_flags=_build_riichi_flags(state),
        furiten_flag=state.furiten,
        dora_indicators=state.dora_indicators,
        wall_remaining=state.wall_remaining,
        scores=list(state.scores),
        dealer_seat=state.dealer_seat,
        round_wind=state.round_wind_idx,
        round_id=round_id,
        honba=state.honba,
        kyoutaku=state.kyoutaku,
        seat=state.my_seat,
        pending_claim_tile=state.last_pending_tile,
        compute_lookahead_flag=USE_LOOKAHEAD,
        # [SUPPRIME] game_format=state.game_format,
        is_last_normal_round=is_last_normal_round,
        is_extension=is_extension,
        extension_index=extension_index,
        normal_round_count=normal_round_count,
        # [SUPPRIME] round_progress=state.round_idx + 1,
        rounds_remaining=rounds_remaining,
        red_state=state.red_state(),
    )

# ═══════════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODELES
# ═══════════════════════════════════════════════════════════════════════

_sl = None
_pmcpa = None   # [pMCPA] instance partagee

HEAD_NAMES = ['discard', 'riichi', 'chow', 'pong', 'kong']

ACTION_TYPE_TO_HEAD = {
    DISCARD: 'discard',
    RIICHI:  'riichi',
    CHOW:    'chow',
    PONG:    'pong',
    KONG:    'kong',
}


def load_sl_models():
    global _sl, _pmcpa
    if _sl is not None:
        return _sl
    if not USE_SL:
        return None

    try:
        from mahjong_ai.agents.suphx_cnn import (
            build_model, IN_CHANNELS_ORACLE, adapt_in_channels,
        )

        models = {}
        for head in HEAD_NAMES:
            path = os.path.join(CHECKPOINT_DIR, f'{head}.pt')
            if not os.path.isfile(path):
                print(f"[sniffer] Fichier manquant : {path}", flush=True)
                return None

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Chargement {head} : {path}...", flush=True)
            t0 = time.time()
            ck = torch.load(path, map_location='cpu', weights_only=False)

            if 'policy' in ck:
                print(f"[sniffer] {head} : format RL detecte", flush=True)
                model = build_model(head, use_oracle=True)
                my_state = model.state_dict()
                for k, v in ck['policy'].items():
                    if k not in my_state:
                        continue
                    if my_state[k].shape == v.shape:
                        my_state[k].copy_(v)
                    elif k == 'trunk.input_conv.weight':
                        # [ROUGES] ancien ckpt 272 canaux -> 283 : poids nuls
                        # pour les 11 canaux rouges, oracle remis a sa place
                        my_state[k].copy_(
                            adapt_in_channels(v, my_state[k].shape[1]))
                model.load_state_dict(my_state)
                val_acc = ck.get('val_acc', None)
                val_str = f"{val_acc:.4f}" if val_acc is not None else "RL"
            elif 'model_state_dict' in ck:
                cfg = ck.get('config', {})
                model = build_model(
                    head,
                    channels=cfg.get('channels', 256),
                    n_blocks=cfg.get('n_blocks', 16),
                    binary_hidden=cfg.get('binary_hidden', 256),
                    # [ROUGES] toujours N_CHANNELS : un ancien checkpoint a
                    # 268 canaux est adapte au chargement (poids rouges = 0)
                    in_channels=N_CHANNELS,
                )
                model.load_state_dict(ck['model_state_dict'])
                val_acc = ck.get('val_acc', None)
                val_str = f"{val_acc:.4f}" if val_acc is not None else "?"
            else:
                print(f"[sniffer] {head} : format inconnu "
                      f"({list(ck.keys())})", flush=True)
                return None

            model.eval()
            models[head] = model
            dt = time.time() - t0
            print(f"[sniffer] {head} charge en {dt:.2f}s "
                  f"(val_acc={val_str})", flush=True)

        _sl = models

        # [pMCPA] Instancie l'adaptateur si active
        if PMCPA_ENABLE:
            try:
                from mahjong_ai.agents.pmcpa import PMCPA

                head_sizes = {
                    DISCARD: 34, PONG: 2, CHOW: 4,
                    KONG: 2, RIICHI: 2,
                }
                _pmcpa = PMCPA(
                    models=models,
                    head_sizes=head_sizes,
                    device='cpu',
                    k=PMCPA_K,
                    lr=PMCPA_LR,
                    grad_steps=PMCPA_GRAD_STEPS,
                    timeout_s=PMCPA_TIMEOUT,
                    verbose=True,
                )
                print(f"[sniffer] pMCPA active (K={PMCPA_K}, "
                      f"lr={PMCPA_LR}, steps={PMCPA_GRAD_STEPS}, "
                      f"timeout={PMCPA_TIMEOUT}s)", flush=True)
            except Exception as e:
                print(f"[sniffer] pMCPA non disponible : {e}", flush=True)
                import traceback
                traceback.print_exc()
                _pmcpa = None

        return models
    except Exception as e:
        print(f"[ERREUR] Chargement SL/RL : {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def run_all_heads(models, obs):
    """Forward 5 tetes.

    [FIX] Detecte la taille d'input attendue par chaque modele :
        - SL standard   : in_channels = N_CHANNELS (pas d'oracle)
        - RL avec oracle: in_channels = N_CHANNELS + ORACLE_DIM
    Ajoute les canaux oracle de zeros UNIQUEMENT si le modele les attend.
    """
    results = {}
    with torch.no_grad():
        for head in HEAD_NAMES:
            model = models[head]

            # Detecte in_channels attendu depuis le premier conv
            expected_c = None
            for name, p in model.named_parameters():
                if name.endswith('input_conv.weight'):
                    expected_c = p.shape[1]
                    break
            if expected_c is None:
                raise RuntimeError(f"input_conv introuvable dans {head}")

            actual_c = obs.shape[0]
            if expected_c == actual_c:
                obs_input = obs
            elif expected_c == actual_c + 4:
                # Modele avec oracle -> ajoute 4 canaux de zeros
                oracle_zeros = np.zeros((4, obs.shape[1]), dtype=obs.dtype)
                obs_input = np.concatenate([obs, oracle_zeros], axis=0)
            else:
                raise RuntimeError(
                    f"Channel mismatch {head}: modele attend {expected_c}, "
                    f"obs a {actual_c}. Verifie N_CHANNELS et IN_CHANNELS_ORACLE."
                )

            obs_t = torch.as_tensor(obs_input, dtype=torch.float32).unsqueeze(0)
            logits = model(obs_t)[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            results[head] = probs
    return results

# ═══════════════════════════════════════════════════════════════════════
# MASQUES
# ═══════════════════════════════════════════════════════════════════════

def get_discard_mask(state):
    mask = (state.hand_counts > 0).astype(np.float32)
    if state.last_claimed_tile is not None:
        mask[state.last_claimed_tile] = 0.0
    if state.last_chi_tiles is not None:
        a, b, c = sorted(state.last_chi_tiles)
        forbidden = {a, b, c}
        if a < 27:
            suit_start = (a // 9) * 9
            suit_end = suit_start + 8
            if a - 1 >= suit_start:
                forbidden.add(a - 1)
            if c + 1 <= suit_end:
                forbidden.add(c + 1)
        for t in forbidden:
            if 0 <= t < N_TILES:
                mask[t] = 0.0
    if not mask.any():
        mask = (state.hand_counts > 0).astype(np.float32)
    return mask


def get_pong_mask(state, phase, pending_tile, ops):
    if phase != "CALL" or state.own_riichi:
        return np.array([1.0, 0.0], dtype=np.float32)
    if pending_tile is None or pending_tile < 0 or pending_tile >= N_TILES:
        return np.array([1.0, 0.0], dtype=np.float32)
    if state.hand_counts[pending_tile] >= 2:
        return np.array([1.0, 1.0], dtype=np.float32)
    return np.array([1.0, 0.0], dtype=np.float32)


def get_chow_mask(state, phase, pending_tile, ops):
    mask = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if phase != "CALL" or state.own_riichi:
        return mask
    if pending_tile is None or pending_tile < 0 or pending_tile >= 27:
        return mask
    if state.my_seat is None or state.last_discarder_seat is None:
        return mask
    if state.last_discarder_seat != (state.my_seat - 1) % 4:
        return mask
    col = pending_tile % 9
    h = state.hand_counts
    if col >= 2 and h[pending_tile - 2] >= 1 and h[pending_tile - 1] >= 1:
        mask[CHOW_LOWER] = 1.0
    if 1 <= col <= 7 and h[pending_tile - 1] >= 1 and h[pending_tile + 1] >= 1:
        mask[CHOW_MID] = 1.0
    if col <= 6 and h[pending_tile + 1] >= 1 and h[pending_tile + 2] >= 1:
        mask[CHOW_UPPER] = 1.0
    return mask


def get_kong_mask(state, phase, pending_tile, ops):
    if state.own_riichi:
        return np.array([1.0, 0.0], dtype=np.float32)

    # [FIX KAN-TIMING] Pas de kan tant qu'on n'a pas pioche depuis notre
    # dernier appel (chii/pon/kan). Regle Majsoul/Tenhou : un kan exige
    # une pioche depuis le dernier meld. La tuile rinshan apres un kan
    # remet le drapeau a True -> double kan reste possible.
    if not getattr(state, "_has_drawn_since_call", True):
        return np.array([1.0, 0.0], dtype=np.float32)

    if phase == "DISCARD":
        for t in range(N_TILES):
            if state.hand_counts[t] >= 4:
                return np.array([1.0, 1.0], dtype=np.float32)
        for m in state.melds:
            if isinstance(m, dict) and m.get("type") == "pon":
                t = m["tiles"][0]
                if 0 <= t < N_TILES and state.hand_counts[t] >= 1:
                    return np.array([1.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)
    if phase == "CALL":
        if pending_tile is not None and 0 <= pending_tile < N_TILES:
            if state.hand_counts[pending_tile] >= 3:
                return np.array([1.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([1.0, 0.0], dtype=np.float32)

def get_riichi_mask(state, phase):
    if phase != "DISCARD" or state.own_riichi:
        return np.array([1.0, 0.0], dtype=np.float32)
    for m in state.melds:
        if m["type"] != "kan_closed":
            return np.array([1.0, 0.0], dtype=np.float32)
    if state.wall_remaining < 4:
        return np.array([1.0, 0.0], dtype=np.float32)
    if len(state.hand) < 14:
        return np.array([1.0, 0.0], dtype=np.float32)
    try:
        from mahjong.shanten import Shanten
        _shanten = Shanten()
        counts = np.zeros(34, dtype=np.int32)
        for t in state.hand:
            counts[t] += 1
        for m in state.melds:
            for t in m["tiles"]:
                counts[t] += 1
        for m in state.melds:
            if m["type"] in ("kan_closed", "shouminkan", "kan_open"):
                counts[m["tiles"][0]] -= 1
        for t in range(34):
            if counts[t] > 0:
                trial = counts.copy()
                trial[t] -= 1
                if _shanten.calculate_shanten(trial.tolist()) == 0:
                    return np.array([1.0, 1.0], dtype=np.float32)
    except Exception:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([1.0, 0.0], dtype=np.float32)


def has_yaku(state, win_tile):
    try:
        from mahjong.hand_calculating.hand import HandCalculator
        from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
        from mahjong.meld import Meld

        used = {}

        def mk(t):
            c = used.get(t, 0)
            used[t] = c + 1
            return t * 4 + c

        tiles136 = []
        melds = []
        for m in state.melds:
            ts = [mk(t) for t in m["tiles"]]
            tiles136 += ts
            mt = m["type"]
            if mt == "chi":
                melds.append(Meld(meld_type=Meld.CHI, tiles=ts, opened=True))
            elif mt == "pon":
                melds.append(Meld(meld_type=Meld.PON, tiles=ts, opened=True))
            elif mt in ("kan_open", "shouminkan"):
                melds.append(Meld(meld_type=Meld.KAN, tiles=ts, opened=True))
            elif mt == "kan_closed":
                melds.append(Meld(meld_type=Meld.KAN, tiles=ts, opened=False))
            else:
                return False
        tiles136 += [mk(t) for t in state.hand]
        win136 = mk(win_tile)
        tiles136.append(win136)

        winds = [27, 28, 29, 30]
        seat_wind = winds[(state.my_seat - state.dealer_seat) % 4]
        round_wind = winds[state.round_wind_idx % 4]

        config = HandConfig(
            is_tsumo=False,
            is_riichi=bool(state.own_riichi),
            is_houtei=(state.wall_remaining == 0),
            player_wind=seat_wind,
            round_wind=round_wind,
            options=OptionalRules(has_open_tanyao=True,
                                    has_double_yakuman=True),
        )
        res = HandCalculator().estimate_hand_value(
            tiles136, win136, melds=melds, config=config)
        if res.error is not None:
            return False
        return bool(res.han and res.han > 0) or bool(res.yaku)
    except Exception as e:
        print(f"[WARN] has_yaku: {e}", flush=True)
        return False


def get_ron_mask(state, phase, pending_tile, ops):
    if phase != "CALL" or state.my_seat is None:
        return np.array([1.0, 0.0], dtype=np.float32)
    if pending_tile is None or pending_tile < 0 or pending_tile >= N_TILES:
        return np.array([1.0, 0.0], dtype=np.float32)
    if state.furiten:
        return np.array([1.0, 0.0], dtype=np.float32)
    try:
        from mahjong.shanten import Shanten
        _shanten = Shanten()
        counts = np.zeros(34, dtype=np.int32)
        for t in state.hand:
            counts[t] += 1
        for m in state.melds:
            for t in m["tiles"]:
                counts[t] += 1
        for m in state.melds:
            if m["type"] in ("kan_closed", "shouminkan", "kan_open"):
                counts[m["tiles"][0]] -= 1
        if int(counts.sum()) >= 14:
            return np.array([1.0, 0.0], dtype=np.float32)
        river = getattr(state, "_river_all", None)
        river = river[state.my_seat] if river else state.discards[state.my_seat]
        for t in set(river):
            if 0 <= t < N_TILES:
                trial = counts.copy()
                trial[t] += 1
                if _shanten.calculate_shanten(trial.tolist()) == -1:
                    return np.array([1.0, 0.0], dtype=np.float32)
        trial = counts.copy()
        trial[pending_tile] += 1
        if _shanten.calculate_shanten(trial.tolist()) == -1:
            if has_yaku(state, pending_tile):
                return np.array([1.0, 1.0], dtype=np.float32)
            return np.array([1.0, 0.0], dtype=np.float32)
    except Exception as e:
        print(f"[WARN] get_ron_mask: {e}", flush=True)
    return np.array([1.0, 0.0], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
# RECOMMANDATION
# ═══════════════════════════════════════════════════════════════════════

def _normalized(probs, mask):
    if mask.sum() <= 0:
        return None
    m = probs * mask
    if m.sum() <= 0:
        return None
    return m / m.sum()


def _best_riichi_discard(state, head_probs):
    try:
        from mahjong.shanten import Shanten
        _shanten = Shanten()
        counts = np.zeros(34, dtype=np.int32)
        for t in state.hand:
            counts[t] += 1
        for m in state.melds:
            for t in m["tiles"]:
                counts[t] += 1
        for m in state.melds:
            if m["type"] in ("kan_closed", "shouminkan", "kan_open"):
                counts[m["tiles"][0]] -= 1

        valid_options = []
        for t in range(34):
            if counts[t] > 0:
                trial = counts.copy()
                trial[t] -= 1
                if _shanten.calculate_shanten(trial.tolist()) == 0:
                    valid_options.append(t)

        if not valid_options:
            return ""

        riichi_discard_mask = np.zeros(N_TILES, dtype=np.float32)
        for t in valid_options:
            riichi_discard_mask[t] = 1.0

        probs = head_probs["discard"].copy()
        masked = probs * riichi_discard_mask
        if masked.sum() <= 0:
            return tile_str(valid_options[0])
        best = int(np.argmax(masked))
        return tile_str(best)
    except Exception:
        return ""


def compute_final_recommendation(state, head_probs, phase):
    pending = state.last_pending_tile
    ops = state.last_ops

    if phase == "CALL":
        ron_mask = get_ron_mask(state, phase, pending, ops)
        if ron_mask[1] > 0.5:
            return ("RON", 1.0, "")

        kan_mask = get_kong_mask(state, phase, pending, ops)
        m = _normalized(head_probs["kong"], kan_mask)
        if m is not None and m[1] > 0.5:
            detail = (f"daiminkan({tile_str(pending)})"
                      if pending is not None else "")
            return ("KAN", float(m[1]), detail)

        pon_mask = get_pong_mask(state, phase, pending, ops)
        m = _normalized(head_probs["pong"], pon_mask)
        if m is not None and m[1] > 0.5:
            detail = f"pon({tile_str(pending)})" if pending is not None else ""
            return ("PON", float(m[1]), detail)

        chi_mask = get_chow_mask(state, phase, pending, ops)
        m = _normalized(head_probs["chow"], chi_mask)
        if m is not None and chi_mask[1:].sum() > 0:
            best_variant = int(np.argmax(m[1:])) + 1
            if m[best_variant] > 0.5:
                labels = {CHOW_LOWER: "lower", CHOW_MID: "mid", CHOW_UPPER: "upper"}
                return ("CHI", float(m[best_variant]),
                        f"chi {labels.get(best_variant, '?')}")

        return ("PASSER", 1.0, "")

    if phase == "DISCARD":
        kan_mask = get_kong_mask(state, phase, pending, ops)
        m = _normalized(head_probs["kong"], kan_mask)
        if m is not None and m[1] > 0.5:
            detail = ""
            for t in range(N_TILES):
                if state.hand_counts[t] >= 4:
                    detail = f"ankan({tile_str(t)})"
                    break
            if not detail:
                for meld in state.melds:
                    if isinstance(meld, dict) and meld.get("type") == "pon":
                        t = meld["tiles"][0]
                        if state.hand_counts[t] >= 1:
                            detail = f"shouminkan({tile_str(t)})"
                            break
            return ("KAN", float(m[1]), detail)

        riichi_mask = get_riichi_mask(state, phase)
        m = _normalized(head_probs["riichi"], riichi_mask)
        if m is not None and m[1] > 0.5:
            riichi_discard_str = _best_riichi_discard(state, head_probs)
            detail = f"-> puis defausser {riichi_discard_str}" if riichi_discard_str else ""
            return ("RIICHI", float(m[1]), detail)

        discard_mask = get_discard_mask(state)
        discard_probs = _normalized(head_probs["discard"], discard_mask)
        if discard_probs is not None:
            best_idx = int(np.argmax(discard_probs))
            return ("DISCARD", float(discard_probs[best_idx]),
                    disp_tile(state, best_idx))
        return None

    return None


# ═══════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════

def show_all_heads(state, head_probs, phase, inference_tag=""):
    hand_str = disp_hand(state.hand, state.red_hand)
    meld_str = " ".join(
        f"[{' '.join(tile_str(t) for t in m['tiles'])}]" for m in state.melds
    )
    pending = state.last_pending_tile
    ops = state.last_ops

    ts_now = datetime.now().strftime("%H:%M:%S")
    pmcpa_tag = ""
    if _pmcpa is not None:
        if _pmcpa.is_adapted:
            pmcpa_tag = "  [pMCPA adapte]"
        elif _pmcpa._thread and _pmcpa._thread.is_alive():
            pmcpa_tag = "  [pMCPA en cours...]"
        else:
            pmcpa_tag = "  [pMCPA offline]"

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  [{ts_now}] TON TOUR  (phase: {phase}){pmcpa_tag}", flush=True)
    print(f"  Main : {hand_str}", flush=True)
    if meld_str:
        print(f"  Melds: {meld_str}", flush=True)
    print(f"  ({len(state.hand)} tuiles + {len(state.melds)} meld(s))",
          flush=True)

    total_discards = sum(len(d) for d in state.discards)
    if state.my_seat is not None:
        my_disc = len(state.discards[state.my_seat])
        opp_disc = [len(state.discards[(state.my_seat + 1 + i) % 4])
                    for i in range(3)]
    else:
        my_disc = "?"
        opp_disc = "?"
    print(f"  [DBG] Defausses: total={total_discards} "
          f"(toi={my_disc}, opp={opp_disc})", flush=True)
    print(f"  [DBG] Mur restant: {state.wall_remaining} tuiles", flush=True)
    print(f"  [DBG] furiten={state.furiten} "
          f"opp_hand_sizes={state.opp_hand_sizes}", flush=True)
    if state.furiten:
        kinds = [k for k, v in (("permanent", state.furiten_perm),
                                ("temporaire", state.furiten_temp),
                                ("riichi", state.furiten_riichi)) if v]
        print(f"  [FURITEN] {' + '.join(kinds)}", flush=True)
    print("-" * 70, flush=True)

    if phase == "DISCARD":
        probs = head_probs["discard"]
        mask = get_discard_mask(state)
        m = _normalized(probs, mask)
        if m is not None:
            top5 = np.argsort(-m)[:5]
            top5_str = "  ".join(f"{disp_tile(state, int(t))}({m[t]:.2f})"
                                  for t in top5 if m[t] > 0.01)
            print(f"  DISCARD  {top5_str}", flush=True)

        probs = head_probs["kong"]
        mask = get_kong_mask(state, phase, pending, ops)
        m = _normalized(probs, mask)
        legal = "oui" if mask[1] > 0 else "non"
        if m is not None:
            print(f"  KAN      passer({m[0]:.2f})  kan({m[1]:.2f})  "
                  f"[kan legal: {legal}]", flush=True)
        else:
            print(f"  KAN      [kan legal: {legal}]", flush=True)

        probs = head_probs["riichi"]
        mask = get_riichi_mask(state, phase)
        m = _normalized(probs, mask)
        legal = "oui" if mask[1] > 0 else "non"
        if m is not None:
            print(f"  RIICHI   passer({m[0]:.2f})  riichi({m[1]:.2f})  "
                  f"[riichi legal: {legal}]", flush=True)
        else:
            print(f"  RIICHI   [riichi legal: {legal}]", flush=True)

    elif phase == "CALL":
        ron_mask = get_ron_mask(state, phase, pending, ops)
        legal = "oui" if ron_mask[1] > 0 else "non"
        detail = (f"  (sur {tile_str(pending)})"
                  if pending is not None and pending >= 0 else "")
        print(f"  RON      [rule-based, ron legal: {legal}]{detail}", flush=True)

        probs = head_probs["kong"]
        mask = get_kong_mask(state, phase, pending, ops)
        m = _normalized(probs, mask)
        legal = "oui" if mask[1] > 0 else "non"
        if m is not None:
            print(f"  KAN      passer({m[0]:.2f})  kan({m[1]:.2f})  "
                  f"[kan legal: {legal}]", flush=True)
        else:
            print(f"  KAN      [kan legal: {legal}]", flush=True)

        probs = head_probs["pong"]
        mask = get_pong_mask(state, phase, pending, ops)
        m = _normalized(probs, mask)
        legal = "oui" if mask[1] > 0 else "non"
        if m is not None:
            print(f"  PON      passer({m[0]:.2f})  pon({m[1]:.2f})  "
                  f"[pon legal: {legal}]", flush=True)
        else:
            print(f"  PON      [pon legal: {legal}]", flush=True)

        probs = head_probs["chow"]
        mask = get_chow_mask(state, phase, pending, ops)
        m = _normalized(probs, mask)
        legal = "oui" if mask[1:].sum() > 0 else "non"
        if m is not None:
            print(f"  CHI      passer({m[0]:.2f})  lower({m[1]:.2f})  "
                  f"mid({m[2]:.2f})  upper({m[3]:.2f})  "
                  f"[chi legal: {legal}]", flush=True)
        else:
            print(f"  CHI      [chi legal: {legal}]", flush=True)

    print("-" * 70, flush=True)
    final = compute_final_recommendation(state, head_probs, phase)
    if final is None:
        print(f"  * ACTION : (rien a faire)", flush=True)
    else:
        name, proba, detail = final
        suffix = f" {detail}" if detail else ""
        print(f"  * ACTION : {name}{suffix}  (p={proba:.2f})", flush=True)

    print("=" * 70, flush=True)
    print(flush=True)


# ═══════════════════════════════════════════════════════════════════════
# DECISION
# ═══════════════════════════════════════════════════════════════════════

def decide(state):
    models = load_sl_models()
    if models is None:
        state.pending_action = None
        return

    phase = state.pending_action.get("name") if state.pending_action else None
    state.pending_action = None

    if phase not in ("DISCARD", "CALL"):
        return

    try:
        obs = build_observation(state)
    except Exception as e:
        print(f"[ERREUR] build_observation : {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    if obs.shape != (N_CHANNELS, N_TILES):
        print(f"[ERREUR] obs.shape={obs.shape} != ({N_CHANNELS}, {N_TILES})",
              flush=True)
        return

    t0 = time.time()
    # [pMCPA] Utilise la policy adaptee si disponible, sinon offline
    if _pmcpa is not None and _pmcpa.is_adapted:
        try:
            head_probs = _pmcpa.run_all_heads_adapted(obs)
            adapted_tag = "[pMCPA]"
        except Exception as e_inf:
            print(f"[WARN] pMCPA inference : {e_inf}", flush=True)
            head_probs = run_all_heads(models, obs)
            adapted_tag = "[offline]"
    else:
        head_probs = run_all_heads(models, obs)
        if _pmcpa is not None:
            adapted_tag = "[offline - adaptation en cours]"
        else:
            adapted_tag = ""
    t1 = time.time()

    show_all_heads(state, head_probs, phase, inference_tag=adapted_tag)
    print(f"  (inference {adapted_tag} : {(t1-t0)*1000:.1f} ms)", flush=True)

    state.last_claimed_tile = None
    state.last_chi_tiles = None
    if phase == "DISCARD":
        state.last_pending_tile = None
        state.last_ops = None


# ═══════════════════════════════════════════════════════════════════════
# ADDON
# ═══════════════════════════════════════════════════════════════════════

_state = None


def _say(msg):
    ts_now = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts_now}] {msg}", flush=True)


class SnifferAddon:
    def __init__(self):
        global _state
        _state = GameState()
        _say("Sniffer demarre, en attente de partie...")
        if USE_SL:
            _say(f"Checkpoint dir : {CHECKPOINT_DIR}")
            _say(f"Look-ahead SL : {'ON' if USE_LOOKAHEAD else 'OFF (recommande)'}")
            _say(f"pMCPA : {'ON' if PMCPA_ENABLE else 'OFF'}")
            load_sl_models()

    def websocket_message(self, flow):
        if flow.websocket is None:
            return
        try:
            msg = flow.websocket.messages[-1]
        except (IndexError, AttributeError):
            return
        if msg.is_text or msg.from_client:
            return

        try:
            method, data, action_name = parse_message(msg.content)
        except Exception:
            return

        if method is None or action_name is None or not data:
            return

        try:
            _state.handle(method, data, action_name)
        except Exception as e:
            _say(f"[ERREUR] handle : {e}")
            import traceback
            traceback.print_exc()

        if _state.pending_action is not None:
            try:
                decide(_state)
            except Exception as e:
                _say(f"[ERREUR] decide : {e}")
                import traceback
                traceback.print_exc()
                _state.pending_action = None


addons = [SnifferAddon()]