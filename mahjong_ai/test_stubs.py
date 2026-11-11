"""Stubs partages par les tests (mahjong / torch / zstandard).

Importer ce module AVANT mahjong_ai.* : si les vraies libs sont absentes, des
stubs minimaux sont installes (REAL_MAHJONG / REAL_TORCH disent lesquelles
sont reelles). Avec les vraies libs, rien n'est remplace.
"""
import functools
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # pour `import mahjong_ai`

# --------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------- #
def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _sets_ok(c):
    i = 0
    while i < 34 and c[i] == 0:
        i += 1
    if i == 34:
        return True
    if c[i] >= 3:
        c[i] -= 3
        r = _sets_ok(c)
        c[i] += 3
        if r:
            return True
    if i < 27 and i % 9 <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
        c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
        r = _sets_ok(c)
        c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
        return r
    return False


def _agari(counts, kokushi):
    c = [int(x) for x in counts]
    if sum(c) % 3 != 2:
        return False
    for p in range(34):
        if c[p] >= 2:
            c[p] -= 2
            ok = _sets_ok(c)
            c[p] += 2
            if ok:
                return True
    if sum(c) == 14 and sum(1 for x in c if x == 2) == 7:
        return True
    if kokushi:
        yao = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        if all(c[t] >= 1 for t in yao) and sum(c[t] for t in yao) == 14:
            return True
    return False


@functools.lru_cache(maxsize=500_000)
def _shanten_cached(tiles, kokushi):
    c = list(tiles)
    if _agari(c, kokushi):
        return -1
    for t in range(34):
        if c[t] < 4:
            c[t] += 1
            ok = _agari(c, kokushi)
            c[t] -= 1
            if ok:
                return 0
    return 1


class _StubShanten:
    def calculate_shanten(self, tiles, use_chiitoitsu=True, use_kokushi=True):
        return _shanten_cached(tuple(int(x) for x in tiles), bool(use_kokushi))


AKA_IDS = {16, 52, 88}


class _StubCalc:
    LAST = None          # derniere evaluation : dict(tiles, win_tile, aka)
    CALLS = []

    @staticmethod
    def estimate_hand_value(tiles, win_tile, melds=None,
                            dora_indicators=None, ura_dora_indicators=None,
                            config=None):
        tiles = list(tiles)
        assert len(set(tiles)) == len(tiles), f"ids 136 dupliques : {sorted(tiles)}"
        assert all(0 <= t < 136 for t in tiles), f"id hors plage : {sorted(tiles)}"
        assert win_tile in tiles, f"win_tile {win_tile} absent de {sorted(tiles)}"
        use_aka = bool(getattr(config.options, "has_aka_dora", False))
        aka = sum(1 for t in tiles if t in AKA_IDS) if use_aka else 0
        rec = {"tiles": tiles, "win_tile": win_tile, "aka": aka,
               "use_aka": use_aka}
        _StubCalc.LAST = rec
        _StubCalc.CALLS.append(rec)
        han = 1 + aka
        total = 1000 * han
        return types.SimpleNamespace(
            error=None, han=han, fu=30, yaku=[],
            cost={"total": total, "main": total // 2, "additional": total // 4,
                  "kyoutaku_bonus": 0})


def install_stubs():
    global REAL_MAHJONG, REAL_TORCH
    try:
        import mahjong  # noqa: F401
        REAL_MAHJONG = True
    except Exception:
        REAL_MAHJONG = False
        _stub("mahjong").__path__ = []
        _stub("mahjong.constants", EAST=27, SOUTH=28, WEST=29, NORTH=30)
        _stub("mahjong.shanten", Shanten=_StubShanten)
        _stub("mahjong.hand_calculating").__path__ = []
        _stub("mahjong.hand_calculating.hand", HandCalculator=_StubCalc)

        class OptionalRules:
            def __init__(self, **kw):
                self.has_aka_dora = False
                self.__dict__.update(kw)

        class HandConfig:
            def __init__(self, **kw):
                self.options = OptionalRules()
                self.__dict__.update(kw)

        _stub("mahjong.hand_calculating.hand_config",
              HandConfig=HandConfig, OptionalRules=OptionalRules)

        class Meld:
            CHI, PON, KAN, SHOUMINKAN = "chi", "pon", "kan", "shouminkan"

            def __init__(self, meld_type=None, tiles=None, opened=True):
                self.type, self.tiles, self.opened = meld_type, tiles, opened

        _stub("mahjong.meld", Meld=Meld)
    try:
        import torch  # noqa: F401
        REAL_TORCH = True
    except Exception:
        REAL_TORCH = False

        class _Arr(np.ndarray):
            def narrow(self, dim, start, length):
                sl = [slice(None)] * self.ndim
                sl[dim] = slice(start, start + length)
                return np.asarray(self)[tuple(sl)].view(_Arr)

            def new_zeros(self, shape):
                return np.zeros(shape, dtype=self.dtype).view(_Arr)

            def contiguous(self):
                return np.ascontiguousarray(self).view(_Arr)

        class _Module:
            def __init__(self, *a, **k):
                pass

            def _load_from_state_dict(self, *a, **k):
                pass

        nn = _stub("torch.nn", Module=_Module, Conv2d=object, BatchNorm2d=object,
                   ReLU=object, ModuleList=list, Linear=object, Sequential=object,
                   Flatten=object)
        _stub("torch", nn=nn, cat=lambda ps, dim=0: np.concatenate(
            [np.asarray(p) for p in ps], axis=dim).view(_Arr), _Arr=_Arr)
    try:
        import zstandard  # noqa: F401
    except Exception:
        _stub("zstandard")


install_stubs()

