"""Constantes et utilitaires tuiles/shanten partages par TOUTES les etapes."""
from functools import lru_cache

from mahjong.shanten import Shanten

N_TILES = 34
COPIES_PER_TILE = 4
HAND_SIZE = 13

_shanten_calc = Shanten()


@lru_cache(maxsize=200_000)
def _shanten_cached(counts_tuple):
    """Cache LRU : 200k mains distinctes en memoire (~200 Mo max)."""
    return _shanten_calc.calculate_shanten(
        list(counts_tuple),
        use_chiitoitsu=True,
        use_kokushi=False,
    )


def shanten_of(counts34):
    """Shanten standard (melds+pair). -1 = main complete (agari)."""
    return _shanten_cached(tuple(int(x) for x in counts34))