"""
test_furiten_sniffer.py -- Scenarios de furiten rejoues message par message
dans GameState.handle() du sniffer, puis lus via state.furiten et
get_ron_mask() (= ce que le bot verrait reellement).

Usage :
    py test_furiten_sniffer.py [chemin/vers/sniffer_suphx.py]

Isolation : torch / protobuf / ms / mahjong_ai.envs.* sont remplaces par des
stubs (seul le code de furiten est teste). Si la lib `mahjong` est
installee, c'est elle qui sert au calcul des attentes ; sinon un oracle
independant (decomposition standard + chiitoi + kokushi) la remplace, avec
la meme convention que la lib (shanten_of : kokushi OFF ; Shanten : ON).

Regles testees (Mahjong Soul / Tenhou) :
  - furiten permanent (sutehai) : une tuile d'attente est dans MA riviere,
    y compris si elle a ete prise par un autre joueur (chi/pon/kan) ;
  - furiten temporaire : ron manque (avec yaku) -> interdit jusqu'a MA
    prochaine defausse ;
  - furiten riichi : ron manque en riichi -> interdit jusqu'a la fin de la
    manche ;
  - chankan manque = ron manque ;
  - remise a zero a chaque nouvelle manche.
"""
import contextlib
import importlib.util
import io
import os
import sys
import types

import numpy as np

os.environ["SL_ENABLE"] = "0"
os.environ["SNIFFER_TRACE"] = "0"
os.environ["PMCPA_ENABLE"] = "0"

# ---------------------------------------------------------------- oracle
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


def agari(counts, kokushi):
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
    if sum(c) == 14 and sorted(c, reverse=True)[:7] == [2] * 7 \
            and sum(1 for x in c if x == 2) == 7:
        return True                                   # chiitoitsu
    if kokushi:
        yao = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        if all(c[t] >= 1 for t in yao) and sum(c[t] for t in yao) == 14:
            return True
    return False


class _StubShanten:
    def calculate_shanten(self, tiles, use_chiitoitsu=True, use_kokushi=True):
        c = list(tiles)
        if agari(c, use_kokushi):
            return -1
        for t in range(34):
            if c[t] < 4:
                c[t] += 1
                ok = agari(c, use_kokushi)
                c[t] -= 1
                if ok:
                    return 0
        return 1


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def load_sniffer(path):
    try:
        from mahjong.shanten import Shanten as RealShanten
        real = True
    except Exception:
        RealShanten = _StubShanten
        real = False
        pk = _stub("mahjong")
        pk.__path__ = []
        _stub("mahjong.shanten", Shanten=_StubShanten)
    calc = RealShanten()

    _stub("torch")
    _stub("google"); sys.modules["google"].__path__ = []
    _stub("google.protobuf"); sys.modules["google.protobuf"].__path__ = []
    _stub("google.protobuf.json_format", MessageToDict=lambda *a, **k: {})
    _stub("ms"); sys.modules["ms"].__path__ = []
    _stub("ms.protocol_pb2")
    sys.modules["ms"].protocol_pb2 = sys.modules["ms.protocol_pb2"]
    pk = _stub("mahjong_ai"); pk.__path__ = []
    pe = _stub("mahjong_ai.envs"); pe.__path__ = []
    _stub("mahjong_ai.envs.suphx_features", build_suphx_obs=lambda **k: None,
          N_CHANNELS=1, N_TILES=34, RED_TYPES=(4, 13, 22))
    _stub("mahjong_ai.envs.multi_agent", DISCARD=0, PONG=1, CHOW=2, KONG=3,
          RIICHI=4, CHOW_PASS=0, CHOW_LOWER=1, CHOW_MID=2, CHOW_UPPER=3)
    _stub("mahjong_ai.envs.tiles",
          shanten_of=lambda c: calc.calculate_shanten(
              [int(x) for x in c], use_chiitoitsu=True, use_kokushi=False))
    spec = importlib.util.spec_from_file_location("sniffer_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sniffer_under_test"] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod, real


# ---------------------------------------------------------------- harness
YAKU = [True]


class Game:
    """Rejoue des messages comme le serveur : moi = siege 0 (dealer)."""

    def __init__(self, mod):
        self.m = mod
        mod.has_yaku = lambda state, t: YAKU[0]
        self.s = mod.GameState()
        mod._state = self.s

    def send(self, name, **data):
        with contextlib.redirect_stdout(io.StringIO()):
            self.s.handle(None, data, name)

    def new_round(self, hand14):
        self.send("ActionNewRound", chang=0, ju=0, ben=0, tiles=list(hand14),
                  scores=[25000] * 4, doras=["1m"], left_tile_count=69)

    def me_discard(self, tile, riichi=False):
        self.send("ActionDiscardTile", seat=0, tile=tile, is_liqi=riichi)

    def me_draw(self, tile):
        self.send("ActionDealTile", seat=0, tile=tile, operation={"x": 1})

    def opp(self, seat, tile):
        """L'adversaire `seat` pioche (message DealTile) puis defausse."""
        self.send("ActionDealTile", seat=seat, left_tile_count=60)
        self.send("ActionDiscardTile", seat=seat, tile=tile, operation={},
                  is_liqi=False)

    def pon_by(self, seat, tile):
        self.send("ActionChiPengGang", seat=seat, tiles=[tile] * 3)

    def kakan_by(self, seat, tile):
        self.send("ActionAnGangAddGang", seat=seat, type="add_gang", tile=tile)

    def ankan_me(self, tile):
        self.send("ActionAnGangAddGang", seat=0, type="ankan", tile=tile)

    @property
    def furiten(self):
        return bool(self.s.furiten)

    def ron_ok(self, tile):
        t = self.m.tile_to_34(tile)
        with contextlib.redirect_stdout(io.StringIO()):
            mask = self.m.get_ron_mask(self.s, "CALL", t, None)
        return bool(mask[1] > 0)


RESULTS = []


def check(scn, label, cond):
    RESULTS.append((scn, label, bool(cond)))
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}")


H1 = ['2m', '3m', '4m', '5p', '6p', '7p', '2s', '3s', '4s',
      '7m', '7m', '6s', '7s']            # attente 5s / 8s (ryanmen 6s7s)


def junk(g, n=3, tiles=('9p', '9p', '1z')):
    for i in range(n):
        g.opp(1 + i % 3, tiles[i % len(tiles)])


def scenarios(mod):
    YAKU[0] = True

    print("\n=== S1 furiten permanent : 5s dans ma riviere, attente 5s/8s ===")
    g = Game(mod); g.new_round(H1 + ['5s']); g.me_discard('5s')
    check("S1", "flag furiten = True", g.furiten)
    g.opp(1, '8s')
    check("S1", "ron sur 8s refuse (l'autre attente) ", not g.ron_ok('8s'))

    print("\n=== S1b controle : riviere sans attente -> pas furiten ===")
    g = Game(mod); g.new_round(H1 + ['9p']); g.me_discard('9p')
    check("S1b", "flag furiten = False", not g.furiten)
    g.opp(1, '8s')
    check("S1b", "ron sur 8s autorise", g.ron_ok('8s'))

    print("\n=== S2 furiten par l'autre attente (8s dans la riviere) ===")
    g = Game(mod); g.new_round(H1 + ['8s']); g.me_discard('8s')
    check("S2", "flag furiten = True", g.furiten)
    g.opp(1, '5s')
    check("S2", "ron sur 5s refuse", not g.ron_ok('5s'))

    print("\n=== S3 tuile defaussee AVANT d'etre tenpai, attente ensuite ===")
    g = Game(mod)
    g.new_round(['2m', '3m', '4m', '5p', '6p', '7p', '2s', '3s', '4s',
                 '7m', '7m', '6s', '5s', '1z'])
    g.me_discard('5s')
    check("S3", "pas tenpai -> pas furiten", not g.furiten)
    junk(g)
    g.me_draw('7s'); g.me_discard('1z')
    check("S3", "tenpai 5s/8s avec 5s en riviere -> furiten", g.furiten)

    print("\n=== S3b le flag suit la main (attente change) ===")
    g = Game(mod); g.new_round(H1 + ['5s']); g.me_discard('5s')
    check("S3b", "furiten au depart", g.furiten)
    junk(g)
    g.me_draw('1z'); g.me_discard('7s')
    check("S3b", "main cassee (plus tenpai) -> plus furiten", not g.furiten)
    junk(g)
    g.me_draw('7s'); g.me_discard('1z')
    check("S3b", "tenpai de nouveau -> furiten de nouveau", g.furiten)

    print("\n=== S5 tuile de ma riviere PRISE par un adversaire compte encore ===")
    g = Game(mod); g.new_round(H1 + ['5s']); g.me_discard('5s')
    g.pon_by(1, '5s'); g.opp(1, '9p')
    check("S5", "5s pon par seat1 : toujours furiten (regle reelle)",
          g.furiten)
    g.opp(2, '8s')
    check("S5", "ron sur 8s refuse", not g.ron_ok('8s'))

    print("\n=== S6 furiten temporaire : ron manque puis meme tour ===")
    g = Game(mod); g.new_round(H1 + ['9p']); g.me_discard('9p')
    g.opp(1, '8s')
    check("S6", "8s : ron possible (rien de manque encore)", g.ron_ok('8s'))
    g.opp(2, '5s')                       # j'ai passe sur 8s
    check("S6", "flag furiten = True apres le ron manque", g.furiten)
    check("S6", "ron sur 5s refuse (furiten temporaire)",
          not g.ron_ok('5s'))

    print("\n=== S7 le furiten temporaire tombe a MA defausse ===")
    g.opp(3, '9p')
    check("S7", "toujours furiten avant ma pioche", g.furiten)
    g.me_draw('9s'); g.me_discard('9s')
    check("S7", "apres ma defausse : plus furiten", not g.furiten)
    g.opp(1, '5s')
    check("S7", "ron sur 5s de nouveau autorise", g.ron_ok('5s'))

    print("\n=== S8 [regle supposee] ron manque SANS yaku : pas de furiten ===")
    g = Game(mod); g.new_round(H1 + ['9p']); g.me_discard('9p')
    YAKU[0] = False
    g.opp(1, '8s')                       # tuile gagnante mais sans yaku
    g.opp(2, '9p')
    YAKU[0] = True
    g.opp(3, '5s')
    check("S8", "5s : ron autorise (8s sans yaku n'a pas cree de furiten)",
          g.ron_ok('5s'))

    print("\n=== S9 furiten riichi : ron manque en riichi = jusqu'a la fin ===")
    g = Game(mod); g.new_round(H1 + ['9p']); g.me_discard('9p', riichi=True)
    check("S9", "riichi pose, pas furiten", not g.furiten)
    g.opp(1, '8s')                       # je passe
    g.opp(2, '9p')
    g.opp(3, '5s')
    check("S9", "ron sur 5s refuse (furiten riichi)", not g.ron_ok('5s'))
    g.me_draw('3p')
    check("S9", "flag toujours True a ma pioche (14 tuiles)", g.furiten)
    g.me_discard('3p')
    g.opp(1, '5s')
    check("S9", "apres ma defausse : toujours refuse (permanent)",
          not g.ron_ok('5s'))

    print("\n=== S9b controle : riichi, tuile NON gagnante passee ===")
    g = Game(mod); g.new_round(H1 + ['9p']); g.me_discard('9p', riichi=True)
    g.opp(1, '3p'); g.opp(2, '9p'); g.opp(3, '5s')
    check("S9b", "ron sur 5s autorise", g.ron_ok('5s'))

    print("\n=== S10 remise a zero a la nouvelle manche ===")
    g = Game(mod); g.new_round(H1 + ['9p']); g.me_discard('9p', riichi=True)
    g.opp(1, '8s'); g.opp(2, '9p')
    check("S10", "furiten riichi actif", g.furiten)
    g.new_round(H1 + ['9p']); g.me_discard('9p')
    check("S10", "nouvelle manche : furiten remis a False", not g.furiten)
    g.opp(1, '8s')
    check("S10", "ron sur 8s autorise", g.ron_ok('8s'))

    print("\n=== S11 chankan manque = furiten temporaire ===")
    g = Game(mod)
    g.new_round(['2m', '3m', '4m', '5p', '6p', '7p', '2s', '3s', '4s',
                 '7m', '7m', '6s', '1z', '9p'])
    g.me_discard('9p')
    g.opp(1, '8s'); g.pon_by(2, '8s'); g.opp(2, '9p')   # pon avant mon tenpai
    g.opp(3, '9p')
    g.me_draw('7s'); g.me_discard('1z')
    check("S11", "tenpai 5s/8s, pas furiten", not g.furiten)
    g.send("ActionDealTile", seat=1, left_tile_count=50)
    g.send("ActionDealTile", seat=2, left_tile_count=49)
    g.kakan_by(2, '8s')                  # chankan possible, je passe
    g.send("ActionDealTile", seat=2, left_tile_count=48)   # rinshan
    g.send("ActionDiscardTile", seat=2, tile='9p', operation={}, is_liqi=False)
    g.opp(3, '5s')
    check("S11", "ron sur 5s refuse apres chankan manque", not g.ron_ok('5s'))

    print("\n=== S13 chiitoitsu : furiten permanent ===")
    g = Game(mod)
    g.new_round(['1m', '1m', '3m', '3m', '5p', '5p', '7p', '7p', '2s', '2s',
                 '9s', '9s', '4m', '6m'])
    g.me_discard('4m')                   # attente 6m ; 4m en riviere
    check("S13", "attente 6m, riviere {4m} : pas furiten", not g.furiten)
    junk(g)
    g.me_draw('4m'); g.me_discard('6m')  # attente 4m ; 4m en riviere
    check("S13", "attente 4m, 4m en riviere : furiten", g.furiten)

    print("\n=== S14 kokushi 13 attentes : une terminale en riviere ===")
    g = Game(mod)
    g.new_round(['1m', '9m', '1p', '9p', '1s', '9s', '1z', '2z', '3z', '4z',
                 '5z', '6z', '7z', '1m'])
    g.me_discard('1m')                   # 13 orphelins distincts
    check("S14", "flag furiten = True (1m est une attente)", g.furiten)
    g.opp(1, '9m')
    check("S14", "ron refuse sur 9m", not g.ron_ok('9m'))

    print("\n=== S15 ankan dans la main : arithmetique des tuiles ===")
    g = Game(mod)
    g.new_round(['1z', '1z', '1z', '1z', '2m', '3m', '4m', '5p', '6p', '7p',
                 '2s', '3s', '9m', '9m'])
    g.ankan_me('1z'); g.me_draw('1s'); g.me_discard('1s')
    check("S15", "ankan + attente 1s/4s, 1s en riviere : furiten", g.furiten)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sniffer_suphx.py"
    mod, real = load_sniffer(path)
    print(f"[test] fichier : {path}")
    print(f"[test] lib mahjong reelle : {'oui' if real else 'non (oracle)'}")
    scenarios(mod)
    n_fail = sum(1 for _, _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} controles OK")
    if n_fail:
        print("ECHECS :")
        for scn, label, ok in RESULTS:
            if not ok:
                print(f"  - {scn} : {label}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
