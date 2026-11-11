"""
test_red_sniffer.py -- Suivi des 5 rouges (0m, 0p, 0s) dans sniffer_suphx.py.

Usage :  py test_red_sniffer.py [chemin/vers/sniffer_suphx.py]
(a placer a cote de test_furiten_sniffer.py : reutilise son harnais, qui
remplace torch / protobuf / ms / mahjong_ai.* par des stubs.)

Rejoue des messages Mahjong Soul contenant des "0m"/"0p"/"0s" puis lit
state.red_state() -- exactement ce qui est donne a build_suphx_obs.
"""
import sys

import test_furiten_sniffer as T
from test_furiten_sniffer import Game, load_sniffer

RESULTS = []


def check(label, cond):
    RESULTS.append((label, bool(cond)))
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}")


BASE = ['1m', '2m', '3m', '6m', '7m', '8m', '1p', '2p', '3p', '1s', '2s', '3s', '1z']


def hand(*extra):
    return (list(extra) + BASE)[:13]


def run(mod):
    print("\n=== R0 reconnaissance des rouges ===")
    check("0m/0p/0s -> 0/1/2, 5m/E/None -> -1",
          [mod.tile_red_index(x) for x in ("0m", "0p", "0s", "5m", "1z", None)]
          == [0, 1, 2, -1, -1, -1])
    check("tile_to_34('0p') == 13 (le rouge reste un 5p)", mod.tile_to_34("0p") == 13)

    print("\n=== R1 main initiale + indicateur rouge ===")
    g = Game(mod)
    g.send("ActionNewRound", chang=0, ju=0, ben=0, scores=[25000] * 4,
           tiles=hand('0m', '0s') + ['9p'], doras=['0p'], left_tile_count=69)
    s = g.s
    check("red_hand = [5m oui, 5p non, 5s oui]", s.red_hand.tolist() == [True, False, True])
    check("indicateur de dora rouge (5p)", s.red_indicator.tolist() == [False, True, False])
    check("dora_indicators = [5p]", s.dora_indicators == [13])
    rs = s.red_state()
    check("red_state['hand'][mon siege] = mes rouges",
          rs['hand'][0].tolist() == [True, False, True])

    print("\n=== R2 pioche rouge puis defausse ===")
    g.me_draw('0p')
    check("pioche 0p : red_hand[5p]", s.red_hand[1])
    g.me_discard('0p')
    check("defausse 0p : plus en main", not s.red_hand[1])
    check("defausse 0p : en riviere de mon siege (0)", s.red_river[0][1])
    check("last_discard_red = 1", s.last_discard_red == 1)

    print("\n=== R3 defausse rouge d'un adversaire, puis pon par un autre ===")
    g.opp(2, '0s')
    check("0s defausse par seat2 : red_river[2][5s]", s.red_river[2][2])
    check("pending (tuile en attente d'appel) = rouge 5s", s.red_state()['pending'] == 2)
    g.send("ActionChiPengGang", seat=1, tiles=['0s', '5s', '5s'])
    check("pon par seat1 : rouge de la riviere -> meld de seat1",
          (not s.red_river[2][2]) and s.red_meld[1][2])

    print("\n=== R4 mon pon avec mon 5m rouge ===")
    g = Game(mod)
    g.send("ActionNewRound", chang=0, ju=0, ben=0, scores=[25000] * 4,
           tiles=hand('0m', '5m') + ['9p'], doras=['1m'], left_tile_count=69)
    s = g.s
    g.me_discard('9p')
    g.opp(1, '5m')
    check("5m normal defausse : pending non rouge", s.red_state()['pending'] == -1)
    g.send("ActionChiPengGang", seat=0, tiles=['5m', '5m', '0m'])
    check("mon pon (rouge de ma main) : red_meld[0][5m], plus en main",
          s.red_meld[0][0] and not s.red_hand[0])

    print("\n=== R5 ankan : le rouge fait partie du kan ===")
    g = Game(mod)
    g.send("ActionNewRound", chang=0, ju=0, ben=0, scores=[25000] * 4,
           tiles=hand('0p', '5p', '5p') + ['5p'], doras=['1m'], left_tile_count=69)
    s = g.s
    g.ankan_me('5p')
    check("ankan 5p : red_meld[0][5p] et plus en main",
          s.red_meld[0][1] and not s.red_hand[1])

    print("\n=== R6 kakan : tuile ajoutee rouge (moi) ===")
    g = Game(mod)
    g.send("ActionNewRound", chang=0, ju=0, ben=0, scores=[25000] * 4,
           tiles=hand('5s', '5s') + ['9p'], doras=['1m'], left_tile_count=69)
    s = g.s
    g.me_discard('9p')
    g.opp(1, '5s')
    g.send("ActionChiPengGang", seat=0, tiles=['5s', '5s', '5s'])
    g.me_discard('1z')
    g.me_draw('0s')
    g.send("ActionAnGangAddGang", seat=0, type="add_gang", tile='0s')
    check("kakan avec 0s : rouge dans le meld",
          s.red_meld[0][2] and not s.red_hand[2])

    print("\n=== R6b kakan adverse avec rouge ===")
    g = Game(mod)
    g.new_round(hand() + ['9p'])
    s = g.s
    g.me_discard('9p')
    g.opp(1, '5m')
    g.send("ActionChiPengGang", seat=2, tiles=['5m', '5m', '5m'])
    g.opp(2, '9p')
    g.send("ActionDealTile", seat=2, left_tile_count=50)
    g.send("ActionAnGangAddGang", seat=2, type="add_gang", tile='0m')
    check("kakan adverse avec 0m : red_meld[seat2][5m]", s.red_meld[2][0])
    check("pending du chankan = rouge 5m", s.last_discard_red == 0)

    print("\n=== R7 nouvelle manche : tout est remis a zero ===")
    g = Game(mod)
    g.send("ActionNewRound", chang=0, ju=0, ben=0, scores=[25000] * 4,
           tiles=hand('0m') + ['9p'], doras=['0s'], left_tile_count=69)
    g.me_discard('0m')
    g.new_round(hand() + ['9p'])
    s = g.s
    check("red_hand/river/meld/indicator remis a zero",
          not s.red_hand.any() and not any(r.any() for r in s.red_river)
          and not any(m.any() for m in s.red_meld) and not s.red_indicator.any()
          and s.last_discard_red == -1)

    print("\n=== R8 nouveau dora (kan) : relu depuis `doras` ===")
    g = Game(mod)
    g.new_round(hand() + ['9p'])
    s = g.s
    check("dora initial = 1m", s.dora_indicators == [0])
    g.send("ActionDiscardTile", seat=0, tile='9p', is_liqi=False,
           doras=['1m', '0p'])
    check("apres la defausse : 2 indicateurs, le 2e est un 5p rouge",
          s.dora_indicators == [0, 13] and s.red_indicator[1])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sniffer_suphx.py"
    mod, _ = load_sniffer(path)
    print(f"[test] fichier : {path}")
    run(mod)
    bad = [l for l, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} controles OK")
    for l in bad:
        print("  - ECHEC :", l)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
