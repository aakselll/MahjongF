"""
Test exhaustif des regles de fin de partie Mahjong Soul dans game_env.py.

[FIX 2025-XX] Corrections des tests apres premiere execution :
    - Tests ou le dealer etait en fait 1er (commentaires faux)
    - Tests ou oorasu s'applique legitimement (qqun a 30k+)
    - Tests ou agari yame prime sur le repeat

Usage :
    py -3.14 -m mahjong_ai.test_end_rules
"""
from unittest.mock import patch

import numpy as np

from mahjong_ai.training.game_env import (
    GameEnv, N_ROUNDS_NORMAL, N_ROUNDS_MAX, MAX_KYOKU,
    TARGET_SCORE, TOBI_THRESHOLD, RANK_REWARDS, _decide_dealer_repeat,
)


_results = {"pass": 0, "fail": 0}


def check(name, env, outcome, dealer_round_reward,
          expected_done, expected_reason,
          dealer_repeats=False, dealer_real=0, tenpai=False):
    shanten_value = 0 if tenpai else 3
    with patch('mahjong_ai.training.game_env.shanten_of',
                return_value=shanten_value):
        done, reason = env._check_game_end(
            outcome, dealer_repeats, dealer_real, dealer_round_reward)

    ok = (done == expected_done and reason == expected_reason)
    mark = "OK" if ok else "XX"
    _results["pass" if ok else "fail"] += 1
    print(f"  [{mark}] {name:55s} -> done={done}, reason={reason}")
    if not ok:
        print(f"       ATTENDU : done={expected_done}, reason={expected_reason}")


def section(title):
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")


def new_env():
    env = GameEnv(seed=42)
    env.round_env._shanten_counts = lambda seat: np.zeros(34, dtype=np.int32)
    return env


# ====================================================================== #
# 1. TOBI
# ====================================================================== #
def test_tobi():
    section("1. TOBI (fin si un joueur passe sous 0 absolu)")

    env = new_env(); env.game_scores = [-25.0, 0.0, 0.0, 0.0]; env.round_num = 1
    check("Score pile a -25 (0 absolu)", env, 'exhaustive', 0.0, True, "tobi")

    env = new_env(); env.game_scores = [-30.0, 0.0, 0.0, 0.0]; env.round_num = 2
    check("Score -30 (sous 0 absolu)", env, 'tsumo', 1.0, True, "tobi")

    env = new_env(); env.game_scores = [-20.0, 0.0, 0.0, 0.0]; env.round_num = 1
    check("Score -20 (au-dessus de 0)", env, 'exhaustive', 0.0, False, None)

    env = new_env(); env.game_scores = [-30.0, 10.0, 10.0, 10.0]; env.round_num = 1
    check("Tobi a East 1", env, 'tsumo', 0.0, True, "tobi")

    env = new_env(); env.game_scores = [10.0, 5.0, 5.0, -30.0]; env.round_num = 4
    env.is_enchousen = False
    check("Tobi prioritaire (dealer 1er en East 4)",
          env, 'tsumo', 1.0, True, "tobi", dealer_real=0)


# ====================================================================== #
# 2. AGARI YAME
# ====================================================================== #
def test_agari_yame():
    section("2. AGARI YAME (all last + dealer 1er + gagne)")

    # --- Cas ou agari yame s'applique ---
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]  # p0=1er
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er, tsumo", env, 'tsumo', 1.5, True, "agari_yame")

    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er, ron", env, 'ron', 2.0, True, "agari_yame")

    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er, multi_ron", env, 'multi_ron', 1.0,
          True, "agari_yame")

    # Sud 4 (enchousen), dealer 1er gagne
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 8; env.is_enchousen = True
    check("Sud 4 (enchousen), dealer 1er, gagne",
          env, 'tsumo', 1.5, True, "agari_yame")

    # --- Cas ou agari yame NE s'applique PAS ---
    # East 4, dealer 1er, exhaustive (pas de win) -> oorasu (qqun >= 30k)
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er, exhaustive (pas agari)",
          env, 'exhaustive', 0.0, True, "oorasu")

    # East 4, dealer 2eme, gagne -> pas de fin (personne >= 30k)
    # [FIX 2026-11] scores [5, 15, ...] = 40k pour p1 -> oorasu legitime
    # (convention : 0.0 = 25 000 pts, TARGET_SCORE 5.0 = 30 000 pts).
    # Ici p1 = 29.5k (1er), dealer p0 = 29k (2eme) : personne a 30k.
    env = new_env(); env.game_scores = [4.0, 4.5, 0.0, -5.0]  # p1=1er
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 2eme, gagne, personne 30k (enchousen)",
          env, 'tsumo', 1.5, False, None, dealer_real=0)

    # East 3 (pas all last) -> pas agari yame
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 3; env.is_enchousen = False
    check("East 3, dealer 1er, gagne (pas all last)",
          env, 'tsumo', 1.5, False, None)

    # East 4, dealer 1er, gagne mais reward <= 0 (pas une vraie win)
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er, gagne mais reward=0",
          env, 'tsumo', 0.0, True, "oorasu")


# ====================================================================== #
# 3. TENPAI YAME
# ====================================================================== #
def test_tenpai_yame():
    section("3. TENPAI YAME (all last + dealer 1er + tenpai sur ryuukyoku)")

    # --- Cas ou tenpai yame s'applique ---
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er, tenpai sur exhaustive",
          env, 'exhaustive', 0.0, True, "tenpai_yame", tenpai=True)

    # Sud 4 (all last enchousen), dealer 1er tenpai
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 8; env.is_enchousen = True
    check("Sud 4, dealer 1er, tenpai sur exhaustive",
          env, 'exhaustive', 0.0, True, "tenpai_yame", tenpai=True)

    # --- Cas ou tenpai yame NE s'applique PAS ---
    # NOT tenpai -> oorasu (qqun >= 30k)
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er, NOT tenpai sur exhaustive",
          env, 'exhaustive', 0.0, True, "oorasu", tenpai=False)

    # outcome=tsumo avec reward=0 -> pas de win -> oorasu
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er, outcome=tsumo mais reward=0",
          env, 'tsumo', 0.0, True, "oorasu", tenpai=True)

    # Dealer 2eme tenpai -> pas de tenpai yame, personne >= 30k -> enchousen
    # [FIX 2026-11] memes scores corriges (p1 = 29.5k < 30k)
    env = new_env(); env.game_scores = [4.0, 4.5, 0.0, -5.0]  # p1=1er
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 2eme, tenpai, personne 30k (enchousen)",
          env, 'exhaustive', 0.0, False, None, tenpai=True)


# ====================================================================== #
# 4. OORASU
# ====================================================================== #
def test_oorasu():
    section("4. OORASU (fin East 4 si quelqu'un a >= 30k)")

    # East 4, !repeat, qqun >= 30k
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, fin, qqun >= 30k", env, 'exhaustive', 0.0,
          True, "oorasu", dealer_repeats=False)

    # East 4, !repeat, personne >= 30k -> enchousen
    env = new_env(); env.game_scores = [3.0, 2.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, fin, personne >= 30k (enchousen)",
          env, 'exhaustive', 0.0, False, None, dealer_repeats=False)

    # East 4, dealer repeat (dealer 1er + gagne) -> agari yame (pas oorasu)
    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 1er repeat + gagne (agari yame)",
          env, 'tsumo', 1.5, True, "agari_yame",
          dealer_repeats=True, dealer_real=0)

    # East 4, dealer 2eme repeat -> pas de fin, continue
    # Scores [5, 15, ...] : p1=1er, dealer=0=2eme
    env = new_env(); env.game_scores = [5.0, 15.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("East 4, dealer 2eme repeat (continue)",
          env, 'exhaustive', 0.0, False, None,
          dealer_repeats=True, dealer_real=0)


# ====================================================================== #
# 5. ENCHOUSEN_30K
# ====================================================================== #
def test_enchousen_30k():
    section("5. ENCHOUSEN_30K (Sud 1-3, agari porte qqun a 30k+)")

    env = new_env(); env.game_scores = [5.0, 8.0, 2.0, -2.0]
    env.round_num = 6; env.is_enchousen = True
    check("Sud 2, agari, qqun >= 30k", env, 'tsumo', 1.0,
          True, "enchousen_30k")

    env = new_env(); env.game_scores = [4.9, 4.5, 2.0, -2.0]  # max 29.9k
    env.round_num = 6; env.is_enchousen = True
    check("Sud 2, ryuukyoku SANS passage de 30k (continue)",
          env, 'exhaustive', 0.0, False, None)

    # [FIX 2026-11] paiement de noten qui fait franchir 30k en extension :
    # regle MJSoul "加时赛中一旦有玩家超过30000点即立刻结束游戏" -- le
    # franchissement compte QUELLE QUE SOIT la facon de marquer (agari
    # OU paiements de noten sur ryuukyoku). L'ancien code ne verifiait
    # que les agari.
    env = new_env(); env.game_scores = [5.2, 8.0, 2.0, -2.0]
    env.round_num = 6; env.is_enchousen = True
    check("Sud 2, ryuukyoku, noten fait franchir 30k (fin)",
          env, 'exhaustive', 0.0, True, "enchousen_30k")

    env = new_env(); env.game_scores = [3.0, 2.0, 0.0, -5.0]
    env.round_num = 6; env.is_enchousen = True
    check("Sud 2, agari, personne >= 30k (continue)",
          env, 'tsumo', 1.0, False, None)

    env = new_env(); env.game_scores = [2.0, 6.0, 3.0, -1.0]
    env.round_num = 7; env.is_enchousen = True
    check("Sud 3, agari, qqun >= 30k", env, 'ron', 2.0,
          True, "enchousen_30k")

    env = new_env(); env.game_scores = [5.5, 8.0, 0.0, -5.0]
    env.round_num = 5; env.is_enchousen = True
    check("Sud 1, agari, qqun >= 30k", env, 'tsumo', 1.0,
          True, "enchousen_30k")


# ====================================================================== #
# 6. ENCHOUSEN_MAX
# ====================================================================== #
def test_enchousen_max():
    section("6. ENCHOUSEN_MAX (Sud 4 sans repeat)")

    # Sud 4, !repeat, dealer 2eme -> enchousen_max
    # Scores [5, 3, 2, 0] : p0=1er... attends, dealer=0 est 1er !
    # Pour eviter agari yame, il faut outcome=exhaustive et pas tenpai
    # [FIX 2026-11] scores [5, ...] = 30k pile -> enchousen_30k legitime
    # avec le fix regle 4 (le check 30k s'applique aussi au ryuukyoku).
    # Pour tester enchousen_max SANS franchissement : max < 30k.
    env = new_env(); env.game_scores = [4.0, 3.0, 2.0, 0.0]  # max 29k
    env.round_num = 8; env.is_enchousen = True
    check("Sud 4, no repeat, exhaustive",
          env, 'exhaustive', 0.0, True, "enchousen_max",
          dealer_repeats=False, tenpai=False)

    # Sud 4, dealer 1er repeat + gagne -> agari yame (prime sur enchousen_max)
    env = new_env(); env.game_scores = [5.0, 3.0, 2.0, 0.0]
    env.round_num = 8; env.is_enchousen = True
    check("Sud 4, dealer 1er repeat + gagne (agari yame)",
          env, 'tsumo', 1.5, True, "agari_yame",
          dealer_repeats=True, dealer_real=0)

    # Sud 4, dealer 2eme repeat -> continue
    # [FIX 2026-11] memes scores < 30k (p1 = 29k, 1er ; dealer p0 2eme)
    env = new_env(); env.game_scores = [3.0, 4.0, 2.0, 0.0]  # p1=1er
    env.round_num = 8; env.is_enchousen = True
    check("Sud 4, dealer 2eme repeat (continue)",
          env, 'exhaustive', 0.0, False, None,
          dealer_repeats=True, dealer_real=0)


# ====================================================================== #
# 7. MAX_KYOKU
# ====================================================================== #
def test_max_kyoku():
    section("7. MAX_KYOKU (securite anti-boucle)")

    # [FIX 2026-11] scores < 30k pour atteindre la limite sans declencher
    # enchousen_30k avant max_kyoku.
    env = new_env(); env.game_scores = [4.0, 3.0, 2.0, 0.0]  # max 29k
    env.round_num = 5; env.is_enchousen = True
    env.kyoku_count = MAX_KYOKU - 1
    check("kyoku_count atteint la limite",
          env, 'exhaustive', 0.0, True, "max_kyoku")


# ====================================================================== #
# 8. PRIORITES
# ====================================================================== #
def test_priorites():
    section("8. PRIORITES (ordre des checks)")

    env = new_env(); env.game_scores = [10.0, 5.0, 5.0, -30.0]
    env.round_num = 4; env.is_enchousen = False
    check("Tobi prioritaire sur agari yame",
          env, 'tsumo', 1.5, True, "tobi")

    env = new_env(); env.game_scores = [10.0, 5.0, 0.0, -5.0]
    env.round_num = 4; env.is_enchousen = False
    check("Agari yame prioritaire sur oorasu",
          env, 'tsumo', 1.5, True, "agari_yame")


# ====================================================================== #
# 9. RENCHAN A L'EXHAUSTIVE (regression fix 2026-11)
# ====================================================================== #
def test_renchan_exhaustive():
    section("9. RENCHAN EXHAUSTIVE (tenpai REEL du dealer, pas le paiement)")

    # [FIX RENCHAN EXHAUSTIVE 2026-11] a l'exhaustive sans aucun tenpai,
    # le paiement vaut 0 : l'ancien proxy 'paiement >= 0' donnait renchan
    # a tort (le dealer est noten !). Ce bug etait masque par l'artefact
    # -0.1*shanten (paiement alors < 0) ; son retrait l'a expose :
    # renchans infinis jusqu'au cap MAX_KYOKU en jeu aleatoire.
    def check9(name, outcome, reward, want, tenpai=None, use_tenpai=True):
        if use_tenpai:
            got = _decide_dealer_repeat(outcome, reward, tenpai)
        else:
            got = _decide_dealer_repeat(outcome, reward)
        ok = (got == want)
        _results["pass" if ok else "fail"] += 1
        print(f"  [{'OK' if ok else 'XX'}] {name:55s}"
              f" -> {got} (attendu {want})")

    check9("exhaustive, 0 tenpai (paiement 0, dealer noten) : PAS renchan",
           "exhaustive", 0.0, (False, False), tenpai=False)
    check9("exhaustive, 4 tenpai (paiement 0, dealer tenpai) : renchan",
           "exhaustive", 0.0, (True, True), tenpai=True)
    check9("exhaustive, dealer tenpai (+3.0) : renchan + honba",
           "exhaustive", 3.0, (True, True), tenpai=True)
    check9("exhaustive, dealer noten (-1.0) : PAS renchan",
           "exhaustive", -1.0, (False, False), tenpai=False)
    check9("exhaustive, dealer tenpai en riichi (net 0) : renchan",
           "exhaustive", 0.0, (True, True), tenpai=True)
    check9("abortive : renchan SANS honba (inchange)",
           "abortive_suucha", 0.0, (True, False), use_tenpai=False)
    check9("tsumo gagnant : renchan", "tsumo", 2.0, (True, True),
           use_tenpai=False)
    check9("tsumo perdant : PAS renchan", "tsumo", -2.0, (False, False),
           use_tenpai=False)
    check9("exhaustive sans info tenpai (repli proxy, compat) : renchan",
           "exhaustive", 0.0, (True, True), use_tenpai=False)


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    print(f"{'=' * 75}")
    print(f"  TESTS DES REGLES DE FIN DE PARTIE (Mahjong Soul)")
    print(f"{'=' * 75}")
    print(f"\n  Constantes :")
    print(f"    N_ROUNDS_NORMAL  = {N_ROUNDS_NORMAL}")
    print(f"    N_ROUNDS_MAX     = {N_ROUNDS_MAX}")
    print(f"    TARGET_SCORE     = {TARGET_SCORE}")
    print(f"    TOBI_THRESHOLD   = {TOBI_THRESHOLD}")
    print(f"    MAX_KYOKU        = {MAX_KYOKU}")

    test_tobi()
    test_agari_yame()
    test_tenpai_yame()
    test_oorasu()
    test_enchousen_30k()
    test_enchousen_max()
    test_max_kyoku()
    test_priorites()
    test_renchan_exhaustive()

    print(f"\n{'=' * 75}")
    print(f"  RESULTAT : {_results['pass']} pass / {_results['fail']} fail")
    print(f"{'=' * 75}")

    return 0 if _results['fail'] == 0 else 1


if __name__ == '__main__':
    exit(main())