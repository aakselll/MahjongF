"""
test_riichi_sticks.py — Comptabilite des batons de riichi (vraie lib mahjong).

Regression du bug : le cout du baton etait range dans `pending`, ecrase au
discard suivant -> un riichi etait gratuit et le gagnant empochait des batons
que personne n'avait payes. Verifie aussi le report des batons apres une
manche sans gagnant et le renchan du dealer en riichi.

Usage :
    py -3.14 -m mahjong_ai.test_riichi_sticks
"""
import numpy as np

from mahjong_ai.envs.multi_agent import (
    MultiAgentMahjongEnv, DISCARD, RIICHI, RIICHI_YES,
)
from mahjong_ai.training.game_env import GameEnv

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}"
          + ("" if cond else f"   {detail}"))


# Dealer tenpai (123456789m 111p + 9s, attente tanki), mains junk ailleurs.
TENPAI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 26]
JUNK = [0, 4, 8, 9, 13, 17, 18, 22, 26, 27, 29, 31, 33]


def make_env(hands, wall_full, seed=11):
    env = MultiAgentMahjongEnv(seed=seed, compute_lookahead=False)
    original_deal = env._deal

    def custom_deal():
        original_deal()
        for s in range(4):
            h = np.zeros(34, dtype=np.int32)
            for t in hands[s]:
                h[t] += 1
            env.hands[s] = h
        env.wall = list(wall_full)
        env._initial_wall_size = len(env.wall)

    env._deal = custom_deal
    return env


def lowest_legal(req):
    return int(np.nonzero(req["mask"] > 0)[0][0])


def play_round(step_fn, req, env_riichi_flags):
    """Joue jusqu'a la fin de manche. Dealer : RIICHI_YES puis defausse 26."""
    while True:
        t = req["action_type"]
        if t == DISCARD:
            m = req["mask"]
            if req["seat"] == 0 and m[26] > 0 and not env_riichi_flags()[0]:
                act = 26
            else:
                act = lowest_legal(req)
        elif t == RIICHI:
            act = RIICHI_YES
        else:
            act = 0                      # pas de pon/chi/kan/ron
        res = step_fn(act)
        if res["round_done"]:
            return res
        req = res["next_request"]


def test_riichi_survives_to_exhaustive():
    print("\n=== B1 : dealer en riichi -> exhaustive : le baton est preleve ===")
    # [RETRAIT SHAPING 2026-11] plus besoin de neutraliser
    # EXHAUSTIVE_SHANTEN_COEF : les paiements de noten sont exacts par defaut.
    # pioches (de la fin) : dealer 33, puis honneurs ; jamais 33 ensuite
    wall = [28, 28, 28, 28, 27, 29, 30, 33]
    env = make_env([TENPAI, JUNK, JUNK, JUNK], wall)
    ge = GameEnv(seed=1)
    ge.round_env = env
    req = ge.reset_round()
    res = play_round(lambda a: ge.step(a), req, lambda: env.riichi)
    fr = res["final_rewards"]
    check("manche = exhaustive", res["outcome"] == "exhaustive", res["outcome"])
    check("dealer : +3.0 (tenpai) -1.0 (baton) = +2.0",
          abs(fr[0] - 2.0) < 1e-9, fr[0])
    check("les 3 noten paient 1.0",
          all(abs(fr[s] + 1.0) < 1e-9 for s in (1, 2, 3)), fr)
    check("baton reporte (GameEnv.kyoutaku == 1)", ge.kyoutaku == 1, ge.kyoutaku)
    check("conservation : somme des scores + batons == 0",
          abs(sum(ge.game_scores) + ge.kyoutaku) < 1e-9,
          (ge.game_scores, ge.kyoutaku))
    check("dealer tenpai en riichi garde le renchan",
          ge.honba == 1 and ge.dealer_seat == 0, (ge.honba, ge.dealer_seat))

    print("\n=== B2 : manche suivante : le gagnant empoche le baton reporte ===")
    s1 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]       # tanki 8m (cf. T8)
    env2 = make_env([JUNK, s1, JUNK, JUNK], [7, 27])
    ge.round_env = env2
    req = ge.reset_round()
    check("la manche demarre avec 1 baton sur la table",
          env2.riichi_deposits == 1, env2.riichi_deposits)
    res = ge.step(33)                                   # le dealer defausse
    fr = res["final_rewards"]
    check("tsumo du siege 1", res["outcome"] == "tsumo", res["outcome"])
    check("batons empoches (kyoutaku remis a 0)", ge.kyoutaku == 0, ge.kyoutaku)
    # Si ceci echoue alors que B1 passe : cost['total'] de la lib n'inclut pas
    # kyoutaku_bonus comme _distribute_win_payment le suppose.
    check("conservation apres 2 manches : somme des scores == 0",
          abs(sum(ge.game_scores)) < 1e-9, (ge.game_scores, fr))


def test_stick_not_overwritten():
    print("\n=== B3 : le baton n'est plus ecrase par les coups suivants ===")
    env = make_env([TENPAI, JUNK, JUNK, JUNK], [28, 28, 28, 28, 27, 29, 30, 33])
    req = env.reset_round()
    res = env.step(RIICHI_YES)
    check("stick_paid enregistre a la declaration", env.stick_paid[0] == 1.0)
    res = env.step(26)
    for _ in range(6):
        if res["round_done"]:
            break
        r = res["next_request"]
        res = env.step(lowest_legal(r) if r["action_type"] == DISCARD else 0)
    check("stick_paid toujours la apres plusieurs coups", env.stick_paid[0] == 1.0)


def test_abortive_no_refund():
    print("\n=== B4 : abortive 4 riichi : 4 batons payes, aucun remboursement ===")
    env = make_env([JUNK] * 4, [1, 2, 3, 4])
    env._deal()
    env.stick_paid = [1.0, 1.0, 1.0, 1.0]
    out = env._final(0.0, None, env._abortive_draw_rewards(), "abortive_4riichi")
    check("chacun paie 1.0",
          all(abs(out["final_rewards"][s] + 1.0) < 1e-9 for s in range(4)),
          out["final_rewards"])


if __name__ == '__main__':
    test_riichi_survives_to_exhaustive()
    test_stick_not_overwritten()
    test_abortive_no_refund()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} OK")
    raise SystemExit(1 if n_fail else 0)
