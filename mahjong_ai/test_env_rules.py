"""
test_env_rules.py — Tests des regles de MultiAgentMahjongEnv.

Couvre : flux de base, pon/chi + kuikae, kan (timing apres appel,
rinshan, double ankan, daiminkan), riichi (offre, discard force,
tsumogiri), tsumo, ron (accepte/refuse), furiten (permanent/temporaire),
chankan, triple ron, abortives, exhaustive + noten, _should_declare_ron,
helpers.
"""
import numpy as np

from mahjong_ai.envs.multi_agent import (
    MultiAgentMahjongEnv,
    DISCARD, PONG, CHOW, KONG, RIICHI, RON,
    CHOW_PASS, CHOW_LOWER, CHOW_MID, CHOW_UPPER,
    KONG_NO, KONG_YES, RIICHI_NO, RIICHI_YES, RON_NO, RON_YES,
)

RESULTS = []

def check(name, cond, detail=""):
    if cond:
        RESULTS.append((name, True, ""))
        print(f"  [OK  ] {name}")
    else:
        RESULTS.append((name, False, str(detail)))
        print(f"  [FAIL] {name}   {detail}")

# ------------------------------------------------------------------ #
# Helpers d'injection
# ------------------------------------------------------------------ #
# Main junk : 13 tuiles dispersees, jamais tenpai, jamais de doublon.
JUNK = [0, 4, 8, 9, 13, 17, 18, 22, 26, 27, 29, 31, 33]

def make_env(hands, melds=None, wall_tail=None, wall_full=None, seed=11):
    """Cree un env avec mains/mur injectes (meme technique que pmcpa.py).
    wall_tail : tuiles en FIN de mur (piochees en premier : wall.pop()).
    wall_full : remplace tout le mur (pour tester l'exhaustive vite).
    """
    env = MultiAgentMahjongEnv(seed=seed, compute_lookahead=False)
    original_deal = env._deal

    def custom_deal():
        original_deal()
        for s in range(4):
            h = np.zeros(34, dtype=np.int32)
            for t in hands[s]:
                h[t] += 1
            env.hands[s] = h
        env._clear_reds()      # [ROUGES] mains injectees : pas de rouge
        if melds:
            for s in range(4):
                env.melds[s] = list(melds[s])
            env._any_call = True
        if wall_full is not None:
            env.wall = list(wall_full)
            env._initial_wall_size = len(env.wall)
        elif wall_tail is not None:
            filler = [t for t in range(34)] * 2
            env.wall = filler + list(wall_tail)
            env._initial_wall_size = len(env.wall)

    env._deal = custom_deal
    return env

def first_legal(req, avoid=()):
    mask = req["mask"]
    for t in range(len(mask)):
        if mask[t] > 0 and t not in avoid:
            return t
    for t in range(len(mask)):
        if mask[t] > 0:
            return t
    return 0

# ------------------------------------------------------------------ #
def test_basic():
    print("\n=== T1 : flux de base ===")
    env = make_env([JUNK for _ in range(4)], wall_tail=[33])
    req = env.reset_round()
    check("premiere request : DISCARD siege 0 (dealer)",
          req["action_type"] == DISCARD and req["seat"] == 0)
    expected = (env.hands[0] > 0).astype(np.float32)
    check("masque = main>0 (14 tuiles apres pioche)",
          np.allclose(req["mask"], expected) and env.hands[0].sum() == 14)
    res = env.step(0)
    check("personne ne revendique -> siege 1 pioche et defausse",
          (not res["round_done"])
          and res["next_request"]["action_type"] == DISCARD
          and res["next_request"]["seat"] == 1)

def test_pong_kuikae():
    print("\n=== T2 : pon + kuikae (3 copies -> 1 reste en main) ===")
    s1 = [0, 0, 0] + JUNK[3:]
    env = make_env([JUNK, s1, JUNK, JUNK], wall_tail=[33])
    env.reset_round()
    res = env.step(0)                      # siege 0 defausse 1m
    r = res["next_request"]
    check("daiminkan propose en premier (3 x 1m en main)",
          r["action_type"] == KONG and r["seat"] == 1)
    res = env.step(KONG_NO)
    r = res["next_request"]
    check("pon propose apres refus du daiminkan",
          r["action_type"] == PONG and r["seat"] == 1)
    res = env.step(1)                      # PONG_YES
    r = res["next_request"]
    check("apres pon : DISCARD direct (ni pioche ni KONG)",
          r["action_type"] == DISCARD and r["seat"] == 1)
    check("kuikae : 1m interdit alors qu'il reste 1 copie en main",
          r["mask"][0] == 0.0 and env.hands[1][0] == 1)

def test_chi_kuikae():
    print("\n=== T3 : chi + kuikae (copie de la reclamee en main) ===")
    # [FIX] main de 13 : 2m3m4m5m + JUNK[4:] (9 elements)
    s1 = [1, 2, 3, 4] + JUNK[4:]
    env = make_env([JUNK, s1, JUNK, JUNK], wall_tail=[33])
    env.reset_round()
    res = env.step(4)                      # siege 0 defausse 5m
    r = res["next_request"]
    check("chi propose au shimocha, seule variante LOWER",
          r["action_type"] == CHOW and r["seat"] == 1
          and r["mask"][CHOW_LOWER] == 1.0
          and r["mask"][CHOW_MID] == 0.0
          and r["mask"][CHOW_UPPER] == 0.0)
    res = env.step(CHOW_LOWER)
    r = res["next_request"]
    check("apres chi : DISCARD direct",
          r["action_type"] == DISCARD and r["seat"] == 1)
    check("kuikae : la 5m propre (copie en main) interdite",
          r["mask"][4] == 0.0 and env.hands[1][4] == 1)
    check("kuikae (regle T-3 du env) : la 2m aussi interdite",
          r["mask"][1] == 0.0 and env.hands[1][1] == 1)

def test_kan_after_chi():
    print("\n=== T4 : kan interdit juste apres chi, autorise apres pioche ===")
    JK = [0, 8, 9, 13, 18, 22]             # junk sans 4m ni 9p
    s1 = [2, 3, 4, 17, 17, 17, 17] + JK    # 3m4m5m + 9p x4 + junk
    env = make_env([JUNK, s1, JUNK, JUNK], wall_tail=[33])
    env.reset_round()
    res = env.step(4)                      # siege 0 defausse 5m
    check("chi propose", res["next_request"]["action_type"] == CHOW)
    res = env.step(CHOW_LOWER)
    r = res["next_request"]
    check("PAS de KONG juste apres le chi (9p x4 en main !)",
          r["action_type"] == DISCARD and r["seat"] == 1)
    res = env.step(first_legal(r, avoid={17}))
    for _ in range(3):                     # sieges 2, 3, 0
        r = res["next_request"]
        check(f"tour intermediaire : DISCARD siege {r['seat']}",
              r["action_type"] == DISCARD)
        res = env.step(first_legal(r))
    r = res["next_request"]
    check("KONG (ankan 9p) propose apres la pioche suivante",
          r["action_type"] == KONG and r["seat"] == 1)
    res = env.step(KONG_YES)
    r = res["next_request"]
    check("apres ankan : rinshan piochee puis DISCARD",
          r["action_type"] == DISCARD and r["seat"] == 1)
    check("meld ankan enregistre + nouvelle dora revelee",
          any(m["type"] == "kan_closed" and m["tiles"][0] == 17
              for m in env.melds[1])
          and len(env.dora_indicators) == 2)

def test_double_ankan():
    print("\n=== T5 : double ankan via rinshan ===")
    s0 = [17, 17, 17, 17, 26, 26, 26, 26] + JUNK[:5]
    env = make_env([s0, JUNK, JUNK, JUNK], wall_tail=[33])
    req = env.reset_round()
    check("1er KONG (ankan 9p) propose au dealer",
          req["action_type"] == KONG and req["seat"] == 0)
    res = env.step(KONG_YES)
    r = res["next_request"]
    check("2e KONG (ankan 9s) propose apres la rinshan",
          r["action_type"] == KONG and r["seat"] == 0)
    res = env.step(KONG_YES)
    r = res["next_request"]
    check("apres le 2e ankan : DISCARD",
          r["action_type"] == DISCARD and r["seat"] == 0)
    check("2 kan_closed + 2 doras de kan revelees",
          sum(1 for m in env.melds[0] if m["type"] == "kan_closed") == 2
          and len(env.dora_indicators) == 3)

def test_daiminkan():
    print("\n=== T6 : daiminkan accepte ===")
    s1 = [0, 0, 0] + JUNK[3:]
    env = make_env([JUNK, s1, JUNK, JUNK], wall_tail=[33])
    env.reset_round()
    res = env.step(0)
    check("KONG (daiminkan sur le discard) propose",
          res["next_request"]["action_type"] == KONG
          and res["next_request"]["seat"] == 1)
    res = env.step(KONG_YES)
    r = res["next_request"]
    check("rinshan piochee puis DISCARD",
          r["action_type"] == DISCARD and r["seat"] == 1
          and env.hands[1].sum() == 11)
    check("meld kan_open + dora revelee",
          any(m["type"] == "kan_open" and m["tiles"][0] == 0
              for m in env.melds[1])
          and len(env.dora_indicators) == 2)

def test_riichi():
    print("\n=== T7 : riichi, discard force, tsumogiri auto ===")
    s0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 26]  # tanki 9s
    env = make_env([s0, JUNK, JUNK, JUNK], wall_tail=[33])
    req = env.reset_round()
    check("RIICHI propose au dealer tenpai",
          req["action_type"] == RIICHI and req["seat"] == 0)
    res = env.step(RIICHI_YES)
    r = res["next_request"]
    forced = [t for t in range(34) if r["mask"][t] > 0]
    check("discard force = tuiles preservant le tenpai (9s ou la pioche)",
          r["action_type"] == DISCARD and set(forced) == {26, 33})
    res = env.step(26)
    check("riichi enregistre + depot de 1000",
          env.riichi[0] is True and env.riichi_deposits == 1)
    seats_seen = []
    for _ in range(4):
        r = res["next_request"]
        seats_seen.append(r["seat"])
        res = env.step(first_legal(r))
    check("le siege 0 en riichi tsumogire SANS request ([1,2,3,1])",
          seats_seen == [1, 2, 3, 1])

def test_tsumo():
    print("\n=== T8 : tsumo (siege non-dealer) ===")
    s1 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]  # tanki 8m
    env = make_env([JUNK, s1, JUNK, JUNK], wall_tail=[7, 27])
    env.reset_round()
    res = env.step(33)                     # s0 defausse ; s1 pioche 8m
    check("tsumo detecte : manche terminee",
          res["round_done"] and res["outcome"] == "tsumo")
    fr = res["final_rewards"]
    check("paiements : gagnant +, les 3 autres -",
          fr[1] > 0 and all(fr[s] < 0 for s in (0, 2, 3)))

def test_ron_accept():
    print("\n=== T9 : ron accepte (yakuhai) ===")
    s0 = [32] + JUNK[1:]
    s1 = [32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]  # FF + 3 runs + 11p
    env = make_env([s0, s1, JUNK, JUNK], wall_tail=[27])
    env.reset_round()
    res = env.step(32)                     # s0 defausse F
    r = res["next_request"]
    check("RON propose au siege 1",
          r["action_type"] == RON and r["seat"] == 1)
    res = env.step(RON_YES)
    check("ron accepte : manche terminee",
          res["round_done"] and res["outcome"] == "ron")
    fr = res["final_rewards"]
    check("paiements : gagnant +, discarder -",
          fr[1] > 0 and fr[0] < 0)

def test_ron_refuse_furiten():
    print("\n=== T10 : ron refuse -> furiten temp -> efface a la pioche ===")
    s0 = [32] + JUNK[1:]
    s1 = [32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    env = make_env([s0, s1, JUNK, JUNK], wall_tail=[27])
    env.reset_round()
    res = env.step(32)
    res = env.step(RON_NO)                 # s1 REFUSE le ron
    r = res["next_request"]
    check("furiten temporaire actif apres refus",
          env._furiten_temp[1] is True)
    check("PON propose ensuite (furiten ne bloque que le ron)",
          r["action_type"] == PONG and r["seat"] == 1)
    res = env.step(0)                      # refuse le pon
    r = res["next_request"]
    check("furiten temporaire EFFACE apres la pioche de s1",
          env._furiten_temp[1] is False)
    check("s1 tenpai -> RIICHI propose apres sa pioche",
          r["action_type"] == RIICHI and r["seat"] == 1)

def test_furiten_permanent():
    print("\n=== T11 : furiten permanent bloque le ron ===")
    s0 = [32] + JUNK[1:]
    s1 = [32, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    env = make_env([s0, s1, JUNK, JUNK], wall_tail=[27])
    inner = env._deal
    def deal_with_discard():
        inner()
        env.discards[1][32] = 1            # s1 a defausse un F plus tot
    env._deal = deal_with_discard
    env.reset_round()
    res = env.step(32)                     # s0 defausse F
    r = res["next_request"]
    check("_is_furiten(1) == True", env._is_furiten(1) is True)
    check("PAS de RON propose -> PON propose directement",
          r["action_type"] == PONG and r["seat"] == 1)

def test_chankan():
    print("\n=== T12 : chankan (ron sur shouminkan) — TRIPWIRE ===")
    JK9 = [4, 8, 9, 13, 18, 22, 26, 27, 29]
    s1_hand = [0] + JK9                    # 4e 1m + 9 junk (main de 10)
    s2 = [1, 2, 9, 10, 11, 12, 13, 14, 32, 32, 32, 27, 27]
    # s2 : 23m + 123p + 456p + FFF + EE -> attend 1m/4m, yaku FFF
    env = make_env(
        [JUNK, s1_hand, s2, JUNK],
        melds=[[], [{"type": "pon", "tiles": [0, 0, 0]}], [], []],
        wall_tail=[33],
    )
    env.reset_round()
    res = env.step(33)                     # s0 defausse le rouge
    r = res["next_request"]
    check("KONG (shouminkan 1m) propose a s1 apres sa pioche",
          r["action_type"] == KONG and r["seat"] == 1)
    res = env.step(KONG_YES)               # s1 declare le kakan
    r = res["next_request"]
    check("CHANKAN : RON propose a s2 (yakuhai, pas besoin de riichi)",
          r["action_type"] == RON and r["seat"] == 2)
    res = env.step(RON_YES)
    check("chankan RESOLU : manche terminee en ron",
          res["round_done"] and res["outcome"] == "ron")
    fr = res["final_rewards"]
    check("paiements : gagnant +, kan-declareur -",
          fr[2] > 0 and fr[1] < 0)

def test_triple_ron():
    print("\n=== T13 : triple ron -> abortive ===")
    s1 = [32, 32, 32, 1, 2, 12, 13, 14, 15, 16, 17, 26, 26]
    s2 = [33, 33, 33, 1, 2, 18, 19, 20, 21, 22, 23, 15, 15]
    s3 = [31, 31, 31, 1, 2, 9, 10, 11, 24, 25, 26, 4, 4]
    env = make_env([JUNK, s1, s2, s3], wall_tail=[33])
    env.reset_round()
    res = env.step(0)                      # s0 defausse 1m
    check("triple ron -> abortive_3ron immediat",
          res["round_done"] and res["outcome"] == "abortive_3ron")

def test_abortive_unit():
    print("\n=== T14 : compteurs abortifs (unite) ===")
    env = MultiAgentMahjongEnv(seed=1, compute_lookahead=False)
    env._abortive_riichi_count = 4
    check("4 riichi -> abortive_4riichi",
          env._check_abortive_draw() == "abortive_4riichi")
    env._abortive_riichi_count = 0
    env._abortive_kan_count = 4
    check("4 kan -> abortive_4kan",
          env._check_abortive_draw() == "abortive_4kan")
    env._abortive_kan_count = 0
    check("rien -> None", env._check_abortive_draw() is None)

def test_exhaustive():
    print("\n=== T15 : exhaustive draw + paiements noten ===")
    s1 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]  # tenpai tanki 8m
    env = make_env([JUNK, s1, JUNK, JUNK], wall_full=[33] * 12)
    req = env.reset_round()
    steps, res = 0, None
    while steps < 100:
        if req["action_type"] == DISCARD:
            if req["seat"] == 1 and req["mask"][33] > 0:
                action = 33               # s1 tsumogiri : preserve le tenpai
            else:
                action = first_legal(req)
        else:
            action = 0                    # refuser riichi/kan/pong/chow/ron
        res = env.step(action)
        steps += 1
        if res["round_done"]:
            break
        req = res["next_request"]
    check("exhaustive draw atteinte",
          res is not None and res["round_done"]
          and res["outcome"] == "exhaustive")
    fr = res["final_rewards"]
    check("noten : le tenpai +, les noten -",
          fr[1] > 0 and all(fr[s] < 0 for s in (0, 2, 3)))

def test_should_declare_ron():
    print("\n=== T16 : _should_declare_ron (unite) ===")
    class FakeWin:
        def __init__(self, total):
            self.cost = {"total": total}
    env = MultiAgentMahjongEnv(seed=1, compute_lookahead=False)
    def run(round_idx, scores, seat, total):
        env.set_game_context(0, round_idx, scores)
        return env._should_declare_ron(seat, FakeWin(total))
    check("E1 : toujours ron", run(0, [10, 5, 0, -5], 3, 1000) is True)
    check("E3 : toujours ron", run(2, [10, 5, 0, -5], 3, 1000) is True)
    check("E4, 4e, petit gain reste dernier -> REFUS (heuristique doc.)",
          run(3, [10, 5, 0, -5], 3, 1000) is False)
    check("E4, 4e, gros gain quitte la derniere place -> ron",
          run(3, [10, 5, 0, -5], 3, 8000) is True)
    check("E4, 1er -> ron", run(3, [10, 5, 0, -5], 0, 1000) is True)
    check("enchousen (round_idx 4) : logique derniere manche active",
          run(4, [10, 5, 0, -5], 3, 1000) is False)

def test_helpers():
    print("\n=== T17 : helpers (unite) ===")
    env = MultiAgentMahjongEnv(seed=1, compute_lookahead=False)
    env.hands = [np.zeros(34, dtype=np.int32) for _ in range(4)]
    env.melds = [[] for _ in range(4)]
    env.hands[0][17] = 4
    check("_find_closed_kong : 4 x 9p", env._find_closed_kong(0) == 17)
    env.hands[0][17] = 3
    check("_find_closed_kong : 3 -> None", env._find_closed_kong(0) is None)
    env.hands[1][5] = 1
    env.melds[1] = [{"type": "pon", "tiles": [5, 5, 5]}]
    check("_find_shouminkan : pon + 1 en main", env._find_shouminkan(1) == 5)
    env.hands[1][5] = 0
    check("_find_shouminkan : pon sans copie -> None",
          env._find_shouminkan(1) is None)
    env.hands = [np.zeros(34, dtype=np.int32) for _ in range(4)]
    env.hands[0][1] = 1; env.hands[0][2] = 1     # 2m 3m
    m = env._chow_variants(0, 3)                 # chi sur 4m
    check("_chow_variants : 23m -> LOWER seul",
          m is not None and m[CHOW_LOWER] == 1.0
          and m[CHOW_MID] == 0.0 and m[CHOW_UPPER] == 0.0)
    env.hands[0][1] = 0; env.hands[0][4] = 1     # 3m 5m
    m = env._chow_variants(0, 3)
    check("_chow_variants : 35m -> MID seul",
          m is not None and m[CHOW_MID] == 1.0 and m[CHOW_LOWER] == 0.0)
    check("_chow_variants : honneur -> None", env._chow_variants(0, 27) is None)

def test_furiten_unit():
    print("\n=== T18 : _is_furiten (unite) ===")
    env = MultiAgentMahjongEnv(seed=1, compute_lookahead=False)
    env.hands = [np.zeros(34, dtype=np.int32) for _ in range(4)]
    env.melds = [[] for _ in range(4)]
    env.discards = [np.zeros(34, dtype=np.int32) for _ in range(4)]
    env._furiten_temp = [False] * 4
    for t in [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]:
        env.hands[1][t] += 1
    check("tenpai, attente pas dans defausses -> pas furiten",
          env._is_furiten(1) is False)
    env.discards[1][7] = 1
    check("tenpai, attente (8m) dans ses defausses -> furiten",
          env._is_furiten(1) is True)
    env._furiten_temp[1] = True
    check("flag temporaire -> True", env._is_furiten(1) is True)
    env._furiten_temp[1] = False
    env.hands[1][7] += 1
    check("non-tenpai (14 tuiles) -> pas furiten",
          env._is_furiten(1) is False)

def main():
    print("=" * 64)
    print("TESTS MultiAgentMahjongEnv")
    print("=" * 64)
    test_basic(); test_pong_kuikae(); test_chi_kuikae()
    test_kan_after_chi(); test_double_ankan(); test_daiminkan()
    test_riichi(); test_tsumo(); test_ron_accept()
    test_ron_refuse_furiten(); test_furiten_permanent(); test_chankan()
    test_triple_ron(); test_abortive_unit(); test_exhaustive()
    test_should_declare_ron(); test_helpers(); test_furiten_unit()
    print("\n" + "=" * 64)
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_ok
    print(f"RESULTAT : {n_ok} OK, {n_fail} FAIL / {len(RESULTS)}")
    if n_fail:
        print("\nEchecs :")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}   {detail}")
    print("=" * 64)

if __name__ == "__main__":
    main()