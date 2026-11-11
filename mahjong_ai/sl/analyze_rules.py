r"""Inspect MJAI games and infer the rules visible in the dataset.

This tool is deliberately read-only. It does not decide which rules *should*
apply; it reports evidence found in a sample of compressed MJAI files.

Example:
    py -3.14 -m mahjong_ai.sl.analyze_rules D:\data --files 20
"""
import argparse
import json
import os
import random
from collections import Counter

from mahjong_ai.sl.mjai_parser import parse_mjai_zst_file


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scores(value):
    if not isinstance(value, list) or len(value) != 4:
        return None
    converted = [_number(v) for v in value]
    return converted if all(v is not None for v in converted) else None


def _delta(value):
    scores = _scores(value)
    if scores is not None:
        return scores
    if isinstance(value, dict):
        result = [0.0] * 4
        for key, raw in value.items():
            try:
                seat = int(key)
            except (TypeError, ValueError):
                continue
            amount = _number(raw)
            if 0 <= seat < 4 and amount is not None:
                result[seat] = amount
        return result
    return None


def _rank(scores, seat):
    if scores is None or not 0 <= seat < 4:
        return None
    ordered = sorted(range(4), key=lambda p: (-scores[p], p))
    return ordered.index(seat) + 1


def _find_scores(event):
    for key in ("scores", "score", "tehais_scores"):
        result = _scores(event.get(key))
        if result is not None:
            return result, key
    return None, None


def _terminal_event(events, start_index, next_start):
    """Return the result event carrying deltas for one kyoku."""
    results = []
    for event in events[start_index + 1:next_start]:
        event_type = event.get("type")
        if event_type in ("hora", "ryukyoku"):
            results.append(event)
    return results[-1] if results else {}


def _has_end_game(events, start_index, next_start):
    return any(
        event.get("type") == "end_game"
        for event in events[start_index + 1:next_start]
    )


def _extract_tenpai(event):
    """Return a 4-seat tenpai vector when the MJAI event exposes one."""
    for key in ("tenpai", "tenpais", "tenpai_flags"):
        value = event.get(key)
        if isinstance(value, list) and len(value) == 4:
            result = []
            valid = True
            for item in value:
                if isinstance(item, bool):
                    result.append(item)
                elif isinstance(item, (int, float)) and item in (0, 1):
                    result.append(bool(item))
                else:
                    valid = False
                    break
            if valid:
                return result, key
    return None, None


def _analyse_game(path, events):
    starts = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("type") == "start_kyoku"
    ]
    rounds = []
    for position, (index, start) in enumerate(starts):
        next_start = starts[position + 1][0] if position + 1 < len(starts) else len(events)
        terminal = _terminal_event(events, index, next_start)
        next_round = starts[position + 1][1] if position + 1 < len(starts) else None
        end_game_after = _has_end_game(events, index, next_start)

        start_scores, start_score_key = _find_scores(start)
        end_scores, end_score_key = _find_scores(terminal)
        deltas = _delta(terminal.get("deltas"))
        if end_scores is None and next_round is not None:
            end_scores, end_score_key = _find_scores(next_round)
        if end_scores is None and start_scores is not None and deltas is not None:
            end_scores = [a + b for a, b in zip(start_scores, deltas)]
            end_score_key = "start_scores+deltas"

        winners = []
        if terminal.get("type") == "hora":
            actor = terminal.get("actor")
            if actor is not None:
                winners.append(actor)
        for event in events[index + 1:next_start]:
            if event.get("type") == "hora" and event.get("actor") is not None:
                if event["actor"] not in winners:
                    winners.append(event["actor"])

        oya = start.get("oya")
        dealer_repeats = (
            next_round is not None
            and oya is not None
            and next_round.get("oya") == oya
        )
        round_number = start.get("kyoku")
        bakaze = start.get("bakaze")
        is_last_east = bakaze == "E" and round_number == 4
        is_south = bakaze == "S" or (
            isinstance(round_number, int) and round_number > 4
        )

        rounds.append({
            "index": position + 1,
            "bakaze": bakaze,
            "kyoku": round_number,
            "oya": oya,
            "honba": start.get("honba"),
            "terminal_type": terminal.get("type"),
            "terminal_reason": terminal.get("reason"),
            "end_game_after": end_game_after,
            "winners": winners,
            "start_scores": start_scores,
            "end_scores": end_scores,
            "score_source": end_score_key or start_score_key,
            "dealer_repeats": dealer_repeats,
            "next_bakaze": next_round.get("bakaze") if next_round else None,
            "is_last_east": is_last_east,
            "is_south_or_later": is_south,
            "tobi": bool(end_scores and min(end_scores) < 0),
            "has_30k": bool(end_scores and max(end_scores) >= 30000),
            "dealer_rank": _rank(end_scores, oya),
            "tenpai": _extract_tenpai(terminal)[0],
            "tenpai_source": _extract_tenpai(terminal)[1],
        })

    final_round = rounds[-1] if rounds else None
    return {
        "path": path,
        "event_count": len(events),
        "rounds": rounds,
        "round_count": len(rounds),
        "has_south": any(r["is_south_or_later"] for r in rounds),
        "has_tobi": any(r["tobi"] for r in rounds),
        "has_30k": any(r["has_30k"] for r in rounds),
        "has_repeat": any(r["dealer_repeats"] for r in rounds),
        "last_round": rounds[-1] if rounds else None,
        "last_round_is_e4": bool(final_round and final_round["bakaze"] == "E"
                                and final_round["kyoku"] == 4),
        "last_round_is_s4": bool(final_round and final_round["bakaze"] == "S"
                                and final_round["kyoku"] == 4),
        "has_end_game_event": any(
            event.get("type") == "end_game" for event in events
        ),
    }


def _infer(games):
    rounds = [r for game in games for r in game["rounds"]]
    last_east = [r for r in rounds if r["is_last_east"]]
    south = [r for r in rounds if r["is_south_or_later"]]
    final_rounds = [game["last_round"] for game in games if game["last_round"]]
    final_e4 = [r for r in final_rounds if r["bakaze"] == "E" and r["kyoku"] == 4]
    final_s4 = [r for r in final_rounds if r["bakaze"] == "S" and r["kyoku"] == 4]
    tenpai_observed = [r for r in rounds if r["tenpai"] is not None]
    end_game_after_round = [r for r in rounds if r["end_game_after"]]

    agari_yame_candidates = [
        r for r in last_east
        if r["terminal_type"] == "hora"
        and r["dealer_rank"] == 1
        and r["winners"]
        and r["oya"] in r["winners"]
        and r["next_bakaze"] is None
    ]
    # A missing next round is evidence of a terminal decision, not proof of
    # agari-yame: tobi, target score, or an omitted continuation can look the same.
    return {
        "games": len(games),
        "rounds": len(rounds),
        "with_south": sum(game["has_south"] for game in games),
        "without_south": sum(not game["has_south"] for game in games),
        "tonpuu_games": sum(not game["has_south"] for game in games),
        "hanchan_or_longer_games": sum(game["has_south"] for game in games),
        "with_tobi": sum(game["has_tobi"] for game in games),
        "tobi_rounds": sum(r["tobi"] for r in rounds),
        "final_tobi_games": sum(
            bool(game["last_round"] and game["last_round"]["tobi"])
            for game in games
        ),
        "with_30k": sum(game["has_30k"] for game in games),
        "with_dealer_repeat": sum(game["has_repeat"] for game in games),
        "terminal_types": Counter(
            r["terminal_type"] for r in rounds if r["terminal_type"]
        ),
        "bakaze": Counter(r["bakaze"] for r in rounds),
        "round_count": Counter(game["round_count"] for game in games),
        "last_east_hora_dealer_first_no_next": len(agari_yame_candidates),
        "south_rounds": len(south),
        "end_game_events": sum(game["has_end_game_event"] for game in games),
        "final_round_labels": Counter(
            f"{r['bakaze']}{r['kyoku']}" for r in final_rounds
        ),
        "final_e4_count": len(final_e4),
        "final_s4_count": len(final_s4),
        "end_game_after_round_count": len(end_game_after_round),
        "last_east_with_end_game": sum(
            r["is_last_east"] and r["end_game_after"] for r in rounds
        ),
        "last_south_with_end_game": sum(
            r["bakaze"] == "S" and r["kyoku"] == 4 and r["end_game_after"]
            for r in rounds
        ),
        "tenpai_observed_count": len(tenpai_observed),
        "tenpai_fields": Counter(
            r["tenpai_source"] for r in tenpai_observed
        ),
    }


def _print_game(game):
    print(f"\n{os.path.basename(game['path'])}")
    print(f"  manches={game['round_count']} evenements={game['event_count']}")
    for r in game["rounds"]:
        scores = (
            "scores=?"
            if r["end_scores"] is None
            else "scores=" + ",".join(f"{s:.0f}" for s in r["end_scores"])
        )
        print(
            f"  {r['bakaze']}{r['kyoku']} oya={r['oya']} "
            f"result={r['terminal_type'] or '?'} "
            f"repeat={'oui' if r['dealer_repeats'] else 'non'} "
            f"{scores} tobi={'oui' if r['tobi'] else 'non'} "
            f"end_game={'oui' if r['end_game_after'] else 'non'}"
        )
        if r["is_last_east"] or (r["bakaze"] == "S" and r["kyoku"] == 4):
            print(
                f"    ALL-LAST {r['bakaze']}{r['kyoku']}: "
                f"dealer_rank={r['dealer_rank']} winners={r['winners']} "
                f"tenpai={r['tenpai'] or '?'}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Dossier contenant les fichiers .zst")
    parser.add_argument("--files", type=int, default=10,
                        help="Nombre de parties à analyser (défaut: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", dest="json_path",
                        help="Ecrit aussi le rapport détaillé en JSON")
    args = parser.parse_args()

    paths = []
    for root, _, names in os.walk(args.data_dir):
        paths.extend(
            os.path.join(root, name)
            for name in names
            if name.endswith((".zst", ".json.zst"))
        )
    paths.sort()
    random.Random(args.seed).shuffle(paths)

    games = []
    for path in paths[:max(0, args.files)]:
        events = parse_mjai_zst_file(path)
        if events:
            games.append(_analyse_game(path, events))

    summary = _infer(games)
    print("=== Analyse des regles observees ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=list))
    print("\n=== Detail des parties ===")
    for game in games:
        _print_game(game)
    if not games:
        print("\nAucune partie lisible trouvee.")
        return 1
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "games": games}, handle,
                      ensure_ascii=False, indent=2)
        print(f"\nRapport JSON: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
