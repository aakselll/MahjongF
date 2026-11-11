"""
classify_outcome() etait definie a l'identique dans multi_env.py (etape 1)
et multi_env2.py (etape 2) -- factorisee ici. L'environnement actif
(multi_agent.py) n'en a pas besoin : son _final() renseigne deja la cle
"outcome" directement. Cette fonction ne sert donc plus que de filet de
securite dans vectorized_runner.py et pour les etapes archivees (legacy/).
"""


def classify_outcome(final_rewards):
    """Classe le resultat d'une manche a partir de son dict final_rewards."""
    winners = [s for s, v in final_rewards.items() if v > 0]
    losers = [v for v in final_rewards.values() if v < 0]
    if not winners:
        return "exhaustive"
    if len(winners) == 1 and len(losers) == 3:
        return "tsumo"
    if len(winners) == 1:
        return "ron"
    return "multi_ron"
