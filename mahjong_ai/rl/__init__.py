"""Package RL reworké (PPO on-policy mono-processus, envs vectorisés).

Remplace training/train_rl_suphx.py + self_play_worker.py +
parameter_server.py (déplacés dans training/legacy/).

Points clés du rework (cf. ANALYSE_RL.md pour le diagnostic complet) :
    * PPO strictement ON-POLICY : un batch de parties est collecté avec la
      policy courante, consommé en minibatches, puis JETE. Ratio = 1
      exactement au premier passage (vérifié par test_rl_rework.py).
    * Mono-processus : plus de Manager/Queue IPC (source des crashs et du
      staleness des données de l'ancien pipeline).
    * Forwards batchés : toutes les requêtes en attente des n_envs parties
      sont groupées par modèle (policy RL / discard SL / riichi / kong /
      pong / chow) et passées en UN forward par tick.
    * Mode par défaut : 1 siège RL vs 3 sièges SL figés -> le mean_rank
      des parties d'entraînement EST la métrique de progrès (2.50 = SL).
    * Guidage oracle 3 phases du papier conservé (gamma : 1 -> 0).
    * KL vers le SL désactivé par défaut (il annulait l'apprentissage).
"""
