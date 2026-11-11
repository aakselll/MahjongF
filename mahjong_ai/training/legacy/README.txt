Ancien pipeline RL (avant le rework mahjong_ai/rl/).

Ces fichiers sont conserves pour reference mais NE SONT PLUS UTILISES :
  train_rl_suphx.py   ancien trainer (replay off-policy, IPC Manager)
  self_play_worker.py ancien worker self-play (forwards batch-1)
  parameter_server.py ancien serveur de parametres IPC
  ppo_discard.py      premier trainer PPO mono-processus (obsolete)
  replay_buffer.py    buffer replay Manager (obsolete)

Le pipeline actif est mahjong_ai/rl/ (PPO on-policy mono-processus,
envs vectorises). Voir ANALYSE_RL.md a la racine pour le diagnostic.
Ils restent importables (imports absolus mahjong_ai.*) si besoin.
