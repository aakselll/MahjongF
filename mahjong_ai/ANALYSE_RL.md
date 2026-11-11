# ANALYSE ET REWORK DE LA PHASE RL — mahjong_ai

**Diagnostic : pourquoi l'entraînement RL n'apprenait rien, et ce qui a été refait.**

Le pipeline RL d'origine (`training/train_rl_suphx.py` + `self_play_worker.py`
+ `parameter_server.py`) cumulait des défauts algorithmiques qui, pris
ensemble, rendent l'apprentissage **mathématiquement impossible** à
l'échelle du projet. Ce document détaille chaque cause avec ses lignes de
code, décrit le rework (`mahjong_ai/rl/`), et montre la validation.

---

## 1. Résumé exécutif

| # | Défaut de l'ancien pipeline | Impact | Correction dans le rework |
|---|---|---|---|
| P1 | PPO **off-policy** : buffer replay FIFO de 150 parties, échantillonnage aléatoire, réutilisation illimitée, poids publiés aux workers toutes les N steps | ratio d'importance ≠ 1, avantages sans rapport avec la policy courante, clip PPO qui annule les gradients | batch collecté puis **jeté** ; ratio = 1 au 1er passage (mesuré 8e-08) |
| P2 | **Budget de gradient ~100× trop petit** : 3 pas de gradient par 1000 transitions | 1500 pas sur 500k steps pour un CNN de 6,7M params → la policy ne bouge pas | minibatches : ~32 pas/update → ~30× plus d'optimisation |
| P3 | **Self-play symétrique par défaut** (4 sièges, même policy, rewards somme-nulle) | gradient de 1er ordre ≈ 0 ; aucun progrès mesurable (mr=2.50 par symétrie) | défaut : **1 siège RL vs 3 SL figés** → mr < 2.50 = progrès |
| P4 | **Ancre KL 0.05 vers le SL** pendant 80% du run | la policy est attirée en permanence vers le SL ; v10 : ‖W_o‖ 4.0→0.18 | KL **désactivée** par défaut (optionnelle via `--kl-coef`) |
| P5 | **Entropie adaptative erratique** : h_target=2.0 inatteignable, alpha au cap | bonus d'entropie ~7× la policy loss → policy aplatie → 34% de 4es places | entropie **off** par défaut ; mode adaptatif recalibré (cible = entropie mesurée) |
| P6 | **Collecte non batchée** : forwards batch-1 CPU, têtes SL appelées une à une dans l'env, DFS look-ahead qui crash en silence | ~1-3 min/partie → très peu d'updates | **envs vectorisés** (n_envs parties interleavées), forwards groupés par modèle, look-ahead réparé puis coupé du défaut RL |
| P7 | **Reward GRP sans correction terminale** : Σ deltas = Φ_final (prédiction) ≠ résultat réel | signal biaisé par l'erreur du GRP, le vrai classement n'entre jamais dans le RL | correction terminale : **Σ r_k = reward de rang vrai** (testé exact) |
| P8 | Divers bugs (look-ahead mort, docstrings périmées, dropout oracle incohérent) | 80 canaux d'entrée constants ; confusion | **look-ahead RÉPARÉ** (31 tests) mais off par défaut (checkpoints SL entraînés canaux à zéro) ; reste documenté |

L'ancien code **savait** qu'il était malade — ses propres commentaires
documentent les symptômes (rdev non nul « car le buffer est un replay »,
« budget ~100x trop petit », « 0.25/2.50 par symétrie, PAS une mesure de
progres », « alpha a fini a 0.41... policy aplatie ») — mais les patchs
successifs (v8→v11) traitaient les symptômes, jamais les causes P1-P3.

---

## 2. Diagnostic détaillé

### P1 — PPO off-policy (la cause racine)

`train_rl_suphx.py` :
- les workers poussent des **parties entières** dans une `mp.Queue`, vidée
  dans un buffer local FIFO de `--buffer-size` 150 parties (l.512-547) ;
- le trainer attend 20 parties puis **échantillonne au hasard dans tout le
  buffer** : `np.random.choice(buf_len, size=20, replace=False)` (l.767) ;
- `--max-episode-reuse` défaut **0 = réutilisation illimitée** (l.1195) ;
- les poids ne sont publiés aux workers que tous les `--publish-every`
  steps (défaut 2000) — les parties du buffer ont donc été jouées par des
  politiques différentes, parfois vieilles de 150 parties.

PPO n'est valable que pour de petites déviations entre la policy qui a
généré les données (π_old) et la policy optimisée (π). Ici
`ratio = exp(logp_new − logp_old)` est structurellement loin de 1 : le
code le **mesurait** (`rdev`, l.406-409, en avouant la cause) mais
continuait. Conséquences :
1. les avantages (calculés sur des trajectoires d'autres policies) ne
   correspondent plus à la policy courante → gradient **biaisé** ;
2. dès que le ratio sort de [0.8, 1.2], le clip met le gradient à zéro
   → la policy **n'apprend plus** (et quand il ne sort pas, elle apprend
   sur du bruit).

### P2 — Budget de gradient ~100× trop petit

Chaque update consomme ≤ 1000 transitions (`--max-batch-transitions`)
et fait `--ppo-epochs` 3 epochs **full-batch** = **3 pas de gradient**
(l.877-917). `step += len(batch)` (l.942) : 500k steps → 500 updates →
**~1500 pas de gradient au total** pour 6,7M de paramètres.

Le commentaire du code (l.829-839) fait ce calcul pour le **critic**
(« V8 n'a donné que ~800 pas de gradient au critic en 228k steps :
budget ~100x trop petit ») et ajoute une inner-loop pour lui — mais la
**policy** garde le même budget famélique. Pour comparaison, le rework
fait `epochs × ceil(B/minibatch)` pas par itération (ex. 4 × 8 = 32).

### P3 — Self-play symétrique : signal nul

Sans `--opponent-sl` (défaut), les 4 sièges jouent la même policy et se
partagent des rewards quasi somme-nulle (les deltas GRP somment à ~0 sur
les 4 sièges). Améliorer au siège A dégrade B/C/D qui partagent les mêmes
poids : le gradient utile de 1er ordre s'annule. Le code l'admet
(l.135-139) : « les 4 sieges partagent la meme policy -> 0.25 / 2.50 par
symetrie, PAS une mesure de progres ».

Le papier Suphx utilise bien du self-play, mais avec 25,7M de steps, des
batchs énormes et l'oracle guiding comme accélérateur. À l'échelle de ce
repo, le mode **1 siège RL vs 3 SL figés** fournit un signal direct (tout
gain contre des adversaires figés est du progrès réel) et une métrique
lisible en continu : `mean_rank < 2.50`.

### P4 — Ancre KL vers le SL

`--kl-coef` défaut **0.05** (l.1247) : en phases 2/3 (80% du run avec les
fractions par défaut), la loss inclut `0.05 × KL(π ‖ π_SL)` qui tire la
policy vers le SL de départ — aveugle à l'oracle. Le code documente le
désastre résultant (v10 : effondrement de ‖W_o‖ de 4.0 à 0.18) et a
rajouté des garde-fous (`apply_phase1_guards`, `--kl-ref`) au lieu de
retirer la cause. Le papier Suphx **n'a pas** d'ancre KL en RL.

### P5 — Entropie adaptative

Le papier adapte α par α ← α + β(H_target − H) avec β=0.005. Ici
`--h-target` défaut **2.0** est inatteignable (une policy discard
compétente plafonne à H≈1.7) → α monte en permanence au cap (0.15) et le
bonus d'entropie domine la policy loss (run 600k documenté : α=0.41,
**34% de 4es places**). La warm-start SL a déjà une entropie saine ;
l'exploration vient du sampling PPO.

### P6 — Collecte non batchée

- chaque décision = forward **batch-1** du CNN (7M params) sur CPU ;
- les têtes SL (riichi/kong/pong/chow) sont décidées **dans l'env** via
  `_sl_decide` (multi_agent.py l.807) → jamais batchables ;
- `compute_lookahead=True` par défaut → à CHAQUE requête DISCARD le
  builder tente le DFS look-ahead… qui **crash** (voir P8) en brûlant du
  CPU (réinstanciation de `Shanten()`, DFS partiellement exploré, etc.).
  Une partie ≈ 1-3 minutes ; 20 parties/update → dizaines de minutes
  entre deux updates → P2 devient inévitable.

### P7 — Reward GRP sans correction terminale

`game_env.py` renvoie `final_rewards = deltas GRP` (l.285) ;
`self_play_worker.py` les attribue aux transitions. La somme des deltas
d'une partie = Φ_final (la **prédiction** du reward de rang), jamais le
vrai résultat : le reward de rang réel (`RANK_REWARDS` {+15,+5,−5,−15})
n'entre **nulle part** dans le signal RL. Avec un GRP entraîné sur
quelques dizaines de milliers de parties, l'erreur de prédiction biaisé
tout le run.

### P8 — Bugs annexes (vérifiés empiriquement)

- **Look-ahead mort** (`suphx_features.py`) : le DFS `compute_lookahead`
  levait `UnboundLocalError` (l.187 rend `remaining` local, lu l.180),
  avalée par `except Exception: continue` (l.251-255) → les **80 canaux
  look-ahead étaient constants à zéro**, en SL comme en RL. En outre :
  la condition d'émission `sum(remaining)==0` (l.175) était inatteignable
  (la paire n'était pas modélisée dans `_MELD_OPTIONS`), et le scoring
  passait des comptages 34 à `estimate_hand_value` qui attend des ids
  136 (et la main complète, melds inclus).
  **→ RÉPARÉ (2026-11)** : DFS réécrit (paire modélisée, ordre canonique,
  chiitoitsu + kokushi, kans = figure de 3, approfondissement iteratif
  budgets 2→5, contrainte unseen), scoring en ids 136 avec cache
  `_score_with_dora_cached`, monotonie exacte des 4 lignes, config
  dealer dans la clé du cache. **31 assertions** (`test_lookahead_fix.py`)
  + régressions vertes (test_env_rules 68/68, test_rl_rework complet).
  Mesuré : ~17 ms à froid / 0,3 ms à chaud par état ≤ 2 shanten.
  **Conséquence SL** : les checkpoints existants ont été entraînés avec
  ces canaux à zéro → le défaut RL reste `compute_lookahead=False`
  (aucun ré-entraînement nécessaire) ; pour EN bénéficier, ré-entraîner
  (ou fine-tuner) le SL avec le look-ahead actif puis passer
  `--compute-lookahead` au RL.
- Dropout oracle **par élément** dans `select_action` vs **par canal**
  dans `forward_with_dropout` (granularités incohérentes).
- `game_env.py` : en-tête mensonger (« enchousen désactivée » alors qu'elle
  est active) — **corrigé (2026-11)**, cf. section 6 : l'extension Sud est
  le comportement CORRECT (MJSoul 东风战南入), le vrai bug était le check
  30k limité aux agari (corrigé).
- Docstrings périmées partout (250/386/136 canaux vs 279/283/4 réels).
- `ppo_discard.py` (GRP `hidden_dim=128` hardcodé vs 64 réel) et
  `replay_buffer.py` : code mort du pipeline actif.

---

## 3. Le rework : `mahjong_ai/rl/`

```
rl/
├── agent.py     PPOAgent : policy 283 canaux + critic asymétrique (oracle
│                complet, réutilisé de v9) + têtes SL figées + discard SL
│                des adversaires. TOUTES les décisions sont batchées
│                (act_rl / act_sl_discard / act_head). Checkpoint enrichi
│                {'policy','critic','config','step','alpha','phase',
│                'optimizer_*'} — clés 'policy'/'critic' inchangées pour
│                eval_stable_rank.py / export_rl_discard.py.
├── rollout.py   VecRunner : n_envs parties interleavées avec pool de
│                slots ; requêtes groupées par modèle et résolues en UN
│                forward batché par tick ; récompenses fidèles au papier
│                (delta GRP par manche, attribué à toutes les transitions
│                du siège dans la manche) + CORRECTION TERMINALE
│                (Σ r_k = reward de rang vrai) ; G_t = retour restant
│                escompté (ρ=0.85) ; pénalité de danger optionnelle ;
│                mode éval (argmax, gamma=0, sans collecte).
├── ppo.py       update PPO : avantages batch_mean (papier) ou critic V(s)
│                avec repli EV ; minibatches ; value clipping ; IW
│                rejection phase 3 (papier §3.3) ; entropie/KL optionnelles.
└── train.py     orchestrateur : schedule 3 phases (oracle 30% / dropout
                 40% / normal 30% par défaut), warmup critic optionnel,
                 fenêtre glissante de stats de jeu, éval inline, jalons,
                 resume complet, fail-fast sur requêtes non gérées.
```

**Choix structurants** (validés avec le propriétaire du repo) :
1. **1 siège RL vs 3 SL figés** par défaut (`--mode opponent`) ;
   `--mode selfplay` disponible.
2. **Mono-processus** plutôt que workers+IPC : le staleness disparaît
   par construction (plus de poids publiés en retard), le batching
   devient possible, et tout l'historique de crashs IPC/Manager du
   repo (commentaires du code) devient sans objet. Sur machine
   multi-cœurs, augmenter `--n-envs`.
3. **Guidage oracle 3 phases conservé** (papier §3.3) : γ=1 → dropout
   linéaire → γ=0 ; l'oracle masqué est stocké dans la transition pour
   que l'évaluation PPO revoie exactement la même entrée.
4. **KL désactivée** par défaut ; **entropie off** par défaut (les deux
   régularisateurs qui étouffaient l'apprentissage).
5. **Delta GRP + correction terminale** : r_k = Φ(x^k) − Φ(x^(k−1)) et,
   sur la dernière manche, += (R_vrai − Φ_final). C'est un potential-based
   shaping exact : Σ r_k = R_vrai, la policy optimale n'est pas modifiée,
   et le biais de prédiction du GRP disparaît. Sans checkpoint GRP :
   repli sur le score de manche /1000 (sans correction).
6. **Look-ahead réparé mais coupé par défaut** (`compute_lookahead=False`,
   cf. P8) : les checkpoints SL actuels ont été entraînés avec ces 80
   canaux à zéro ; les activer sans ré-entraîner le SL décalerait la
   distribution d'entree. `--compute-lookahead` les active (SL
   ré-entraîné requis).

### Ce qui n'a PAS changé
- `envs/multi_agent.py` (moteur de jeu), `agents/suphx_cnn.py`,
  `agents/oracle_agent.py` (le `Critic` est
  réutilisé), `training/game_env.py` (règles multi-manches + GRP ;
  seul ajout : le paramètre `compute_lookahead`), tout le pipeline SL,
  `eval_stable_rank.py`, `export_rl_discard.py`.
- Les têtes riichi/kong/pong/chow restent les têtes SL figées (seule la
  tête discard est entraînée, comme dans le papier). Les requêtes RON
  sont toujours acceptées (l'ancien chemin refusait un cas étroit de
  dernière manche via une règle interne).

---

## 4. Validation

`test_rl_rework.py` (racine du repo, checkpoints factices 32ch×2blocs,
CPU, ~1 min) :

| Test | Résultat |
|---|---|
| 1. ratio PPO au 1er passage | **7.8e-08** (on-policy strict) |
| 2. Σ rewards de manche par siège | **= reward de rang vrai** (±1e-3) |
| 3. G_t | = Σ ρ^(k−1) · r_k (algorithme v9 exact) |
| 4. update PPO | poids de la policy modifiés |
| 5. save/resume | round-trip complet (poids + optimizers + step + phase) |
| 6. γ ∈ {1.0, 0.5, 0.0}, selfplay, éval | parties complètes, 0 requête non gérée |
| 7. smoke train | 2 itérations complètes, transitions de phase 1→2→3 |

**Démonstration d'apprentissage** (modèles factices, 2 CPU, 7 min,
`--danger-penalty 1.0`, oracle γ=1, lr 3e-4, 40k steps) :

```
[step  4682] GAME: n=32  wr=0.312 mr=2.094±0.350   H=2.282  ||W_o||=2.078
[step 20558] GAME: n=80  wr=0.537 mr=1.788±0.221   H=2.207  ||W_o||=2.098
[step 40315] GAME: n=80  wr=0.525 mr=1.712±0.203   H=1.797  ||W_o||=2.121
           rdev=0.0000 sur tous les updates (35 updates, 696 pas de gradient)
```

Le siège RL part de ~2.1 et descend à **1.71 ± 0.20** contre des
adversaires figés : la boucle collecte → avantages → PPO apprend
réellement. (Avec de VRAIS checkpoints SL + GRP et un GPU, la même
boucle s'attaque maintenant à la tâche réelle.)

---

## 5. Utilisation

```bash
# entraînement (défauts sains : 1 RL vs 3 SL, oracle 30/40/30, PPO on-policy)
python -m mahjong_ai.rl.train \
    --sl-discard checkpoints/sl_tonpusen_light/discard.pt \
    --sl-frozen   checkpoints/sl_tonpusen_light \
    --grp-checkpoint checkpoints/grp_tonpusen.pt \
    --output checkpoints/rl_v3/rl_discard.pt \
    --total-steps 1000000 \
    --n-envs 16 --batch-games 32

# éval (inchangé, format checkpoint compatible)
python -m mahjong_ai.training.eval_stable_rank \
    --agent-checkpoint checkpoints/rl_v3/rl_discard_latest.pt \
    --sl-discard ... --sl-frozen ... --weak-dir ... --n-games 200

# tests du rework
python test_rl_rework.py
```

### Que regarder dans les logs
- `mr=...` (ligne GAME) : mean_rank du siège RL vs 3 SL sur la fenêtre.
  **2.50 = niveau SL ; < 2.50 = progrès.** C'est LA métrique.
- `rdev` : doit rester ~0.000 (si ça monte, bug off-policy — à signaler).
- `dng%` : % de défausses dangereuses (avec `--danger-penalty`); doit
  chuter en phase 1 = la policy exploite l'oracle.
- `H` : entropie ; elle doit rester dans une zone saine (1.0-1.8 pour
  une policy discard compétente). Chute < 0.5 = effondrement.
- `||W_o||` : usage des canaux oracle (croît en phase 1 si la policy
  s'en sert).
- `EVAL gamma=0` (avec `--eval-every`) : niveau en conditions de
  déploiement — l'écart avec la ligne GAME se referme en phases 2/3.

### Hyperparamètres recommandés
- Commencer par `--total-steps 300000 --oracle-frac 0.30
  --transition-frac 0.40` et valider que `mr` décroît en phase 1 avant
  d'allonger.
- GPU : le trainer met tout sur `cuda` automatiquement ; monter
  `--n-envs 32 --batch-games 64` pour saturer le batching.
- `--adv-mode critic` (+30 updates de warmup) si la variance des
  avantages batch_mean freine ; le repli EV est automatique.
- Garder `--entropy-mode off` tant que la policy SL warm-start est
  correcte ; `adaptive` (cible auto) si exploration insuffisante.

## 6. Limites et suites possibles
- ~~Le look-ahead reste mort (P8)~~ **Réparé (2026-11)** : DFS + scoring
  réécrits et testés (`test_lookahead_fix.py`). Pour en bénéficier :
  ré-entraîner (ou fine-tuner) le SL avec `compute_lookahead_flag=True`
  dans le parser, puis `--compute-lookahead` en RL. Le conditionnement
  par tuile défaussée (34 sous-canaux par ligne au lieu d'un broadcast)
  reste une extension possible.
- ~~`test_red_fives.py` échoue en section C~~ **Résolu (2026-11)** :
  double bug dans le TEST lui-même (pas dans le moteur) —
  (1) état injecté impossible : main fermée de 14 tuiles + pon (17
  tuiles) que la vraie lib `mahjong` rejette (`Too many tiles = 17`,
  le stub ne validait pas le compte) ; (2) l'état synthétique (0
  defausse, 0 appel) déclenchait `is_tenhou` et la lib refusait
  `tenhou_with_meld_not_allowed`. Le test utilise maintenant une main
  valide de 11 tuiles + pon et vérifie avec la VRAIE lib que chaque 5
  rouge vaut bien **+1 han** (aka dora, yaku « Aka Dora ») : 54/54 OK.
  Le moteur lui-même n'a jamais été en cause (invariant main+melds =
  13/14 respecté en partie réelle, vérifié sur tous les chemins d'appel
  shanten y compris kans).
- ~~L'enchousen de `game_env.py` est contradictoire~~ **Investigué et
  corrigé (2026-11)** : l'extension Sud est la bonne règle MJSoul pour le
  tonpuu (东风战南入 :东4 sans joueur à 30k → Sud 1-4 en mort subite,
  fin dès qu'un joueur atteint 30k, sinon règlement forcé à la fin de
  Sud 4, le score max gagne). Aucune limite artificielle ne l'annule :
  `n_rounds` (param de GameEnv) est mort, `MAX_KYOKU=30` n'est qu'une
  sécurité anti-boucle (≈ 25 renchans consécutifs pour l'atteindre). Vérifié en
  intégration par `test_enchousen_live.py` (parties réelles : vent Sud
  encodé, cap Sud 4, vainqueur = score max, 13/13). Un seul écart
  trouvé et corrigé dans `game_env.py` : le check 30k de l'extension ne
  portait que sur les agari — un franchissement via paiements de noten
  (ryuukyoku) ne terminait pas la partie, contrairement à la règle
  « 加时赛中一旦有玩家超过30000点即立刻结束游戏 ». Le check couvre
  désormais tsumo/ron/multi_ron/exhaustive (les manches abortives ne
  changent aucun score).
  [RETRAIT SHAPING 2026-11] L'artefact d'entraînement −0,1×shanten par
  joueur sur ryuukyoku (`EXHAUSTIVE_SHANTEN_COEF`, jusqu'à ~0,4/manche)
  a été **supprimé** de `_noten_payment_rewards` : paiements de noten
  exacts (pool 3000 réparti tenpai/noten, rien si 0 ou 4 tenpai),
  `game_scores` conservatifs (somme nulle) — plus de dérive sur les
  seuils 30k/tobi/rangs ni de quasi-égalités de rang. Motifs : le terme
  taxait le fold (défense) de 100 pts fantômes par tuile d'écart au
  tenpai, signal redondant avec le paiement noten réel, et cassait la
  conservation de l'argent.
  **Bug démasqué par le retrait** (corrigé dans le même patch) : à
  l'exhaustive sans aucun tenpai, le paiement vaut 0 et le proxy
  `paiement >= 0` de `_decide_dealer_repeat` donnait renchan à tort
  (règle : dealer noten ⇒ le vent avance). L'artefact rendait le
  paiement strictement négatif et masquait le cas ; sans lui, 85 % des
  parties aléatoires bouclaient en renchans jusqu'au cap MAX_KYOKU
  (reproduit par `test_enchousen_live` : 51/60 fins max_kyoku).
  Le renchan à l'exhaustive se décide désormais sur le **tenpai réel**
  du dealer (`shanten_of(_shanten_counts(0)) == 0`), identique au
  comportement d'avant-retrait dans tous les cas où le proxy était
  correct. Patch : `remove_exhaustive_shanten_penalty.patch`
  (test_riichi_sticks sans neutralisation 13/13 ; test_end_rules
  section 9 : 9 nouveaux checks renchan ; test_enchousen_live 13/13 ;
  test_env_rules 68/68). Attention : changement d'échelle de reward →
  appliquer en début de run, pas en cours d'entraînement.
- `test_v9_returns.py` importe `mahjong_ai.training.self_play_worker`
  (déplacé en `training/legacy/`) : cassé en amont, sans lien avec le
  rework.
- Pool d'adversaires (anciens checkpoints) en complément du mode
  opponent pour limiter le sur-ajustement aux SL.
- pMCPA (adaptation en ligne au déploiement) : hors périmètre.
