# Fixes 2026-11 — test_red_fives.py + règle enchousen (game_env.py)

Investigation demandée : (1) pourquoi `test_red_fives.py` échoue alors que
les 3 cinq rouges (un par famille, +1 han chacun) sont censés être
supportés ; (2) la règle enchousen (extension Sud en mort subite, 30 000 pts)
est-elle bien appliquée, ou annulée par une limite de manches hard-codée ?

**Verdict court : le moteur des 5 rouges est sain ; la règle enchousen est
bien appliquée et aucune limite artificielle ne l'annule. Les échecs
venaient de 3 bugs dans les TESTS + 1 vrai (petit) bug de règle, corrigés
ci-dessous.**

---

## 1. test_red_fives.py — pourquoi il échouait (avec la vraie lib `mahjong`)

L'échec `Too many tiles = 17` (section C, bloc « meld pon de 5s + 5m rouge »)
avait **deux causes, toutes deux dans le test**, pas dans le moteur :

1. **État de jeu impossible** : le test injectait une main fermée de
   **14 tuiles + un pon** (17 tuiles vues par `shanten_of`). En partie réelle,
   une main avec 1 meld fait 13 − 3 + 1 = **11 tuiles fermées** (14 vues par
   le calculateur). La vraie lib `mahjong` valide le compte (≤ 14) et
   rejette ; le stub du dépôt ne validait pas → l'échec n'apparaissait
   qu'avec la vraie lib installée.
2. **Effet de bord `is_tenhou`** : l'état synthétique (0 défausse, 0 appel)
   déclenchait l'heuristique « première tuile » (`is_tenhou=True`), et la
   lib refuse `tenhou_with_meld_not_allowed` (main ouverte + tenhou).
   Corrigé en simulant une défausse (`_discard_count[0] = 1`).

Le bloc utilise désormais une main valide (234m 456m-rouge 55p 234s + pon
5s-rouge, tsumo sur 5p, tanyao ouvert) et — nouveau — **vérifie avec la
vraie lib que chaque 5 rouge vaut exactement +1 han** (yaku « Aka Dora ») :
main 3 han / 2 han / 1 han selon 2 rouges / 1 rouge / 0 rouge.
Résultat : **54/54 OK** (vrai calculateur, vraie lib shanten).

Côté moteur, rien à changer : l'invariant « main + melds = 13/14 tuiles »
est respecté sur **tous** les chemins d'appel shanten (vérifié y compris
ankan/kakan/daiminkan, ryuukyoku, furiten — jamais une requête à 12 ou 17
tuiles en partie réelle).

## 2. Enchousen — la règle est-elle appliquée ?

Règle MJSoul tonpusen (东风战南入, confirmée par les règles officielles) :
à la fin d'East 4, si personne n'a 30 000 pts → extension **Sud** 1-4 en
mort subite ; la partie s'arrête dès qu'un joueur atteint 30 000 pts ; à la
fin de Sud 4 sans franchissement, règlement forcé (le score max gagne).

**Réponse : OUI, appliquée, et PAS annulée par une limite cachée :**

- `GameEnv.n_rounds` (passé à 4 par rl/rollout.py et eval_stable_rank.py)
  est un **paramètre mort** — jamais lu. La progression réelle suit
  `N_ROUNDS_NORMAL=4` → `N_ROUNDS_MAX=8` (cap Sud 4) → `MAX_KYOKU=30`
  (simple sécurité anti-boucle : il faudrait ≈ 25 renchans consécutifs).
- Preuve empirique (nouveau `test_enchousen_live.py`, 60 parties réelles
  jouées avec la vraie lib) : 56/60 parties entrent en Sud quand personne
  n'a 30k ; toutes finissent en Sud 4 max ; le vainqueur = score max ;
  vent Sud correctement encodé (ctx + canaux obs `round_wind`,
  `is_extension`, `extension_index`, `round_id`). 13/13 OK.
- Vos logs RL le confirment aussi : `rounds=5.7–6.15` en moyenne (> 4).
- L'ancienne remarque d'ANALYSE_RL.md (« vent Ouest attendu ») était
  fausse : l'Ouest (西入) c'est l'extension du **hanchan** ; le tonpuu
  s'étend bien en **Sud**. Document corrigé.

**Un seul vrai bug de règle trouvé, corrigé** (`training/game_env.py`,
règle 4 de `_check_game_end`) : le check 30k de l'extension ne portait que
sur les **agari** (tsumo/ron/multi_ron). Un joueur franchissant 30 000 pts
par les **paiements de noten** sur une égalité (ryuukyoku) ne terminait
pas la partie — contraire à « 加时赛中一旦有玩家超过30000点即立刻结束
游戏 ». Le check couvre désormais aussi `exhaustive` (les manches abortives
ne déplacent aucun score, inutile de les couvrir). Impact en pratique :
rare (il faut être à 27-30k et toucher +3 000 de noten), mais réel.

**Caveat fidélité documenté (non corrigé, volontaire)** :
`_noten_payment_rewards` ajoute aux scores un artefact d'entraînement
(−0,1 × shanten par joueur sur ryuukyoku, ≤ ~0,4/manche). Les
`game_scores` de l'env dérivent donc légèrement des points MJSoul réels
(30k franchi un peu plus tard, quasi-égalités de rang influençables).
Si vous voulez une fidélité stricte, retirer la ligne
`r[s] -= EXHAUSTIVE_SHANTEN_COEF * max(sh, 0)` — mais ça change le
shaping utilisé par le pipeline RL actif (à faire entre deux runs).

## 3. test_end_rules.py — 2 échecs préexistants (données de test)

Les 2 cas « East 4, dealer 2eme » utilisaient des scores `[5.0, 15.0, …]`,
soit **40 000 pts** pour p1 dans la convention du moteur (0.0 = 25k,
`TARGET_SCORE` 5.0 = 30k) → `oorasu` se déclenchait légitimement.
Scores corrigés (p1 = 29,5k, dealer 2e) → les cas testent bien l'entrée
en enchousen. Mon fix règle 4 a par ailleurs exposé 3 autres vecteurs de
scores ≥ 30k dans des scénarios « personne à 30k » (Sud 4 / max_kyoku),
corrigés pareil. **34/34 OK** (dont 2 nouveaux cas : franchissement 30k
sur ryuukyoku / non-franchissement).

## Fichiers

| Fichier | Changement |
|---|---|
| `training/game_env.py` | FIX règle 4 (30k sur ryuukyoku aussi), en-têtes corrigés, note `n_rounds` inerte |
| `test_red_fives.py` | main valide 11 tuiles + pon ; vérif réelle +1 han/rouge avec la vraie lib |
| `test_end_rules.py` | 5 vecteurs de scores corrigés + 2 nouveaux cas règle 4 |
| `test_enchousen_live.py` | NOUVEAU — intégration enchousen sur parties réelles |
| `ANALYSE_RL.md` | section 6 et P8 mis à jour (résolution des 2 points) |
| `fixes.patch` | diff unifié des 3 fichiers modifiés vs la livraison `mahjong_ai_rl_rework` |

## Application (depuis E:\MahjongE)

```powershell
# soit copier les fichiers aux memes emplacements, soit :
git apply fixes.patch        # si le depot est versionne et propre
# test_enchousen_live.py est un NOUVEAU fichier : a copier a la racine

# Vérification (votre machine, vraies libs) :
py -3.14 -m mahjong_ai.test_end_rules        # 34/34
py -3.14 test_red_fives.py                   # 54/54
py -3.14 -m mahjong_ai.test_enchousen_live   # 13/13 (~2-3 min, 60 parties)
py -3.14 -m mahjong_ai.test_env_rules        # 68/68 (regression)
```

Toutes ces suites ont été validées ici avec la vraie lib `mahjong` +
torch CPU. Note : `test_v9_returns.py` est cassé en amont (importe
`mahjong_ai.training.self_play_worker`, déplacé en `training/legacy/`) —
sans lien avec ces correctifs.
