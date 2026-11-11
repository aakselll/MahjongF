================================================================================
MAHJONG AI - SUPHX REPRODUCTION (REDUCED SCALE)
================================================================================

A PyTorch implementation of Suphx (Microsoft Research Asia, 2020), an AI for
4-player Japanese Riichi Mahjong based on deep reinforcement learning, at a
reduced parameter scale (~7M params per SL model instead of ~21M in the paper).

Reference: Suphx: Mastering Mahjong with Deep Reinforcement Learning
           arXiv:2003.13590 (https://arxiv.org/abs/2003.13590)


================================================================================
TABLE OF CONTENTS
================================================================================

  1. Overview
  2. Pipeline
  3. Architecture
  4. Input Features
  5. Directory Structure
  6. Installation
  7. Data Preparation
  8. Training
  9. Inference / Play
 10. Deviations from the Paper
 11. Performance Notes
 12. References


================================================================================
1. OVERVIEW
================================================================================

Suphx is a Mahjong AI that reached 10-dan on Tenhou.net (the highest rank) and
is rated above 99.99% of ranked human players. Its contributions are:

  1. Global Reward Prediction (GRP)
     A GRU-based predictor that maps the final game reward (rank-based) to
     per-round signals, enabling tactical play (e.g. defend when leading in the
     last round) without hand-coded rules.

  2. Oracle Guiding
     An oracle agent with access to perfect information (opponents' hands,
     wall) is gradually transitioned to a normal agent via feature dropout,
     accelerating RL training.

  3. Parametric Monte-Carlo Policy Adaptation (pMCPA)
     At run time, the offline policy is fine-tuned to the current round's
     initial hand using simulated rollouts, without building a regular game
     tree.

This repository reproduces the supervised learning (SL) stage and the GRP
component at reduced scale, using the public dataset:
    nikketryhard/tenhou-houou-mjai  (HuggingFace)


================================================================================
2. PIPELINE
================================================================================

  +----------------------------------------------------------------------+
  |  STAGE 1 - Supervised Learning (SL)                                  |
  |                                                                      |
  |   .mjai.json.zst  -->  mjai_parser  -->  PolicyDataset               |
  |                                              |                       |
  |                                              v                       |
  |                         5 independent CNNs (Suphx architecture)      |
  |                          * discard  (34 logits)                      |
  |                          * riichi   (2 logits)                       |
  |                          * chow     (4 logits: pass/lower/mid/upper) |
  |                          * pong     (2 logits)                       |
  |                          * kong     (2 logits)                       |
  |                                                                      |
  |   Loss: cross-entropy, split train/val by FILE (no leakage)          |
  +----------------------------------------------------------------------+
                                    |
                                    v
  +----------------------------------------------------------------------+
  |  STAGE 2 - Global Reward Predictor (GRP)                             |
  |                                                                      |
  |   GameLogBuffer (per-round feature sequence + final rank reward)     |
  |                                              |                       |
  |                                              v                       |
  |                2-layer GRU + 2 FC -> predicts final game reward      |
  |                                                                      |
  |   Per-round RL reward:  r_k = Phi(x^k) - Phi(x^{k-1})                |
  |   (replaces raw round score, captures "rank protection" behaviour)   |
  +----------------------------------------------------------------------+
                                    |
                                    v
  +----------------------------------------------------------------------+
  |  STAGE 3 - Reinforcement Learning (RL)                               |
  |                                                                      |
  |   Distributed PPO self-play (vectorized_runner.py)                   |
  |   + entropy regularization                                           |
  |   + optional oracle guiding (perfect-info dropout schedule)          |
  +----------------------------------------------------------------------+


================================================================================
3. ARCHITECTURE
================================================================================

CNN trunk (shared by all 5 SL models)
-------------------------------------

Faithful to Suphx section 2.2 / Figures 4 and 5:

  * Input:  (N_CHANNELS, 34, 1)  - no pooling
  * 3x1 conv, C=256, BatchNorm, ReLU
  * N_blocks residual blocks of two 3x1 conv C -> C with skip connections
  * Discard head: 1x1 conv 1 -> flatten -> 34 logits
  * Binary heads (riichi/chow/pong/kong):
        3x1 conv 32 -> flatten -> 2 FC layers -> 2 or 4 logits

Default (reduced scale):  C=256, N_blocks=16  ->  ~6.7M params per model
Reference paper scale:    C=256, N_blocks=50  ->  ~21M params

Reward Predictor (GRP)
----------------------

  * Input: per-round feature vector (FEATURE_DIM=15) sequence
    (8 game features + format, horizon flags, honba, kyotaku and
    normalized rounds remaining)
    (8 game features + explicit tonpuu/hanchan format flags)
  * 2-layer GRU (hidden=64 or 128) + 2 FC layers -> scalar output
  * Trained by MSE on final rank rewards, with separate hanchan and tonpuu
    views extracted from the same hanchan logs
  * Frozen during RL (as in the paper)


================================================================================
4. INPUT FEATURES
================================================================================

The observation is a matrix (N_CHANNELS, 34), one row per tile type.
This follows Suphx section 2.2: each feature occupies a 34-wide channel.

  Group                          Channels    Encoding
  ---------------------------------------------------------------------------
  Horizon format                    2         tonpuu / hanchan
  Last normal round                 1         one-hot flag
  Extension                         1         one-hot flag
  Extension index                   8         one-hot bucket
  Round progress                   10         one-hot bucket
  Own hand                          4         "n copies" (Fig. 3):
                                              channel n = 1 if count >= n+1
  Own open melds                    4         same
  Own closed kan                    4         same
  Discards per seat (x4)            4 each    "n copies" per seat
  Last discard per seat (x4)        1 each    one-hot
  Tsumogiri count per seat (x4)     1 each    count per tile
  Opponent melds (x3)               4 each    "n copies"
  Dora indicators                   1         count
  Dora tiles (derived)              1         count
  Pending claim tile                1         one-hot
  Unseen tile counts                1         normalized [0, 1]
  Own riichi                        1         all-ones if active
  Opponents' riichi (x3)            1 each    all-ones if active
  Own furiten                       1         all-ones if permanent furiten
  Seat wind                         4         one-hot
  Round wind                        4         one-hot
  Dealer                            1         binary
  Round id (1-4)                    4         one-hot
  Honba (repeat dealer)             5         bucket (0,1,2,3,4+)
  Kyoutaku (riichi sticks)          5         bucket
  Own score                        20         bucketed
  Each opponent's score (x3)       20 each    bucketed
  Wall remaining                   10         bucketed
  Look-ahead                       80         see below

Look-ahead features (Suphx section 2.2)
---------------------------------------

The paper says:
  "a feature represents whether discarding a specific tile can lead to a
   winning hand of 12,000 round score with replacing 3 hand tiles by tiles
   drawn from the wall or discarded by other players."

Implemented as 4 x 20 = 80 channels:

  * 4 replacement depths: n_replace in {1, 2, 3, 4}
  * 20 score buckets (0, 1k, 1.5k, 2k, ..., 96k, inf)

Channel (n_replace, bucket) = 1 iff a winning hand with that score bucket
is reachable by drawing at most n_replace + 1 tiles from the wall
(n_replace swaps + the winning tile). Tenpai and 1-shanten light up
n_replace=1; 2-shanten needs n_replace >= 2. The 80-channel block is
broadcast over the 34 columns (row-wide 0/1 planes, like the other scalar
encodings); per-discard conditioning is a possible extension.

REPAIRED (2026-11). The original DFS never produced a single active
channel (three bugs: an UnboundLocalError silently swallowed, the pair
was not modelled so the emission condition was unreachable, and scoring
passed 34-counts to a library expecting 136-ids). The rewrite:

  * Standard form DFS: (4 - melds) sets + THE PAIR, canonical meld order
    (no permutation blow-up), cost = tiles drawn from the wall
  * Chiitoitsu (7 distinct pairs) and kokushi enumerated when no meld
  * Kans count as a 3-tile set (4th tile is out of structure, matches
    the engine's _shanten_counts)
  * Iterative deepening over budgets 2..5: each level finds its wins
    while pruning is still strong; results deduplicated, sorted by cost
  * Unseen constraint: drawn tiles must still be in the wall from the
    seat's point of view (all 4 copies visible -> win filtered out)
  * Scoring: full 136-id hand (winning concealed part + ALL meld tiles)
    via estimate_hand_value, cached per (hand, melds, winds, dora)
  * Rows are monotone by construction (row k lit => row k+1 lit)
  * Scoring config unchanged (ron, no riichi): hands without an
    intrinsic yaku (e.g. pure runs with a penchan wait, no pinfu/tanyao)
    light no channel - original semantics

Optimizations applied (speed vs. fidelity trade-offs):
  * Skip DFS when shanten > 2 (MAX_LOOKAHEAD_SHANTEN)
  * MAX_DFS_RESULTS = 30 per budget level, MAX_VISITED = 4000
  * _LOOKAHEAD_SCORE_CAP = 40 cheapest wins scored per state
  * Two-layer lru_cache: DFS results cached independently from scoring
  * HandConfig cached per (seat_wind, round_wind, is_dealer)
  * Single module-level Shanten() instance

Measured (fixed version, CPU): ~17 ms cold / ~0.3 ms warm per eligible
(<= 2 shanten) state; states further from tenpai return instantly.

IMPORTANT - training compatibility:
  * Existing SL checkpoints were trained with these 80 channels ALWAYS
    ZERO (the bug), so they work unchanged with lookahead OFF (default).
  * To USE the lookahead: retrain (or fine-tune) SL with
    compute_lookahead_flag=True in mjai_parser, then run RL with
    --compute-lookahead. Turning it on with current checkpoints would
    shift the input distribution on 80 of 283 channels and likely
    DEGRADE the policy.

Known deviations:
  * Paper says "100+ look-ahead features"; we use 80
  * Paper example uses "3 replacements"; we use up to 4
  * Not per-discard (broadcast row) - see above
  * Wind-pair yakuhai is not scored by the mahjong lib (pon/kan only)
  * Caps may truncate rare expensive restructures (cheap wins first)


================================================================================
5. DIRECTORY STRUCTURE
================================================================================

mahjong_ai/
|
+-- envs/
|   +-- tiles.py                 # shanten, tile constants
|   +-- multi_agent.py           # active game environment (RL)
|   +-- outcomes.py              # round outcome classification
|   +-- suphx_features.py        # (N_CHANNELS, 34) builder - shared SL/RL
|
+-- sl/                          # SUPERVISED LEARNING
|   +-- mjai_parser.py           # .mjai.json.zst -> (obs, action, mask)
|   +-- download_mjai.py         # downloads annual shards from HuggingFace
|   +-- dataset.py               # PolicyDataset, split-by-file
|   +-- train_policy_sl.py       # trains ONE head
|   +-- train_all_sl.py          # orchestrates the 5 heads
|
+-- agents/
|   +-- suphx_cnn.py             # CNN trunk + heads (Fig. 4, 5)
|   +-- reward_predictor.py      # GRU Phi for GRP
|   +-- train_grp_sl.py          # GRP supervised training on human logs
|
+-- training/                    # RL
|   +-- game_env.py              # multi-round wrapper + GRP attribution
|   +-- ppo_core.py              # PPO single-model
|   +-- vectorized_runner.py     # distributed self-play
|   +-- train_rl.py              # orchestrates RL for the 5 heads
|
+-- play/
    +-- game.py                  # local inference
    +-- sniffer_mahjong.py       # adapter for online play


================================================================================
6. INSTALLATION
================================================================================

Requires Python 3.10+ (tested on 3.14). Tested on Windows and Linux.

  git clone <this-repo>
  cd MahjongE
  pip install -e .

Dependencies: torch, numpy, zstandard, mahjong, huggingface_hub.


================================================================================
7. DATA PREPARATION
================================================================================

Download one or more annual shards from HuggingFace:

  python -m mahjong_ai.sl.download_mjai --year 2015
  python -m mahjong_ai.sl.download_mjai --year 2016

This produces data/tenhou/2015/*.mjai.json.zst (~150k files per year, each
containing 1 complete hanchan with all 4 players' perspectives).


================================================================================
8. TRAINING
================================================================================

8.1 Supervised Learning (5 heads)
---------------------------------

  python -m mahjong_ai.sl.train_all_sl \
      --data-dir data/tenhou/2015 \
      --output-dir checkpoints/sl_2015 \
      --cache-dir cache/sl_2015 \
      --max-files 1000 --epochs 2 --batch-size 128 --num-workers 0

Options:
  --heads discard riichi       train a subset
  --skip-existing              skip heads whose .pt already exists
  --channels 256 --n-blocks 16 architecture knobs

Caching: each head has its own samples_<head>.pkl in --cache-dir.
Re-running with the SAME arguments loads the cache (seconds instead of
minutes). Changing --max-files, --data-dir, or N_CHANNELS does NOT invalidate
the cache - delete cache/sl_2015/ manually to force re-parse.

8.2 Global Reward Predictor
---------------------------

  python -m mahjong_ai.agents.train_grp_sl \
      --data-dir data/tenhou/2015 \
      --output checkpoints/grp_sl.pt \
      --max-files 50000 --epochs 10

Then use --warm-start-predictor checkpoints/grp_sl.pt in RL training.

8.3 Reinforcement Learning (REWORKÉ)
-----------------------------------

Le pipeline RL a été entièrement refait (PPO on-policy mono-processus,
envs vectorisés). Ancien pipeline archivé dans training/legacy/.
Diagnostic complet : ANALYSE_RL.md.

  python -m mahjong_ai.rl.train \
      --sl-discard checkpoints/sl_tonpusen_light/discard.pt \
      --sl-frozen   checkpoints/sl_tonpusen_light \
      --grp-checkpoint checkpoints/grp_tonpusen.pt \
      --output checkpoints/rl_v3/rl_discard.pt \
      --total-steps 1000000 --n-envs 16 --batch-games 32

Défauts sains : 1 siège RL vs 3 SL figés (mean_rank < 2.50 = progrès),
guidage oracle 3 phases (gamma 1 -> 0), sans ancre KL ni bonus d'entropie.
Look-ahead : OFF par défaut (voir section 4 - training compatibility).
Tests : python test_rl_rework.py          (rework RL, ~45 s)
        python test_lookahead_fix.py      (look-ahead réparé, ~30 s)
        PYTHONPATH=. python test_env_rules.py   (règles de l'env, 68 tests)


================================================================================
9. INFERENCE / PLAY
================================================================================

  python -m mahjong_ai.play.game \
      --checkpoint checkpoints/sl_2015 \
      --opponents random


================================================================================
10. DEVIATIONS FROM THE PAPER
================================================================================

This is a reduced-scale, partial reproduction. Honest list of gaps:

  #   Paper                          This repo                    Impact
  --------------------------------------------------------------------------
  1   50 residual blocks, 21M params 16 blocks, 6.7M params      Intentional
  2   "100+ look-ahead features"     80 channels                  Reduced
  3   Look-ahead depth: up to 3      up to 4                      Comparable
  4   100M+ rounds dataset           150k files/year              Smaller
  5   Discards as full sequence      4-canaux count + last +      Slight loss
                                     tsumogiri                    of ordering
  6   Temporary furiten tracked      only permanent furiten       Negligible
  7   Oracle guiding, pMCPA          not yet implemented          Missing
  8   CNN without pooling            same                         -
  9   5 models + rule-based winner   same                         -
 10   GRP: 2-layer GRU               same                         -

Known bugs fixed:
  * multi_agent.py used to import a stale environment snapshot without Ron
    decision; RL now trains Ron properly.
  * game_env.py get_stats() / reset_stats() were missing, silently swallowing
    an AttributeError and logging zeros for all RL stats.


================================================================================
11. PERFORMANCE NOTES
================================================================================

Parsing speed
-------------

Parsing dominates when look-ahead is enabled:

  Config                            Files/s    1000 files
  -----------------------------------------------------------------
  Look-ahead OFF                    3-10       3-5 min
  Look-ahead ON (fixed 2026-11)     ~0.1-0.3   ~1-3 h
  Look-ahead ON, paper-faithful     0.01-0.02  15-20 h

Recommendation: keep compute_lookahead_flag=False in both mjai_parser
.build_obs and multi_agent while using existing SL checkpoints (they were
trained with these channels zeroed). Enable it only for a NEW SL training
run once the pipeline is validated - and then also pass
--compute-lookahead to RL training so the input distributions match.

Training speed
--------------

Per head, on a single mid-range GPU:

  Head                          1 epoch on 1000 files
  ---------------------------------------------------
  discard                       ~3-5 min
  riichi / chow / pong / kong   ~1-2 min

Memory
------

  * Model:                ~27 MB (float32)
  * Batch (256, 250, 34): ~8.5 MB
  * Dataset cache pickle: ~200 MB per head per 1000 files
  * LRU caches:           ~30 MB total


================================================================================
12. REFERENCES
================================================================================

  * Li, J., et al. Suphx: Mastering Mahjong with Deep Reinforcement Learning.
    arXiv:2003.13590, 2020.

  * Silver, D., et al. Mastering the game of Go without human knowledge.
    Nature 550, 2017.

  * nikketryhard/tenhou-houou-mjai dataset on HuggingFace.

  * Tenhou.net (https://tenhou.net/) - the platform used for evaluation.


================================================================================
LICENSE & ACKNOWLEDGEMENTS
================================================================================

This project is a research reproduction. It relies on the public dataset
nikketryhard/tenhou-houou-mjai. All credit for the original Suphx architecture
and training pipeline goes to the Microsoft Research Asia team.

================================================================================