# Script à sauvegarder sous mahjong_ai/diag_class_balance.py
import pickle
from collections import Counter

with open('cache/sl_2015/samples_all.pkl', 'rb') as f:
    data = pickle.load(f)

# atype : 0=discard, 4=riichi, 2=chow, 1=pong, 3=kong
names = {0: 'discard', 4: 'riichi', 2: 'chow', 1: 'pong', 3: 'kong'}

for atype, name in names.items():
    sub = [s for s in data if s[1] == atype]
    if not sub:
        continue
    actions = [s[2] for s in sub]
    counts = Counter(actions)
    n = len(actions)

    if name == 'discard':
        # multiclasse : baseline = classe la plus frequente
        majority = counts.most_common(1)[0][1]
        print(f"{name:8s} n={n:7d}  majority_class={majority/n:.3f}  "
              f"(top3={counts.most_common(3)})")
    else:
        # binaire : baseline = max(positifs, negatifs)
        pos = counts.get(1, 0)
        baseline = max(pos, n - pos) / n
        print(f"{name:8s} n={n:7d}  pos={pos:6d} ({pos/n*100:5.1f}%)  "
              f"baseline={baseline:.3f}")