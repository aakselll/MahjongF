import traceback
from mahjong_ai.sl.mjai_parser import parse_mjai_zst_file, events_to_samples

path = "data/tenhou/2015/tenhou-houou-2015/2015010100gm-00a9-0000-028b31cb.mjai.json.zst"

evs = parse_mjai_zst_file(path)
print(f"Events parsés : {len(evs)}")
if evs:
    print(f"1er event    : {evs[0].get('type')}")
    from collections import Counter
    print(f"Types        : {Counter(e.get('type') for e in evs).most_common(10)}")

for pov in range(4):
    try:
        s = events_to_samples(evs, pov=pov)
        print(f"pov={pov} : {len(s)} samples")
        if s:
            print(f"           obs.shape = {s[0][0].shape}")

        # Dans le même test, ajoute ça après avoir récupéré `s` (n'importe quel pov)
        import numpy as np
        from mahjong_ai.envs.suphx_features import _OFFSET

        obs = s[0][0]

        # 1. Vérifier que les canaux attendus sont non-nuls
        la_off = _OFFSET['lookahead']
        la = obs[la_off:la_off+120]
        print(f"look-ahead sum = {la.sum():.3f}  max = {la.max():.3f}")

        own_hand = obs[_OFFSET['own_hand']:_OFFSET['own_hand']+4].sum()
        print(f"own_hand sum = {own_hand}")   # doit être ~13-14

        dora_off = _OFFSET['dora_ind']
        print(f"dora_ind sum = {obs[dora_off:dora_off+1].sum()}")   # ≥ 1

        # 2. Vérifier que le look-ahead n'est PAS toujours nul
        nonzero_la = sum(1 for o,_,_,_ in s if o[la_off:la_off+120].sum() > 0)
        print(f"{nonzero_la}/{len(s)} samples avec look-ahead non nul")

        la_off = _OFFSET['lookahead']
        la_sums = [o[la_off:la_off+120].sum() for o,_,_,_ in s]
        print(f"LA sums: mean={np.mean(la_sums):.3f}  "
            f"p25={np.percentile(la_sums,25):.3f}  "
            f"median={np.median(la_sums):.3f}  "
            f"p75={np.percentile(la_sums,75):.3f}  "
            f"max={np.max(la_sums):.3f}  "
            f"nonzero={sum(1 for x in la_sums if x>0)}/{len(la_sums)}")

    except Exception as e:
        print(f"pov={pov} : EXCEPTION {typea(e).__name__}: {e}")
        traceback.print_exc()

