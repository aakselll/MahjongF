# debug_shards.py
import pickle, glob

shards = sorted(glob.glob("cache/sl_tonpusen/_shards/shard_*.pkl"))
print(f"{len(shards)} shards")

bad = []
for sf in shards:
    try:
        with open(sf, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"CORROMPU : {sf} -> {type(e).__name__}: {e}")
        bad.append(sf)

print(f"\n{len(bad)} shards corrompus sur {len(shards)}")

# Verifier aussi les .tmp residues
tmps = glob.glob("cache/sl_tonpusen/_shards/*.tmp")
print(f"{len(tmps)} fichiers .tmp residuels")
for t in tmps:
    print(f"  {t}")