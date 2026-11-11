# debug_parse.py
import traceback
import glob
import json
import zstandard as zstd

# 1. Le fichier est-il lisible ?
files = glob.glob("data/tenhou_tonpusen/*.mjai.json.zst")
print(f"{len(files)} fichiers trouvés")

path = files[0]
dctx = zstd.ZstdDecompressor()
with open(path, "rb") as fh:
    with dctx.stream_reader(fh) as reader:
        raw = reader.read()

text = raw.decode("utf-8")
lines = text.splitlines()
print(f"\n1. Decompression OK : {len(text)} chars, {len(lines)} lignes")

# 2. Les evenements sont-ils parsables ?
events = []
for line in lines[:5]:
    print(f"   ligne: {line[:100]}")
for line in lines:
    line = line.strip()
    if line:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"   JSON ERROR ligne: {e}")

print(f"\n2. {len(events)} evenements parsés")

# 3. Types d'evenements presents
types = {}
for ev in events:
    t = ev.get("type", "?")
    types[t] = types.get(t, 0) + 1
print(f"\n3. Types d'evenements : {types}")

# 4. Le start_game contient-il kyoku_first ?
start = next((e for e in events if e.get("type") == "start_game"), None)
print(f"\n4. start_game : {start}")

# 5. Detection du format
from mahjong_ai.sl.mjai_parser import _detect_format
fmt = _detect_format(events)
print(f"\n5. Format detecte : {fmt}")

# 6. Extraction des samples (sans try/except)
from mahjong_ai.sl.mjai_parser import events_to_samples_all_povs
try:
    samples, fmt = events_to_samples_all_povs(events)
    print(f"\n6. {len(samples)} samples extraits, format={fmt}")
except Exception:
    print(f"\n6. EXCEPTION dans events_to_samples_all_povs :")
    traceback.print_exc()