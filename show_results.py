import json
from collections import Counter

with open("program.z.local.json") as f:
    data = json.load(f)

shots = []
for result in data["results"]:
    bits = dict(result)["bits"]
    shots.append("".join(map(str, bits)))

print("First 10 measurements:")
for shot in shots[:10]:
    print(shot)

print("\nMeasurement counts:")
for state, count in Counter(shots).items():
    print(f"{state}: {count}")