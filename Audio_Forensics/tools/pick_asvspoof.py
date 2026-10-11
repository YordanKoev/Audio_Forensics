import os
import random
import shutil
import sys

root = sys.argv[1]
n = int(sys.argv[2])
proto = os.path.join(root, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
flac_dir = os.path.join(root, "ASVspoof2019_LA_train", "flac")

rows = [line.split() for line in open(proto)]
real = [r[1] for r in rows if r[-1] == "bonafide"]
fake_by_attack = {}
for r in rows:
    if r[-1] == "spoof":
        fake_by_attack.setdefault(r[3], []).append(r[1])

random.seed(42)
pick_real = random.sample(real, n)
per_attack = n // len(fake_by_attack)
pick_fake = [f for ids in fake_by_attack.values() for f in random.sample(ids, per_attack)]

for name, ids in (("real_flac", pick_real), ("fake_flac", pick_fake)):
    os.makedirs(name, exist_ok=True)
    for i in ids:
        shutil.copy(os.path.join(flac_dir, i + ".flac"), name)
    print(f"{name}: {len(ids)} файла")