import glob
import os
import sys

import soundfile as sf

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
files = glob.glob(os.path.join(src, "**", "*.flac"), recursive=True)
for i, path in enumerate(files, 1):
    data, sr = sf.read(path)
    name = os.path.splitext(os.path.basename(path))[0] + ".wav"
    sf.write(os.path.join(dst, name), data, sr, subtype="PCM_16")
    print(f"{i}/{len(files)}  {name}")
print("Готово.")