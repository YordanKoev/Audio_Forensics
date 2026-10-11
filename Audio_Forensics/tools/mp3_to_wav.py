import argparse
import glob
import os
import sys
 
import soundfile as sf
 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=os.path.join(ROOT, "dataset", "real"))
    ap.add_argument("--delete", action="store_true", help="изтрий MP3 файла след успешно преобразуване")
    args = ap.parse_args()
 
    files = sorted(f for f in glob.glob(os.path.join(args.folder, "**", "*"), recursive=True)
                   if f.lower().endswith(".mp3"))
    print(f"Намерени {len(files)} MP3 файла в {args.folder}")
 
    done = skipped = failed = 0
    for i, path in enumerate(files, 1):
        out = os.path.splitext(path)[0] + ".wav"
        if os.path.exists(out):
            skipped += 1
            continue
        try:
            data, sr = sf.read(path)
            sf.write(out, data.clip(-1, 1), sr, subtype="PCM_16")
        except Exception as exc:
            failed += 1
            print(f"  грешка при {os.path.basename(path)}: {exc}")
            continue
        if args.delete:
            os.remove(path)
        done += 1
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")
    print(f"Готово: {done} преобразувани, {skipped} пропуснати (вече има WAV), {failed} с грешка.")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())