from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

files = sorted(DATA_DIR.rglob("*.csv"))
total = Counter()

if not files:
    raise FileNotFoundError("CSV не найдены в data/raw")

for path in files:
    counts = Counter()

    for chunk in pd.read_csv(
        path,
        usecols=lambda col: col.strip() == "Label",
        chunksize=100_000,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()

        labels = chunk["Label"].astype("string").str.strip()
        counts.update(labels.value_counts().to_dict())

    total.update(counts)

    print(f"\n{path.name}")
    for label, count in counts.most_common():
        print(f"  {label}: {count:,}")

print("\n=== ОБЩЕЕ РАСПРЕДЕЛЕНИЕ ===")
for label, count in total.most_common():
    print(f"{label}: {count:,}")

print(f"\nВсего потоков: {sum(total.values()):,}")
