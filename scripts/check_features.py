from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

columns = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "Label",
]

total = 0
invalid = 0
zero_duration = 0
negative_values = 0

for file in sorted(DATA_DIR.glob("*.csv")):
    print(f"\nПроверка: {file.name}")

    reader = pd.read_csv(
        file,
        usecols=lambda c: c.strip() in columns,
        chunksize=100_000,
        encoding="utf-8-sig",
        low_memory=False,
    )

    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()

        numeric = chunk.drop(columns=["Label"])
        numeric = numeric.apply(
            pd.to_numeric, errors="coerce"
        )

        values = numeric.to_numpy(dtype=np.float64)

        total += len(chunk)

        invalid += int(
            (~np.isfinite(values)).any(axis=1).sum()
        )

        zero_duration += int(
            (numeric["Flow Duration"] <= 0).sum()
        )

        negative_values += int(
            (values < 0).any(axis=1).sum()
        )

print("\n=== ИТОГ ===")
print(f"Всего строк: {total:,}")
print(f"Строк с NaN/Inf: {invalid:,}")
print(f"Длительность <= 0: {zero_duration:,}")
print(f"Строк с отрицательными значениями: {negative_values:,}")
