from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "ndr_flows.csv"

df = pd.read_csv(
    DATA_PATH,
    usecols=["dst_port", "label"],
)

for label, group in df.groupby("label"):
    print(f"\n{'=' * 50}")
    print(f"Класс: {label}")
    print(f"Количество потоков: {len(group):,}")

    counts = group["dst_port"].value_counts()
    percentages = counts / len(group) * 100

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages,
    }).head(15)

    print("\nТоп-15 портов:")
    print(result.to_string(float_format=lambda x: f"{x:.2f}"))

    print(
        f"\nУникальных портов: "
        f"{group['dst_port'].nunique():,}"
    )

    print(
        f"Доля топ-5 портов: "
        f"{percentages.head(5).sum():.2f}%"
    )
