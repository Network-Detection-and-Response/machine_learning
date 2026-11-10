from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

files = sorted(DATA_DIR.rglob("*.csv"))

if not files:
    raise FileNotFoundError(
        "CSV-файлы не найдены. проверь папку data/raw"
    )

print(f"найдено CSV-файлов: {len(files)}")

for path in files:
    print(f"\nфайл: {path.name}")
    print(f"размер: {path.stat().st_size / 1024**2:.2f} МБ")

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        columns = [name.strip() for name in next(reader)]

    print(f"количество признаков: {len(columns)}")
    print("названия столбцов:")
    for i, name in enumerate(columns, start=1):
        print(f"  {i}. {name}")
