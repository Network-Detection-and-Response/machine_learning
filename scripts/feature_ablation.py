from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from ndr_ml.features import FEATURE_NAMES, feature_matrix

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "processed" / "ndr_flows.csv"
RESULT_PATH = ROOT / "models" / "ablation_results.json"

CLASSES = ["Normal", "PortScan", "DDoS", "BruteForce"]

df = pd.read_csv(CSV_PATH)

flows = [
    {
        "dst_port": int(r.dst_port),
        "duration_ms": int(r.duration_ms),
        "total_packets": int(r.total_packets),
        "total_bytes": int(r.total_bytes),
        "tcp_flags": {
            "syn": int(r.syn),
            "ack": int(r.ack),
            "rst": int(r.rst),
        },
    }
    for r in df.itertuples(index=False)
]

X = feature_matrix(flows)
y = df["label"].to_numpy()

train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.25,
    stratify=y,
    random_state=42,
)

experiments = {
    "all_features": list(range(len(FEATURE_NAMES))),
    "without_ports": [
        i for i, name in enumerate(FEATURE_NAMES)
        if name not in {"dst_port", "is_well_known_port"}
    ],
}

results = {}

for experiment, indices in experiments.items():
    print(f"\nОбучение: {experiment}")

    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        max_depth=22,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X[train_idx][:, indices], y[train_idx])
    predictions = model.predict(X[test_idx][:, indices])

    report = classification_report(
        y[test_idx],
        predictions,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y[test_idx],
        predictions,
        labels=CLASSES,
    )

    normal_false_positives = int(cm[0, 1:].sum())
    normal_total = int(cm[0].sum())

    results[experiment] = {
        "f1_macro": float(
            f1_score(y[test_idx], predictions, average="macro")
        ),
        "false_positive_rate": (
            normal_false_positives / normal_total
        ),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
            }
            for name in CLASSES
        },
        "confusion_matrix": cm.tolist(),
    }

    print(
        f"F1-macro: {results[experiment]['f1_macro']:.6f}"
    )
    print(
        f"Normal FPR: "
        f"{results[experiment]['false_positive_rate']:.4%}"
    )

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nРезультаты сохранены: {RESULT_PATH}")
