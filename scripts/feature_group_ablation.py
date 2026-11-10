from pathlib import Path
import json
import time

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

DATA_PATH = ROOT / "data" / "processed" / "ndr_flows.csv"
OUTPUT_PATH = ROOT / "models" / "feature_group_ablation.json"

CLASSES = ["Normal", "PortScan", "DDoS", "BruteForce"]

# Группы признаков
PORT_FEATURES = {
    "dst_port",
    "is_well_known_port",
}

TCP_FEATURES = {
    "syn_count",
    "ack_count",
    "rst_count",
    "syn_ratio",
    "ack_ratio",
    "rst_ratio",
}

TRAFFIC_FEATURES = {
    "duration_ms",
    "total_packets",
    "total_bytes",
    "bytes_per_packet",
    "packets_per_sec",
    "bytes_per_sec",
}

experiments = {
    "all_features": set(FEATURE_NAMES),
    "without_ports": set(FEATURE_NAMES) - PORT_FEATURES,
    "without_ports_and_tcp": TRAFFIC_FEATURES,
    "tcp_only": TCP_FEATURES,
    "traffic_and_tcp": TRAFFIC_FEATURES | TCP_FEATURES,
    "ports_only": PORT_FEATURES,
}

print("Загрузка датасета...")
df = pd.read_csv(DATA_PATH)

flows = [
    {
        "dst_port": int(row.dst_port),
        "duration_ms": int(row.duration_ms),
        "total_packets": int(row.total_packets),
        "total_bytes": int(row.total_bytes),
        "tcp_flags": {
            "syn": int(row.syn),
            "ack": int(row.ack),
            "rst": int(row.rst),
        },
    }
    for row in df.itertuples(index=False)
]

X = feature_matrix(flows)
y = df["label"].to_numpy()

train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.25,
    stratify=y,
    random_state=42,
)

results = {}

for experiment_name, selected_features in experiments.items():

    indices = [
        i
        for i, name in enumerate(FEATURE_NAMES)
        if name in selected_features
    ]

    feature_names = [FEATURE_NAMES[i] for i in indices]

    print(f"\n{'=' * 60}")
    print(f"Эксперимент: {experiment_name}")
    print(f"Признаков: {len(indices)}")
    print(f"Используем: {feature_names}")

    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        max_depth=22,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    start = time.perf_counter()

    model.fit(
        X[train_idx][:, indices],
        y[train_idx],
    )

    predictions = model.predict(
        X[test_idx][:, indices]
    )

    elapsed = time.perf_counter() - start

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

    normal_fpr = (
        cm[0, 1:].sum() / cm[0].sum()
    )

    attack_missed = int(cm[1:, 0].sum())
    attack_total = int(cm[1:, :].sum())

    results[experiment_name] = {
        "features": feature_names,
        "f1_macro": float(
            f1_score(
                y[test_idx],
                predictions,
                average="macro",
            )
        ),
        "normal_fpr": float(normal_fpr),
        "attack_miss_rate": (
            attack_missed / attack_total
        ),
        "recall": {
            name: report[name]["recall"]
            for name in CLASSES
        },
        "precision": {
            name: report[name]["precision"]
            for name in CLASSES
        },
        "confusion_matrix": cm.tolist(),
        "training_seconds": round(elapsed, 2),
    }

    print(
        f"F1-macro: "
        f"{results[experiment_name]['f1_macro']:.4f}"
    )

    print(
        f"Normal FPR: "
        f"{results[experiment_name]['normal_fpr']:.2%}"
    )

    print(
        f"Attack miss rate: "
        f"{results[experiment_name]['attack_miss_rate']:.2%}"
    )

    for name in CLASSES:
        recall = results[experiment_name]["recall"][name]
        print(f"Recall {name}: {recall:.2%}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ")

summary = pd.DataFrame({
    name: {
        "F1-macro": item["f1_macro"],
        "Normal FPR": item["normal_fpr"],
        "Attack miss rate": item["attack_miss_rate"],
        "BruteForce Recall": item["recall"]["BruteForce"],
        "DDoS Recall": item["recall"]["DDoS"],
    }
    for name, item in results.items()
}).T

print(summary.round(4).to_string())

print(f"\nРезультаты сохранены: {OUTPUT_PATH}")
