from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from ndr_ml.features import FEATURE_NAMES, feature_matrix

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data/processed/ndr_flows.csv")
bundle = joblib.load(ROOT / "models/ndr_classifier_baseline.joblib")

model = bundle["model"]

if tuple(bundle["feature_names"]) != tuple(FEATURE_NAMES):
    raise ValueError("Feature schema mismatch")

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

# Используем те же тестовые строки, что и baseline.
from sklearn.model_selection import train_test_split

_, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.25,
    random_state=42,
    stratify=y,
)

X_test = X[test_idx].copy()
y_test = y[test_idx]

baseline_pred = model.predict(X_test)

# Изменяем порт назначения на альтернативные значения.
rng = np.random.default_rng(42)

shifted = X_test.copy()
port_idx = FEATURE_NAMES.index("dst_port")
known_idx = FEATURE_NAMES.index("is_well_known_port")

new_ports = rng.integers(
    low=1024,
    high=65536,
    size=len(shifted),
)

shifted[:, port_idx] = new_ports
shifted[:, known_idx] = np.isin(
    new_ports,
    [21, 22, 80, 443, 3389, 5432],
).astype(int)

shifted_pred = model.predict(shifted)

print("Baseline F1-macro:", f1_score(
    y_test, baseline_pred, average="macro"
))

print("Port-shift F1-macro:", f1_score(
    y_test, shifted_pred, average="macro"
))

print("\nClassification report after port shift:")
print(classification_report(
    y_test, shifted_pred, zero_division=0
))

print(
    "\nChanged predictions:",
    np.mean(baseline_pred != shifted_pred),
)
