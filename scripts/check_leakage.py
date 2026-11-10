
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ndr_ml.features import FEATURE_NAMES, feature_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "ndr_flows.csv"

df = pd.read_csv(DATA_PATH)

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

features_df = pd.DataFrame(X, columns=FEATURE_NAMES)
features_df["label"] = y

print("Всего потоков:", len(features_df))

duplicate_features = features_df.duplicated(
    subset=list(FEATURE_NAMES),
    keep=False,
)

print(
    "Строк с повторяющимися признаками:",
    int(duplicate_features.sum()),
)

conflicts = features_df.groupby(
    list(FEATURE_NAMES)
)["label"].nunique()

print(
    "Векторов признаков с разными метками:",
    int((conflicts > 1).sum()),
)

train_vectors = pd.MultiIndex.from_frame(
    features_df.iloc[train_idx][list(FEATURE_NAMES)]
)

test_vectors = pd.MultiIndex.from_frame(
    features_df.iloc[test_idx][list(FEATURE_NAMES)]
)

overlap = test_vectors.isin(train_vectors)

print(
    "Тестовых строк с идентичным вектором в train:",
    int(overlap.sum()),
)

print(
    "Доля пересечений:",
    f"{overlap.mean():.2%}",
)
