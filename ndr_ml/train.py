"""Train on LABELLED sensor-compatible flow rows, not fabricated attack labels."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from .features import FEATURE_NAMES, feature_matrix
from .api import CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV with flow fields, label and optional capture_id')
    parser.add_argument('--out', default='models/ndr_classifier_v1.joblib')
    parser.add_argument('--metrics', default='models/metrics.json')
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    required = {'label','dst_port','duration_ms','total_packets','total_bytes',
                'syn','ack','rst'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    if set(df.label) != set(CLASSES):
        raise ValueError(f'Training requires all four classes, found: {sorted(set(df.label))}')
    flows = [dict(dst_port=int(row.dst_port), duration_ms=int(row.duration_ms),
                  total_packets=int(row.total_packets), total_bytes=int(row.total_bytes),
                  tcp_flags=dict(syn=int(row.syn),ack=int(row.ack),rst=int(row.rst)))
             for row in df.itertuples()]
    X = feature_matrix(flows)
    y = df.label.to_numpy()
    print('WARNING: baseline only; random-row split is not independent-capture validation.')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
    if set(y_train) != set(CLASSES) or set(y_test) != set(CLASSES):
        raise ValueError('Train/test split must contain all 4 classes. Collect more independent captures.')
    model = RandomForestClassifier(n_estimators=120,min_samples_leaf=2,max_depth=22,
                                    class_weight='balanced_subsample',n_jobs=-1,random_state=42)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    metrics = dict(validation_type='random_row_baseline_only',f1_macro=float(f1_score(y_test,pred,average='macro')),
                   classes=list(CLASSES),
                   confusion_matrix=confusion_matrix(y_test,pred,labels=CLASSES).tolist(),
                   report=classification_report(y_test,pred,labels=CLASSES,output_dict=True,zero_division=0),
                   train_rows=len(y_train),test_rows=len(y_test),
                   important_features=sorted(zip(FEATURE_NAMES,map(float,model.feature_importances_)),key=lambda t:-t[1]))
    out = Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=model,feature_names=FEATURE_NAMES,classes=CLASSES),out)
    metrics_path = Path(args.metrics); metrics_path.parent.mkdir(parents=True,exist_ok=True)
    metrics_path.write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'f1_macro':metrics['f1_macro'],'train_rows':len(y_train),'test_rows':len(y_test)},indent=2))

if __name__ == '__main__':
    main()
