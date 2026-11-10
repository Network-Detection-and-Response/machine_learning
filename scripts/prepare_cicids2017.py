"""Convert CICIDS2017 CSVs to an exploratory baseline using sensor-observable fields.

CICFlowMeter byte lengths may differ from sensor wire-byte counts.
Protocol and source port are unavailable and are never fabricated.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = ['Destination Port','Flow Duration','Total Fwd Packets','Total Backward Packets',
         'Total Length of Fwd Packets','Total Length of Bwd Packets',
         'SYN Flag Count','ACK Flag Count','RST Flag Count','Label']
OUTPUT = ['dst_port','duration_ms','total_packets','total_bytes','syn','ack','rst','label']
CLASSES = ('Normal','PortScan','DDoS','BruteForce')
ROOT = Path(__file__).resolve().parent.parent

def label_map(value):
    v = str(value).strip().lower()
    if v == 'benign': return 'Normal'
    if v == 'portscan': return 'PortScan'
    if v in {'ddos','dos hulk','dos goldeneye','dos slowloris','dos slowhttptest'}: return 'DDoS'
    if v in {'ftp-patator','ssh-patator'} or ('web attack' in v and 'brute force' in v): return 'BruteForce'
    return None

def convert(chunk):
    chunk.columns = chunk.columns.str.strip()
    missing = set(INPUT) - set(chunk.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    nums = chunk[INPUT[:-1]].apply(pd.to_numeric, errors='coerce')
    d = nums['Flow Duration'].to_numpy(dtype=float)
    p = (nums['Total Fwd Packets']+nums['Total Backward Packets']).to_numpy(dtype=float)
    b = (nums['Total Length of Fwd Packets']+nums['Total Length of Bwd Packets']).to_numpy(dtype=float)
    syn = nums['SYN Flag Count'].to_numpy(dtype=float)
    ack = nums['ACK Flag Count'].to_numpy(dtype=float)
    rst = nums['RST Flag Count'].to_numpy(dtype=float)
    port = nums['Destination Port'].to_numpy(dtype=float)
    labels = chunk['Label'].map(label_map).to_numpy()
    values = np.column_stack((d,p,b,syn,ack,rst,port))
    valid = (pd.notna(labels) & np.isfinite(values).all(axis=1) &
             (d > 0) & (p > 0) & (b >= 0) &
             (syn >= 0) & (ack >= 0) & (rst >= 0) &
             (syn <= p) & (ack <= p) & (rst <= p) &
             (port >= 0) & (port <= 65535) &
             (p < 2**53) & (b < 2**53))
    # CIC duration is microseconds; ceil to >=1 ms avoids zero and rate explosions.
    duration = np.maximum(1, np.ceil(np.maximum(d,0)/1000))
    out = pd.DataFrame({
        'dst_port': port[valid].astype('int64'),
        'duration_ms': duration[valid].astype('int64'),
        'total_packets': p[valid].astype('int64'),
        'total_bytes': b[valid].astype('int64'),
        'syn': syn[valid].astype('int64'),
        'ack': ack[valid].astype('int64'),
        'rst': rst[valid].astype('int64'),
        'label': labels[valid],
    })
    return out[OUTPUT]

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--raw',  default=str(Path(__file__).resolve().parent.parent / 'data' / 'raw'))
    a.add_argument('--out',  default=str(ROOT / "data" / "processed" / "ndr_flows.csv"))
    a.add_argument('--max-per-class', type=int, default=40000)
    args = a.parse_args()
    if args.max_per_class < 1: raise ValueError('max-per-class must be > 0')
    paths = sorted(Path(args.raw).rglob('*.csv'))
    if not paths: raise FileNotFoundError(f'No CSVs under {args.raw}')
    rng = np.random.default_rng(42)
    # Uniform bounded sample using random row priorities (not first-N bias).
    kept = {c: pd.DataFrame(columns=OUTPUT+['_priority']) for c in CLASSES}
    counts = {c: 0 for c in CLASSES}
    discarded = 0
    for path in paths:
        print(f'Reading {path.name}', flush=True)
        for chunk in pd.read_csv(path, chunksize=100_000, encoding='utf-8-sig',
                                 usecols=lambda x: x.strip() in INPUT, low_memory=False):
            cleaned = convert(chunk)
            discarded += len(chunk)-len(cleaned)
            for c, part in cleaned.groupby('label'):
                counts[c] += len(part)
                part = part.copy()
                part['_priority'] = rng.random(len(part))
                kept[c] = (part if kept[c].empty else pd.concat((kept[c], part), ignore_index=True)).nsmallest(
                    args.max_per_class, '_priority')
    if any(counts[c] == 0 for c in CLASSES):
        raise ValueError(f'At least one required class absent: {counts}')
    output = pd.concat((kept[c].drop(columns='_priority') for c in CLASSES),ignore_index=True)
    for c in OUTPUT[:-1]: output[c] = output[c].astype('int64')
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.sample(frac=1,random_state=42).to_csv(path,index=False)
    print(f'Saved {len(output):,} rows to {path}')
    print('Available valid rows:', counts)
    print(f'Excluded/rejected: {discarded:,}')
    print('WARNING: baseline only; possible byte-semantic shift and capture leakage.')

if __name__ == '__main__': main()
