"""Single source of truth for online and offline flow features."""
import numpy as np

FEATURE_NAMES = (
    'duration_ms', 'total_packets', 'total_bytes', 'dst_port',
    'syn_count', 'ack_count',
    'rst_count', 'bytes_per_packet', 'packets_per_sec', 'bytes_per_sec',
    'syn_ratio', 'ack_ratio', 'rst_ratio', 'is_well_known_port',
)
WELL_KNOWN = frozenset({21, 22, 80, 443, 3389, 5432})


def extract_features(flow: dict) -> list[float]:
    flags = flow.get('tcp_flags') or {}
    p = int(flow['total_packets'])
    b = int(flow['total_bytes'])
    d = int(flow['duration_ms'])
    if min(p, b, d) < 0:
        raise ValueError('negative duration/packet/byte count')
    if p == 0 and b > 0:
        raise ValueError('bytes without packets')
    syn, ack, rst = (int(flags.get(k, 0)) for k in ('syn', 'ack', 'rst'))
    if min(syn, ack, rst) < 0 or max(syn, ack, rst) > p:
        raise ValueError('invalid TCP flag counts')
    dst = int(flow['dst_port'])
    if not (0 <= dst <= 65535):
        raise ValueError('invalid port')
    result = [d, p, b, dst, syn, ack, rst, b / max(p, 1),
              p * 1000 / max(d, 1), b * 1000 / max(d, 1),
              syn / max(p, 1), ack / max(p, 1), rst / max(p, 1),
              int(dst in WELL_KNOWN)]
    if not np.isfinite(result).all():
        raise ValueError('non-finite feature')
    return result


def feature_matrix(flows: list[dict]) -> np.ndarray:
    return np.asarray([extract_features(f) for f in flows], dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
