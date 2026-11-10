"""Go-backend compatible, fail-closed ML inference API."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
import ipaddress
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from .features import FEATURE_NAMES, feature_matrix

CLASSES = ('Normal', 'PortScan', 'DDoS', 'BruteForce')

class TCPFlags(BaseModel):
    syn: int = Field(default=0, ge=0)
    ack: int = Field(default=0, ge=0)
    rst: int = Field(default=0, ge=0)

class Flow(BaseModel):
    flow_id: str
    src_ip: str
    dst_ip: str
    src_mac: str | None = None
    dst_mac: str | None = None
    src_port: int = Field(ge=0, le=65535)
    dst_port: int = Field(ge=0, le=65535)
    protocol: str
    duration_ms: int = Field(ge=0)
    total_packets: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    tcp_flags: TCPFlags = Field(default_factory=TCPFlags)

    @field_validator('flow_id')
    @classmethod
    def valid_id(cls, value: str) -> str:
        from uuid import UUID
        return str(UUID(value))

    @field_validator('src_ip', 'dst_ip')
    @classmethod
    def valid_ip(cls, value: str) -> str:
        return str(ipaddress.ip_address(value))

    @field_validator('protocol')
    @classmethod
    def valid_protocol(cls, value: str) -> str:
        proto = value.upper()
        if proto not in {'TCP', 'UDP', 'ICMP'}:
            raise ValueError('protocol must be TCP, UDP or ICMP')
        return proto

    @model_validator(mode='after')
    def valid_flags(self):
        if self.total_packets == 0 and self.total_bytes:
            raise ValueError('bytes without packets')
        flags = self.tcp_flags
        if max(flags.syn, flags.ack, flags.rst) > self.total_packets:
            raise ValueError('TCP flag count exceeds packets')
        if self.protocol != 'TCP' and any((flags.syn, flags.ack, flags.rst)):
            raise ValueError('non-TCP flow has TCP flags')
        return self

class PredictRequest(BaseModel):
    flows: list[Flow] = Field(min_length=1, max_length=500)

class Prediction(BaseModel):
    flow_id: str
    probabilities: dict[str, float]
    predicted_class: str
    ml_confidence: float
    anomaly_indicators: list[str]

class PredictResponse(BaseModel):
    predictions: list[Prediction]


def explain(flow: Flow) -> list[str]:
    p = max(flow.total_packets, 1)
    f = flow.tcp_flags
    factors = []
    if f.syn / p > 0.7 and f.ack / p < 0.1 and flow.protocol == 'TCP':
        factors.append('high_syn_low_ack_ratio')
    if f.rst / p > 0.3 and flow.protocol == 'TCP':
        factors.append('high_rst_ratio')
    if flow.total_packets and flow.total_bytes / p < 60:
        factors.append('small_packets')
    if flow.total_packets > 100 and flow.duration_ms > 0 and flow.total_packets * 1000 / flow.duration_ms > 1000:
        factors.append('high_packets_per_second')
    return factors


def load_model(path: Path):
    if not path.is_file():
        return None
    bundle = joblib.load(path)  # Load ONLY locally trained, trusted artifacts.
    if tuple(bundle.get('feature_names', [])) != FEATURE_NAMES:
        raise RuntimeError('Feature schema mismatch')
    model = bundle['model']
    if set(model.classes_) != set(CLASSES):
        raise RuntimeError('Model class schema mismatch')
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(Path(os.getenv('MODEL_PATH', 'models/ndr_classifier_v1.joblib')))
    yield

app = FastAPI(title='NDR ML Detector', version='1.0.0', lifespan=lifespan)

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': app.state.model is not None}

@app.get('/ready')
def ready():
    if app.state.model is None:
        raise HTTPException(503, 'Model not trained: run python -m ndr_ml.train')
    return {'status': 'ready'}

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    model = app.state.model
    if model is None:
        raise HTTPException(503, 'Model not trained')
    flows = request.flows
    matrix = feature_matrix([f.model_dump() for f in flows])
    raw = model.predict_proba(matrix)
    if raw.shape != (len(flows), len(CLASSES)) or not np.isfinite(raw).all():
        raise HTTPException(503, 'Invalid model output')
    order = {name: idx for idx, name in enumerate(model.classes_)}
    predictions = []
    for flow, values in zip(flows, raw):
        probs = {name: float(values[order[name]]) for name in CLASSES}
        if any(not 0 <= v <= 1 for v in probs.values()) or abs(sum(probs.values()) - 1) > 1e-6:
            raise HTTPException(503, 'Invalid probability distribution')
        top = max(probs, key=probs.get)
        predictions.append(Prediction(flow_id=flow.flow_id, probabilities=probs,
                          predicted_class=top, ml_confidence=1 - probs['Normal'],
                          anomaly_indicators=explain(flow)))
    return PredictResponse(predictions=predictions)

@app.post('/predict_batch', response_model=PredictResponse)
def predict_batch(request: PredictRequest):
    return predict(request)
