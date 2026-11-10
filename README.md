# NDR ML Detector

Python/FastAPI multiclass classifier for `Network-Detection-and-Response/backend`.

## Contract (checked against the actual Go client)

`POST /predict` accepts `{ "flows": [ {"flow_id":"UUID", "src_ip":"...", "dst_ip":"...", "src_port":1, "dst_port":443,"protocol":"TCP","duration_ms":1500,"total_packets":45,"total_bytes":35000,"tcp_flags":{"syn":1,"ack":40,"rst":0}} ] }`.

Response is `{ "predictions": [ { "flow_id": "UUID", "probabilities": {"Normal":0.1,"PortScan":0.7,"DDoS":0.1,"BruteForce":0.1}, "predicted_class":"PortScan", "ml_confidence":0.9, "anomaly_indicators":[] } ] }`.

Go requires exactly one prediction per UUID and probabilities summing to ~1. `ml_confidence` is **total attack-class probability** `1 - P(Normal)`, not the probability of `predicted_class`. Explanations are heuristic indicators, not model feature attribution.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

## Training (required before inference)

Supply `data/flows.csv` (not included), with columns `label,src_port,dst_port,protocol,duration_ms,total_packets,total_bytes,syn,ack,rst` and preferably `capture_id` for capture-wise train/test separation. Labels must be `Normal`, `PortScan`, `DDoS`, `BruteForce`. All columns must correspond to *actual sensor measurements*. **Do not map CICIDS columns blindly:** capture time, directional counters, and flag semantics must match sensor implementation. BruteForce cannot be identified reliably from a single NetFlow record alone; temporal aggregates may be needed. No arbitrary metrics or trained weights are included.

```bash
python -m ndr_ml.train --csv data/flows.csv
uvicorn ndr_ml.api:app --host 0.0.0.0 --port 8000
```

`GET /health` signals the process; `GET /ready` returns 503 until a model has been trained and loaded. `POST /predict` also returns 503 without a model. Swagger: `/docs`.

## Docker

```bash
docker build -t ndr-ml .
docker run --rm -p 8000:8000 ndr-ml
```

Train first and ensure the generated model artifact is available at image build time, or mount an explicit trusted `MODEL_PATH`. The Go backend Docker Compose currently uses `ML_SERVICE_URL=http://host.docker.internal:8000`, which may need replacing with `http://ml-service:8000` when deployed together.

## Known limitations

- Random Forest probabilistic output is **not automatically calibrated**. Calibrate with an independent validation split, and report multiclass log loss/Brier score before describing scores as probabilities of real incidents.
- Row-wise split is allowed for exploration only; `capture_id` groups are preferred to prevent cross-session/capture leakage.
- PortScan/DDoS detection depends on observation window and sensor aggregation; one flow is not sufficient for all threat types.
- Real deployment requires load testing, telemetry drift monitoring and an out-of-distribution policy. This is a functional starter, not a verified security product.
