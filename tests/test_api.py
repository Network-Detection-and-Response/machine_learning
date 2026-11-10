from uuid import uuid4
from fastapi.testclient import TestClient
from ndr_ml.api import app, CLASSES
from ndr_ml.features import extract_features, FEATURE_NAMES

FLOW = dict(flow_id=str(uuid4()),src_ip='192.168.1.50',dst_ip='10.0.0.5',
            src_port=54321,dst_port=443,protocol='TCP',duration_ms=1500,
            total_packets=45,total_bytes=35000,tcp_flags=dict(syn=1,ack=40,rst=0))


def test_features():
    x = extract_features(FLOW)
    assert len(x) == len(FEATURE_NAMES)
    assert x[7] == 35000/45


def test_untrained_returns_503(monkeypatch):
    monkeypatch.setenv('MODEL_PATH','/nonexistent/model.joblib')
    with TestClient(app) as client:
        assert client.get('/health').json()['model_loaded'] is False
        assert client.post('/predict',json={'flows':[FLOW]}).status_code == 503


def test_schema_validation(monkeypatch):
    monkeypatch.setenv('MODEL_PATH','/nonexistent/model.joblib')
    with TestClient(app) as client:
        invalid = {**FLOW,'total_packets':1}
        assert client.post('/predict',json={'flows':[invalid]}).status_code == 422


def test_predict_with_model(monkeypatch):
    import numpy as np
    class Model:
        classes_ = np.array(CLASSES)
        def predict_proba(self, x):
            return np.tile(np.array([.1,.7,.1,.1]), (len(x),1))
    monkeypatch.setenv('MODEL_PATH','/nonexistent/model.joblib')
    with TestClient(app) as client:
        app.state.model = Model()
        response = client.post('/predict',json={'flows':[FLOW]})
        assert response.status_code == 200
        result = response.json()['predictions'][0]
        assert result['flow_id'] == FLOW['flow_id']
        assert result['predicted_class'] == 'PortScan'
        assert abs(sum(result['probabilities'].values())-1) < 1e-9
