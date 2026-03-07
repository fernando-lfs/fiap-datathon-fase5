from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

# Payload de exemplo ajustado ao app/schemas.py
sample_payload = {
    "genero": "Menina",
    "instituicao_de_ensino": "Escola Pública",
    "pedra_20": "Ametista",
    "pedra_21": "Ágata",
    "iaa": 8.5,
    "ieg": 7.2,
    "ips": 6.8,
    "ida": 5.5,
    "ipp": 7.0,
    "ipv": 7.2,
    "matem": 6.0,
    "portug": 7.5,
    "ingles": 5.0,
    "indicado": "Não",
    "atingiu_pv": "Não",
    "ponto_virada": "Não",
    "indicado_bolsa": "Não",
}


def test_health_check():
    """Testa o endpoint de saúde (/health)."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"


def test_predict_endpoint_success():
    """
    Testa o fluxo feliz da predição (Cenário Crítico).
    """
    # Mock do objeto modelo do scikit-learn
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]  # Simula classe 1 (Risco)
    # Simula probabilidade: [prob_classe_0, prob_classe_1]
    # 0.8 deve acionar a mensagem "CRÍTICO" na nova lógica
    mock_model.predict_proba.return_value = [[0.2, 0.8]]

    # Patch no joblib.load dentro do namespace de app.main
    with patch("app.main.joblib.load", return_value=mock_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)

            assert response.status_code == 200
            data = response.json()

            assert data["risco_defasagem"] is True
            assert data["probabilidade_risco"] == 0.8
            # CORREÇÃO: Agora esperamos "CRÍTICO" para prob >= 0.8
            assert "CRÍTICO" in data["mensagem"]


def test_predict_endpoint_no_risk():
    """Testa predição quando não há risco (Cenário Estável)."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    # Probabilidade baixa (0.1) -> Mensagem "ESTÁVEL"
    mock_model.predict_proba.return_value = [[0.9, 0.1]]

    with patch("app.main.joblib.load", return_value=mock_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)
            assert response.status_code == 200
            data = response.json()
            assert data["risco_defasagem"] is False
            assert data["probabilidade_risco"] == 0.1
            # Opcional: Validar mensagem de estabilidade
            assert "ESTÁVEL" in data["mensagem"]


def test_predict_validation_error():
    """Testa envio de dados inválidos (validação Pydantic)."""
    invalid_payload = sample_payload.copy()
    # Envia string onde deveria ser float
    invalid_payload["iaa"] = "texto_invalido"

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422


def test_model_info_endpoint():
    """Testa o novo endpoint de metadados."""
    # Precisamos mockar o modelo carregado para o endpoint funcionar
    with patch("app.main.model", MagicMock()):
        with TestClient(app) as client:
            response = client.get("/model/info")
            assert response.status_code == 200
            data = response.json()
            assert "nome_projeto" in data
            assert "features_principais" in data
