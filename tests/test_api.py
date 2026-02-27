from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

# Payload de exemplo para reuso
sample_payload = {
    "genero": "Menina",
    "ano_ingresso": 2018,
    "instituicao_de_ensino": "Escola Pública",
    "pedra_20": "Ametista",
    "pedra_21": "Ágata",
    "pedra_22": "Quartzo",
    "n_av": 4,
    "iaa": 8.5,
    "ieg": 7.2,
    "ips": 6.8,
    "ida": 5.5,
    "matem": 6.0,
    "portug": 7.5,
    "ingles": 5.0,
    "indicado": "Não",
    "atingiu_pv": "Não",
    "ipv": 7.2,
}


def test_health_check():
    """
    Testa o endpoint de saúde (/health).
    Nota: A raiz (/) redireciona para /docs, então testamos /health.
    """
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "project" in data
        assert "model_loaded" in data


def test_predict_endpoint_success():
    """
    Testa o fluxo feliz da predição.
    Mockamos o 'model' dentro do app.main para não depender do arquivo físico.
    """
    # Mock do objeto modelo do scikit-learn
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]  # Simula classe 1 (Risco)
    mock_model.predict_proba.return_value = [[0.2, 0.8]]  # 80% de probabilidade

    # Patch no objeto 'model' dentro do módulo app.main
    with patch("app.main.model", mock_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)

            assert response.status_code == 200
            data = response.json()

            # Verifica a estrutura da resposta
            assert data["risco_defasagem"] is True
            assert data["probabilidade_risco"] == 0.8
            assert "mensagem" in data
            assert "ALERTA" in data["mensagem"]


def test_predict_endpoint_no_risk():
    """Testa predição quando não há risco."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.9, 0.1]]

    with patch("app.main.model", mock_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)
            assert response.status_code == 200
            assert response.json()["risco_defasagem"] is False


def test_predict_error_handling():
    """Testa envio de dados inválidos (validação Pydantic)."""
    invalid_payload = sample_payload.copy()
    invalid_payload["iaa"] = "texto_invalido"  # Deveria ser float

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
        # O Pydantic deve barrar antes mesmo de chegar no modelo (422 Unprocessable Entity)
        assert response.status_code == 422
