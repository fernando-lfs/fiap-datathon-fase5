from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

# Payload de exemplo ajustado ao app/schemas.py
# Representa um aluno com dados completos para teste de integração da rota
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
    """
    Testa o endpoint de saúde (/health).
    Deve retornar status 200 e indicar que a API está online.
    """
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"


def test_predict_endpoint_success():
    """
    Testa o fluxo feliz da predição (Cenário: Risco Crítico).

    Estratégia de Teste (Mocking):
    Em vez de carregar o modelo real (pesado), "mockamos" (simulamos) o objeto joblib.
    Isso garante que o teste seja rápido e isole a lógica da API da lógica do modelo.

    Cenário:
    - Modelo prevê classe 1 (Risco).
    - Probabilidade simulada de 0.9 (90%).
    - Resultado esperado: Mensagem de risco "CRÍTICO" (Threshold >= 0.85).
    """
    # Mock do objeto modelo do scikit-learn
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    # Simula probabilidade: [prob_classe_0, prob_classe_1]
    mock_model.predict_proba.return_value = [[0.1, 0.9]]

    # Patch no joblib.load para injetar nosso mock
    with patch("app.main.joblib.load", return_value=mock_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)

            assert response.status_code == 200
            data = response.json()

            assert data["risco_defasagem"] is True
            assert data["probabilidade_risco"] == 0.9
            assert "CRÍTICO" in data["mensagem"]


def test_predict_endpoint_no_risk():
    """
    Testa predição quando não há risco (Cenário: Estável).

    Cenário:
    - Modelo prevê classe 0 (Sem Risco).
    - Probabilidade de risco baixa (0.1).
    - Resultado esperado: Mensagem "ESTÁVEL".
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.9, 0.1]]

    with patch("app.main.joblib.load", return_value=mock_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample_payload)
            assert response.status_code == 200
            data = response.json()
            assert data["risco_defasagem"] is False
            assert data["probabilidade_risco"] == 0.1
            assert "ESTÁVEL" in data["mensagem"]


def test_predict_validation_error():
    """
    Testa a robustez da API contra dados inválidos.
    O Pydantic deve interceptar o erro antes de chegar ao modelo.
    """
    invalid_payload = sample_payload.copy()
    # Envia string onde deveria ser float
    invalid_payload["iaa"] = "texto_invalido"

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422  # Unprocessable Entity


def test_model_info_endpoint():
    """
    Testa o endpoint de metadados (/model/info).
    Verifica se as informações de versão e features estão presentes.
    """
    with patch("app.main.model", MagicMock()):
        with TestClient(app) as client:
            response = client.get("/model/info")
            assert response.status_code == 200
            data = response.json()
            assert "nome_projeto" in data
            assert "features_principais" in data
