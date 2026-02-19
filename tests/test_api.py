from fastapi.testclient import TestClient
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
    # O 'with' garante que o lifespan (startup) seja executado
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 200
        data = response.json()

        # Verifica a estrutura da resposta
        assert "risco_defasagem" in data
        assert "probabilidade_risco" in data
        assert "mensagem" in data
        assert isinstance(data["risco_defasagem"], bool)
        assert isinstance(data["probabilidade_risco"], float)


def test_predict_error_handling():
    # Testa envio de dados inválidos (ex: string onde deveria ser float)
    invalid_payload = sample_payload.copy()
    invalid_payload["iaa"] = "texto_invalido"  # Deveria ser float

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
        # O Pydantic deve barrar antes mesmo de chegar no modelo (422 Unprocessable Entity)
        assert response.status_code == 422
