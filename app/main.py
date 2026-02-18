import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pathlib import Path
from app.schemas import AlunoInput, PredicaoOutput

# Configuração de Caminhos
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "pipeline.joblib"

app = FastAPI(
    title="API Passos Mágicos - Previsão de Risco",
    description="API para prever risco de defasagem escolar com base em indicadores pedagógicos.",
    version="1.0.0",
)

# Carregamento do Modelo Global
model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso!")


def preprocess_input(data: AlunoInput) -> pd.DataFrame:
    """
    Transforma o input Pydantic em DataFrame compatível com o modelo.
    Aplica as mesmas transformações manuais do feature_engineering.
    """
    # 1. Converter para Dict
    input_dict = data.dict()

    # Ajuste de nome de coluna (n_av -> nº_av) se necessário pelo modelo
    # No feature_engineering, a coluna era 'nº_av'. O Pydantic não aceita caracteres especiais facilmente.
    input_dict["nº_av"] = input_dict.pop("n_av")

    df = pd.DataFrame([input_dict])

    # 2. Mapeamento de Pedras (Lógica replicada de feature_engineering)
    pedra_map = {
        "quartzo": 1,
        "ágata": 2,
        "agata": 2,
        "ametista": 3,
        "topázio": 4,
        "topazio": 4,
    }
    cols_pedra = ["pedra_20", "pedra_21", "pedra_22"]
    for col in cols_pedra:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.lower().map(pedra_map).fillna(0).astype(int)
            )

    # 3. Binarização (Sim/Não)
    binary_map = {"sim": 1, "não": 0, "nao": 0, "s": 1, "n": 0}
    cols_binary = ["indicado", "atingiu_pv"]
    for col in cols_binary:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.lower().map(binary_map).fillna(0).astype(int)
            )

    return df


@app.post("/predict", response_model=PredicaoOutput)
def predict(aluno: AlunoInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    try:
        # Pré-processamento
        df_input = preprocess_input(aluno)

        # Predição
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]  # Probabilidade da classe 1 (Risco)

        # Resposta
        risco = bool(prediction == 1)
        mensagem = (
            "ALERTA: Alto risco de defasagem. Intervenção recomendada."
            if risco
            else "Aluno com bom desempenho. Manter acompanhamento padrão."
        )

        return PredicaoOutput(
            risco_defasagem=risco,
            probabilidade_risco=round(float(proba), 4),
            mensagem=mensagem,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
