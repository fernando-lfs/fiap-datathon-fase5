import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pathlib import Path
from contextlib import asynccontextmanager
from app.schemas import AlunoInput, PredicaoOutput

# Configuração de Caminhos
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "pipeline.joblib"

# Variável global para o modelo
model = None


# Nova lógica de ciclo de vida (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carregar modelo ao iniciar
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("Modelo carregado com sucesso!")
    else:
        print(f"AVISO: Modelo não encontrado em {MODEL_PATH}")
    yield
    # Código para executar ao desligar (se necessário)
    model = None


app = FastAPI(
    title="API Passos Mágicos - Previsão de Risco",
    description="API para prever risco de defasagem escolar com base em indicadores pedagógicos.",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess_input(data: AlunoInput) -> pd.DataFrame:
    """
    Transforma o input Pydantic em DataFrame compatível com o modelo.
    """
    input_dict = data.model_dump()  # model_dump() é o novo dict() no Pydantic V2

    # Ajuste de nome de coluna
    input_dict["nº_av"] = input_dict.pop("n_av")

    df = pd.DataFrame([input_dict])

    # Mapeamento de Pedras
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

    # Binarização
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
        df_input = preprocess_input(aluno)

        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

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
