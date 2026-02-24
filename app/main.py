import pandas as pd
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pathlib import Path
from contextlib import asynccontextmanager
from app.schemas import AlunoInput, PredicaoOutput

# Import necessário para o joblib carregar o pipeline
from src.transformers import PedraMapper, BinaryCleaner
from src.utils import setup_logger

# Configuração de Caminhos
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "pipeline.joblib"

# Logger da Aplicação
app_logger = setup_logger("api", "api.log")

# Logger para Monitoramento de Drift (Salva inputs e outputs)
drift_logger = logging.getLogger("drift_monitor")
drift_logger.setLevel(logging.INFO)
drift_handler = logging.FileHandler("logs/drift_data.csv")
drift_handler.setFormatter(logging.Formatter("%(asctime)s,%(message)s"))
drift_logger.addHandler(drift_handler)

# Variável global para o modelo
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            app_logger.info("Modelo carregado com sucesso!")
        except Exception as e:
            app_logger.critical(f"Falha ao carregar modelo: {e}")
    else:
        app_logger.warning(f"Modelo não encontrado em {MODEL_PATH}")
    yield
    model = None


app = FastAPI(
    title="API Passos Mágicos - Previsão de Risco",
    version="1.0.0",
    lifespan=lifespan,
)


def prepare_input_dataframe(data: AlunoInput) -> pd.DataFrame:
    """
    Converte o input Pydantic para DataFrame.
    NÃO realiza transformações de valores (Pedras/Binários), pois isso está no Pipeline.
    """
    input_dict = data.model_dump()

    # Ajuste de nome de coluna para bater com o treinamento (snake_case)
    if "n_av" in input_dict:
        input_dict["nº_av"] = input_dict.pop("n_av")

    df = pd.DataFrame([input_dict])
    return df


@app.post("/predict", response_model=PredicaoOutput)
def predict(aluno: AlunoInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    try:
        # 1. Prepara DataFrame (Raw Data)
        df_input = prepare_input_dataframe(aluno)

        # 2. Predição (O Pipeline cuida de limpar e transformar)
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

        risco = bool(prediction == 1)
        mensagem = (
            "ALERTA: Alto risco de defasagem. Intervenção recomendada."
            if risco
            else "Aluno com bom desempenho. Manter acompanhamento padrão."
        )

        # 3. Log para Monitoramento de Drift (CSV format)
        # Loga: probabilidade, risco, e os dados de entrada principais
        log_msg = f"{proba},{risco},{aluno.genero},{aluno.instituicao_de_ensino},{aluno.pedra_20},{aluno.inde_2022 if hasattr(aluno, 'inde_2022') else ''}"
        drift_logger.info(log_msg)

        return PredicaoOutput(
            risco_defasagem=risco,
            probabilidade_risco=round(float(proba), 4),
            mensagem=mensagem,
        )

    except Exception as e:
        app_logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro no processamento: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
