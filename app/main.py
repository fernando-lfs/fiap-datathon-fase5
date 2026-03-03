import pandas as pd
import joblib
import logging
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from app.schemas import AlunoInput, PredicaoOutput
from app.config import settings
from src.utils import setup_logger

# IMPORTANTE: Necessário para o joblib reconstruir o pipeline corretamente
from src.feature_engineering import PedraMapper, BinaryCleaner  # noqa: F401

# Garante output pandas também na inferência
sklearn.set_config(transform_output="pandas")

# 1. Configuração de Logs da Aplicação
app_logger = setup_logger("api", "api.log", level=settings.LOG_LEVEL)


# 2. Configuração do Logger de Drift (Isolado)
def get_drift_logger():
    """Configura logger específico para monitoramento de dados (Drift)."""
    logger = logging.getLogger("drift_monitor")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_dir = settings.BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "drift_data.csv")
        formatter = logging.Formatter("%(asctime)s,%(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


drift_logger = get_drift_logger()
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação (carregamento do modelo)."""
    global model
    if settings.MODEL_PATH.exists():
        try:
            model = joblib.load(settings.MODEL_PATH)
            app_logger.info(f"Modelo carregado com sucesso de: {settings.MODEL_PATH}")
        except Exception as e:
            app_logger.critical(f"Falha crítica ao carregar modelo: {e}")
            model = None  # Garante que model seja None se falhar
    else:
        app_logger.warning(
            f"Modelo não encontrado em {settings.MODEL_PATH}. A API retornará erros 503."
        )
    yield
    model = None


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para predição de risco de defasagem escolar - Case Passos Mágicos",
    lifespan=lifespan,
)


def prepare_input_dataframe(data: AlunoInput) -> pd.DataFrame:
    """
    Converte o input Pydantic para DataFrame compatível com o Pipeline.
    """
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # Colunas que podem ser esperadas pelo pipeline (mesmo que descartadas depois)
    # para evitar erro de "columns are missing" do Scikit-Learn se o ColumnTransformer
    # foi treinado vendo essas colunas no X_train original.
    # Nota: inde_22, cg, cf, ct foram removidos do treino, mas mantemos aqui por segurança.
    missing_cols = {
        "idade_22": 0,
        "n_av": 0,
        "ano_ingresso": 2022,
        "ano_nasc": 0,
        "ra": "API_REQ",
        "nome": "API_REQ",
        "turma": "API",
        "fase": 0,
        "ian": 0.0,
        "fase_ideal": 0,
        "inde_22": 0.0,
        "cg": 0,
        "cf": 0,
        "ct": 0,
    }

    for col, default_val in missing_cols.items():
        if col not in df.columns:
            df[col] = default_val

    return df


@app.post("/predict", response_model=PredicaoOutput, tags=["Predição"])
def predict(aluno: AlunoInput):
    if not model:
        raise HTTPException(
            status_code=503, detail="Modelo não carregado ou indisponível no servidor."
        )

    try:
        # 1. Prepara DataFrame
        df_input = prepare_input_dataframe(aluno)

        # 2. Predição
        # O pipeline cuida do pré-processamento completo.
        prediction = model.predict(df_input)[0]

        # Tenta pegar probabilidade, se o modelo suportar
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)[0][1]
        else:
            proba = 1.0 if prediction == 1 else 0.0

        risco = bool(prediction == 1)
        mensagem = (
            "ALERTA: Alto risco de defasagem. Intervenção pedagógica recomendada."
            if risco
            else "Aluno com bom desempenho. Manter acompanhamento padrão."
        )

        # 3. Log para Monitoramento de Drift
        try:
            # Logamos features chave para monitorar mudanças na distribuição dos dados
            # Formato CSV: proba, risco, genero, instituicao, pedra_20
            log_msg = f"{proba:.4f},{risco},{aluno.genero},{aluno.instituicao_de_ensino},{aluno.pedra_20}"
            drift_logger.info(log_msg)
        except Exception as e:
            app_logger.error(f"Falha não-bloqueante ao registrar log de drift: {e}")

        return PredicaoOutput(
            risco_defasagem=risco,
            probabilidade_risco=round(float(proba), 4),
            mensagem=mensagem,
        )

    except Exception as e:
        app_logger.error(f"Erro durante a predição: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Erro no processamento da predição: {str(e)}"
        )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Monitoramento"])
def health_check():
    return {
        "project": settings.PROJECT_NAME,
        "status": "online",
        "model_loaded": model is not None,
    }
