import pandas as pd
import joblib
import logging
import sklearn
import warnings  # <--- NOVO IMPORT
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from app.schemas import AlunoInput, PredicaoOutput
from app.config import settings
from src.utils import setup_logger

# IMPORTANTE: Necessário para o joblib reconstruir o pipeline corretamente
from src.feature_engineering import PedraMapper, BinaryCleaner  # noqa: F401

# 1. Configuração de Silenciamento de Warnings (Polimento de Logs)
# Ignora avisos de feature names do sklearn, pois garantimos a ordem via Pydantic
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Garante output pandas também na inferência
sklearn.set_config(transform_output="pandas")

# Configuração de Logs da Aplicação
app_logger = setup_logger("api", "api.log", level=settings.LOG_LEVEL)


# Configuração do Logger de Drift (Isolado)
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
            model = None
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

    # Injetamos apenas colunas estruturais necessárias
    defaults = {
        "ra": "API_REQ",
        "nome": "API_REQ",
        "turma": "API",
        "n_av": 0,
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df


@app.get("/model/info", tags=["Metadados"])
def get_model_info():
    """Retorna metadados sobre o modelo em produção."""
    if not model:
        raise HTTPException(status_code=503, detail="Modelo indisponível.")

    return {
        "nome_projeto": settings.PROJECT_NAME,
        "versao_api": settings.VERSION,
        "tipo_modelo": "Pipeline Scikit-Learn (Logistic Regression)",
        "status": "Ativo",
        "features_principais": [
            "Indicadores Psicossociais (IEG, IAA, IPS)",
            "Histórico de Classificação (Pedras)",
            "Notas Acadêmicas (Matemática, Português)",
        ],
    }


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
        prediction = model.predict(df_input)[0]

        # Tenta pegar probabilidade
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df_input)[0][1]
            except IndexError:
                proba = 1.0 if prediction == 1 else 0.0
        else:
            proba = 1.0 if prediction == 1 else 0.0

        risco = bool(prediction == 1)

        # Lógica Pedagógica de Resposta
        if proba >= 0.8:
            mensagem = "CRÍTICO: Risco muito alto de defasagem. Intervenção pedagógica imediata recomendada."
        elif proba >= 0.6:
            mensagem = (
                "ALERTA: Alto risco de defasagem. Acompanhamento próximo sugerido."
            )
        elif proba >= 0.5:
            mensagem = "ATENÇÃO: Risco moderado. Monitorar indicadores de engajamento."
        else:
            mensagem = (
                "ESTÁVEL: Aluno com bom prognóstico. Manter acompanhamento padrão."
            )

        # 3. Log para Monitoramento de Drift
        try:
            log_msg = f"{proba:.4f},{risco},{aluno.genero},{aluno.instituicao_de_ensino},{aluno.pedra_20}"
            drift_logger.info(log_msg)
        except Exception as e:
            app_logger.error(f"Falha não-bloqueante ao registrar log de drift: {e}")

        return PredicaoOutput(
            risco_defasagem=risco,
            probabilidade_risco=round(float(proba), 4),
            mensagem=mensagem,
        )

    except ValueError as ve:
        app_logger.error(f"Erro de validação do modelo: {ve}")
        raise HTTPException(
            status_code=422,
            detail=f"Dados de entrada inválidos para o modelo: {str(ve)}",
        )
    except Exception as e:
        app_logger.error(f"Erro genérico durante a predição: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Erro interno no processamento da predição."
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
