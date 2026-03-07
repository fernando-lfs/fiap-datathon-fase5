import pandas as pd
import joblib
import logging
import sklearn
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from app.schemas import AlunoInput, PredicaoOutput
from app.config import settings
from src.utils import setup_logger

# IMPORTANTE: Necessário para o joblib reconstruir o pipeline corretamente
from src.feature_engineering import PedraMapper, BinaryCleaner  # noqa: F401

# 1. Configuração de Silenciamento de Warnings (Polimento de Logs)
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


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    Carrega o modelo serializado (.joblib) na inicialização para memória.
    """
    global model
    if settings.MODEL_PATH.exists():
        try:
            model = joblib.load(settings.MODEL_PATH)
            app_logger.info(f"Modelo carregado com sucesso de: {settings.MODEL_PATH}")
        except Exception as e:
            app_logger.critical(f"Falha crítica ao carregar modelo: {e}")
            model = None
    else:
        app_logger.warning(f"Modelo não encontrado em {settings.MODEL_PATH}.")
    yield
    model = None


# --- Definição da API ---
app = FastAPI(
    title="🔮 API Passos Mágicos - Predição de Risco Escolar",
    version=settings.VERSION,
    description="""
    🎓 Sobre o Projeto
    Esta API fornece serviços de Machine Learning para a **Associação Passos Mágicos**, focando na identificação precoce de alunos com risco de defasagem escolar.
    
    🚀 Funcionalidades Principais
    Predição de Risco: Classifica alunos com base em histórico acadêmico e psicossocial.
    Interpretabilidade: Retorna mensagens pedagógicas claras para auxiliar a tomada de decisão.
    Monitoramento: Registra dados de inferência para análise de Data Drift.
    
    🛠️ Como usar
    Utilize o endpoint `/predict` enviando os dados do aluno. Consulte os exemplos disponíveis no schema para cenários de **Alto Risco** e **Baixo Risco**.
    """,
    lifespan=lifespan,
)


def prepare_input_dataframe(data: AlunoInput) -> pd.DataFrame:
    """
    Converte o input Pydantic para DataFrame compatível com o Pipeline.
    Preenche colunas estruturais (RA, Nome) com valores dummy para satisfazer
    a estrutura esperada pelo modelo, sem afetar a predição.
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


@app.get(
    "/model/info",
    tags=["Auditoria"],
    summary="Obter Metadados do Modelo",
    description="Retorna informações técnicas sobre o modelo carregado em memória, incluindo versão, tipo de algoritmo e features utilizadas.",
)
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


@app.post(
    "/predict",
    response_model=PredicaoOutput,
    tags=["Predição"],
    summary="Calcular Risco de Defasagem Escolar",
    description="""
    Este endpoint processa o perfil completo do aluno para calcular a probabilidade de defasagem.
    Abaixo, detalhamos todos os parâmetros aceitos para facilitar a simulação.

    1. 👤 Perfil e Histórico
    * genero: Gênero do aluno (`Menina` ou `Menino`).
    * instituicao_de_ensino: Tipo de escola de origem (ex: `Escola Pública`).
    * pedra_20 / pedra_21: Classificação global do aluno nos anos anteriores.
      * Hierarquia: `Quartzo` (Baixo) < `Ágata` < `Ametista` < `Topázio` (Alto).
      * Nota: Se o aluno não estava na ONG em 2020/21, deixe como `null`.

    2. 📊 Indicadores PEDE (Escala 0 a 10)
    Estes são os indicadores proprietários da metodologia Passos Mágicos:
    * IEG (Engajamento): Mede a entrega de lições de casa e participação em aulas. **(Fator Crítico)**
    * IAA (Autoavaliação): Como o aluno avalia seu próprio desempenho e sentimentos.
    * IPS (Psicossocial): Avaliação da equipe de psicologia sobre interações sociais e familiares.
    * IPP (Psicopedagógico): Avaliação sobre o desenvolvimento cognitivo e de aprendizado.
    * IDA (Desempenho Acadêmico): Média das provas internas realizadas na associação.
    * IPV (Ponto de Virada): Índice que mede o potencial de transformação social do aluno.

    3. 🏫 Desempenho Escolar (Escala 0 a 10)
    Notas obtidas na escola regular (ensino fundamental/médio):
    * matem: Nota de Matemática.
    * portug: Nota de Português.
    * ingles: Nota de Inglês (Opcional).

    4. 🏆 Indicadores de Sucesso (Sim/Não)
    Variáveis binárias que indicam conquistas ou status específicos:
    * indicado: O aluno foi destaque em alguma atividade?
    * atingiu_pv: O aluno atingiu formalmente o "Ponto de Virada"?
    * ponto_virada: Indicador consolidado de virada de chave no desenvolvimento.
    * indicado_bolsa: O aluno foi indicado para bolsa de estudos em escolas parceiras?

    ---

    🧠 Lógica de Decisão (Saída)
    O modelo retorna uma probabilidade (0 a 1) que é traduzida nas seguintes categorias de intervenção:

    | Probabilidade | Classificação | Ação Recomendada |
    | :--- | :--- | :--- |
    | ≥ 0.80 | 🔴 CRÍTICO | Intervenção pedagógica imediata. Risco altíssimo. |
    | ≥ 0.60 | 🟠 ALERTA | Acompanhamento próximo. Alto risco. |
    | ≥ 0.50 | 🟡 ATENÇÃO | Monitorar indicadores de engajamento. Risco moderado. |
    | < 0.50 | 🟢 ESTÁVEL | Manter acompanhamento padrão. Baixo risco. |
    """,
    responses={
        200: {
            "description": "Predição realizada com sucesso.",
            "content": {
                "application/json": {
                    "example": {
                        "risco_defasagem": True,
                        "probabilidade_risco": 0.8245,
                        "mensagem": "CRÍTICO: Risco muito alto de defasagem. Intervenção pedagógica imediata recomendada.",
                    }
                }
            },
        },
        422: {
            "description": "Erro de Validação. Ocorre se enviar notas fora do intervalo 0-10 ou valores inválidos para campos categóricos.",
        },
        503: {
            "description": "Serviço Indisponível. Ocorre se o arquivo do modelo (pipeline.joblib) não for encontrado no servidor.",
        },
    },
)
def predict(aluno: AlunoInput):
    """
    Realiza a predição de risco de defasagem escolar.

    Regras de Negócio para Mensagens:
    - Probabilidade >= 0.8: Risco CRÍTICO (Intervenção imediata).
    - Probabilidade >= 0.6: ALERTA (Alto risco).
    - Probabilidade >= 0.5: ATENÇÃO (Risco moderado).
    - Probabilidade < 0.5: ESTÁVEL (Baixo risco).

    Registra logs de inferência para monitoramento de Data Drift em logs/drift_data.csv.
    """
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


@app.get(
    "/health",
    tags=["Monitoramento"],
    summary="Verificar Status da API",
    description="Endpoint de Health Check para monitoramento (Liveness/Readiness Probe). Retorna o status da aplicação e se o modelo de ML está carregado em memória.",
)
def health_check():
    return {
        "project": settings.PROJECT_NAME,
        "status": "online",
        "model_loaded": model is not None,
    }
