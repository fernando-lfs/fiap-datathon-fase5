import joblib
import pandas as pd
import sklearn
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.utils import setup_logger

# Import necessário para o joblib reconhecer as classes customizadas ao carregar o pipeline
from src.feature_engineering import PedraMapper, BinaryCleaner  # noqa: F401

# Garante output pandas
sklearn.set_config(transform_output="pandas")

logger = setup_logger("evaluate")


def get_project_root() -> Path:
    """
    Retorna o caminho absoluto para a raiz do projeto.

    Returns:
        Path: Objeto Path apontando para a raiz.
    """
    return Path(__file__).resolve().parent.parent


def evaluate_model():
    """
    Executa a avaliação do modelo treinado utilizando o conjunto de teste.

    Fluxo:
    1. Carrega o pipeline serializado (.joblib).
    2. Carrega os dados de teste processados (X_test, y_test).
    3. Gera predições e calcula métricas.
    4. Exibe relatório de classificação e matriz de confusão.

    Métrica de Negócio (KPI):
    O foco da avaliação é o RECALL da classe positiva (1 - Risco).
    No contexto da Passos Mágicos, o custo de um Falso Negativo (deixar de identificar
    um aluno em risco) é muito superior ao de um Falso Positivo (intervir em um aluno
    que não precisava).
    """
    root = get_project_root()
    data_dir = root / "data" / "processed"
    model_path = root / "app" / "model" / "pipeline.joblib"

    if not model_path.exists():
        logger.error("Modelo não encontrado. Execute o treinamento primeiro.")
        return

    logger.info("Carregando modelo e dados de teste...")

    try:
        # Carrega o pipeline treinado
        pipeline = joblib.load(model_path)

        # Carrega os dados de teste processados (CSV)
        # Usamos read_csv direto para evitar re-processamento desnecessário
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_test_df = pd.read_csv(data_dir / "y_test.csv")
        y_test = y_test_df.values.ravel()

        logger.info(f"Dados de teste carregados: {X_test.shape}")

    except Exception as e:
        logger.error(f"Erro ao carregar recursos: {e}")
        return

    logger.info("Realizando predições...")
    try:
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        logger.critical(f"Erro ao realizar predição: {e}")
        return

    # Relatórios
    logger.info("Gerando métricas...")

    # Acurácia Geral
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Acurácia Global: {acc:.2%}")

    # Relatório Detalhado
    report = classification_report(
        y_test, y_pred, target_names=["Sem Risco (0)", "Risco (1)"], zero_division=0
    )

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO DO MODELO (BASELINE)")
    print("=" * 60)
    print(report)
    print("-" * 30)

    try:
        tn, fp, fn, tp = cm.ravel()
        print("MATRIZ DE CONFUSÃO:")
        print(f"Verdadeiros Negativos (Sem Risco e previu Sem Risco): {tn}")
        print(f"Falsos Positivos      (Sem Risco mas previu Risco):   {fp}")
        print(
            f"Falsos Negativos      (Risco mas previu Sem Risco):   {fn}  <-- PONTO CRÍTICO"
        )
        print(f"Verdadeiros Positivos (Risco e previu Risco):         {tp}")
        print("-" * 30)

        denominator = fn + tp
        if denominator > 0:
            recall_risco = tp / denominator
            print(f"METRICA DE NEGÓCIO (RECALL - CLASSE DE RISCO): {recall_risco:.2%}")
            print("Interpretação: De todos os alunos que realmente têm risco,")
            print(f"o modelo conseguiu identificar {recall_risco:.2%} deles.")
        else:
            logger.warning("Não há exemplos positivos no conjunto de teste.")

    except ValueError:
        logger.warning(f"Matriz de confusão com formato inesperado: {cm.shape}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    evaluate_model()
