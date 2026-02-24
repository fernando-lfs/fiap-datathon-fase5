import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import setup_logger

# Import necessário para o joblib reconhecer as classes customizadas ao carregar
from src.transformers import PedraMapper, BinaryCleaner

logger = setup_logger("evaluate")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_test_data(data_dir: Path):
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")
    return X_test, y_test.values.ravel()


def evaluate_model():
    root = get_project_root()
    data_dir = root / "data" / "processed"
    model_path = root / "app" / "model" / "pipeline.joblib"

    if not model_path.exists():
        logger.error("Modelo não encontrado.")
        return

    logger.info("Carregando modelo e dados de teste...")
    pipeline = joblib.load(model_path)
    X_test, y_test = load_test_data(data_dir)

    logger.info("Realizando predições...")
    y_pred = pipeline.predict(X_test)

    # Relatórios
    report = classification_report(y_test, y_pred, target_names=["Sem Risco", "Risco"])
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Relatório de Classificação ---")
    print(report)

    recall_risco = cm[1][1] / (cm[1][0] + cm[1][1])
    logger.info(f"Recall da Classe de Risco: {recall_risco:.2%}")

    # Loga métricas importantes para histórico
    logger.info(
        f"Matriz de Confusão: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}"
    )


if __name__ == "__main__":
    evaluate_model()
