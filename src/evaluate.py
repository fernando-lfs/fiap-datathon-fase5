import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import setup_logger

# Import necessário para o joblib reconhecer as classes customizadas
from src.transformers import PedraMapper, BinaryCleaner  # noqa: F401

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
        logger.error("Modelo não encontrado. Execute o treinamento primeiro.")
        return

    logger.info("Carregando modelo e dados de teste...")
    pipeline = joblib.load(model_path)
    X_test, y_test = load_test_data(data_dir)

    logger.info("Realizando predições...")
    y_pred = pipeline.predict(X_test)

    # Relatórios
    # target_names ajuda na interpretação do output textual
    report = classification_report(
        y_test, y_pred, target_names=["Sem Risco", "Risco"], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Relatório de Classificação ---")
    print(report)
    print("\n--- Matriz de Confusão ---")
    try:
        tn, fp, fn, tp = cm.ravel()
        print(f"TN: {tn} | FP: {fp}")
        print(f"FN: {fn} | TP: {tp}")

        # Cálculo seguro do Recall da classe positiva
        denominator = fn + tp
        if denominator > 0:
            recall_risco = tp / denominator
            logger.info(f"Recall da Classe de Risco: {recall_risco:.2%}")
        else:
            logger.warning(
                "Não há exemplos positivos no conjunto de teste para calcular Recall."
            )

        logger.info(f"Matriz de Confusão: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    except ValueError:
        # Caso a matriz de confusão não tenha formato 2x2 (ex: só uma classe no teste)
        logger.warning(f"Matriz de confusão com formato inesperado: {cm.shape}")


if __name__ == "__main__":
    evaluate_model()
