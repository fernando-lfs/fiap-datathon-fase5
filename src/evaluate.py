import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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

    print("Carregando modelo e dados de teste...")
    if not model_path.exists():
        raise FileNotFoundError("Modelo não encontrado. Execute src/train.py primeiro.")

    pipeline = joblib.load(model_path)
    X_test, y_test = load_test_data(data_dir)

    print("Realizando predições...")
    y_pred = pipeline.predict(X_test)

    # Métricas
    print("\n--- Relatório de Classificação ---")
    print(
        classification_report(
            y_test, y_pred, target_names=["Sem Risco (0)", "Risco (1)"]
        )
    )

    print("\n--- Matriz de Confusão ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN (Acertou Sem Risco): {cm[0][0]}")
    print(f"FP (Errou - Alarme Falso): {cm[0][1]}")
    print(f"FN (Errou - Perdeu Aluno em Risco): {cm[1][0]}")
    print(f"TP (Acertou Risco): {cm[1][1]}")

    # Justificativa de Negócio
    recall_risco = cm[1][1] / (cm[1][0] + cm[1][1])
    print(f"\n--- Análise de Negócio ---")
    print(f"Recall da Classe de Risco: {recall_risco:.2%}")
    print("Interpretação: De todos os alunos que realmente precisam de ajuda,")
    print(f"o modelo conseguiu identificar {recall_risco:.2%} deles.")


if __name__ == "__main__":
    evaluate_model()
