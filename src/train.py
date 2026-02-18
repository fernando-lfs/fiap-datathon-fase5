import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_train_data(data_dir: Path):
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    # Garante que y é um array 1D
    return X_train, y_train.values.ravel()


def create_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Cria o pipeline de pré-processamento e modelo.
    """
    # 1. Identificação de colunas
    categorical_features = ["genero", "instituicao_de_ensino"]

    # Todas as outras são numéricas
    numerical_features = [
        col for col in X_train.columns if col not in categorical_features
    ]

    # 2. Transformadores
    # Para numéricos: Imputer (preenche nulos com média) + Scaler (normaliza escala)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Para categóricos: OneHotEncoder (transforma texto em colunas binárias)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 3. Montagem do Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 4. Pipeline Final com Modelo Baseline (Regressão Logística)
    # class_weight='balanced' é CRUCIAL pois temos muito mais alunos em risco (1) do que fora (0)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=42, class_weight="balanced", max_iter=1000
                ),
            ),
        ]
    )

    return model


def run_training():
    root = get_project_root()
    data_dir = root / "data" / "processed"
    model_dir = root / "app" / "model"

    print("Carregando dados de treino...")
    X_train, y_train = load_train_data(data_dir)

    print("Construindo pipeline...")
    pipeline = create_pipeline(X_train)

    print("Treinando modelo Baseline (Logistic Regression)...")
    pipeline.fit(X_train, y_train)

    # Salvar modelo
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Modelo treinado e salvo em: {model_path}")


if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        print(f"Erro no treinamento: {e}")
