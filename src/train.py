import pandas as pd
import joblib
import sklearn
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from src.utils import setup_logger
from src.transformers import PedraMapper, BinaryCleaner

# Garante output pandas em todo o pipeline
sklearn.set_config(transform_output="pandas")

logger = setup_logger("train")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_train_data(data_dir: Path):
    try:
        X_train = pd.read_csv(data_dir / "X_train.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv")
        return X_train, y_train.values.ravel()
    except FileNotFoundError:
        logger.error("Arquivos de treino não encontrados.")
        raise


def create_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Cria o pipeline completo de ML.
    """

    # 1. Definição de Grupos de Colunas
    ideal_categorical = ["genero", "instituicao_de_ensino"]

    # NOTA: Pedra 22 removida intencionalmente (Leakage)
    ideal_pedra = ["pedra_20", "pedra_21"]

    ideal_binary = ["indicado", "atingiu_pv", "indicado_bolsa", "ponto_virada"]

    # Seleção dinâmica baseada no que realmente existe no X_train
    cols_categorical = [c for c in ideal_categorical if c in X_train.columns]
    cols_pedra = [c for c in ideal_pedra if c in X_train.columns]
    cols_binary = [c for c in ideal_binary if c in X_train.columns]

    # Numéricas são todas as outras não listadas acima
    exclude_cols = cols_categorical + cols_pedra + cols_binary
    cols_numerical = [
        c
        for c in X_train.select_dtypes(include=["number"]).columns
        if c not in exclude_cols
    ]

    logger.info(f"Features Numéricas: {len(cols_numerical)}")
    logger.info(f"Features Categóricas: {len(cols_categorical)}")
    logger.info(f"Features Pedra: {len(cols_pedra)}")
    logger.info(f"Features Binárias: {len(cols_binary)}")

    # 2. Sub-Pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Para colunas que viraram numéricas via PedraMapper/BinaryCleaner
    generated_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 3. ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cols_numerical),
            ("cat", categorical_transformer, cols_categorical),
            ("gen_num", generated_numeric_transformer, cols_pedra + cols_binary),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # 4. Pipeline Principal
    model_pipeline = Pipeline(
        steps=[
            ("pedra_mapper", PedraMapper()),
            ("binary_cleaner", BinaryCleaner()),
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=42, class_weight="balanced", max_iter=1000
                ),
            ),
        ]
    )

    return model_pipeline


def run_training():
    root = get_project_root()
    data_dir = root / "data" / "processed"
    model_dir = root / "app" / "model"

    logger.info("Iniciando processo de treinamento...")

    if not (data_dir / "X_train.csv").exists():
        raise FileNotFoundError("Dados de treino não encontrados.")

    X_train, y_train = load_train_data(data_dir)

    logger.info("Construindo pipeline...")
    pipeline = create_pipeline(X_train)

    logger.info("Treinando modelo Baseline (Logistic Regression)...")
    pipeline.fit(X_train, y_train)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)

    logger.info(f"Modelo treinado com sucesso e salvo em: {model_path}")


if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        logger.critical(f"Erro fatal no treinamento: {e}")
        raise
