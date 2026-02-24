import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from src.utils import setup_logger
from src.transformers import PedraMapper, BinaryCleaner

logger = setup_logger("train")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_train_data(data_dir: Path):
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    return X_train, y_train.values.ravel()


def create_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Cria o pipeline completo incluindo pré-processamento customizado.
    """
    # 1. Definição de Colunas
    # Colunas que sabemos que são categóricas e precisam de OneHot
    categorical_features = ["genero", "instituicao_de_ensino"]

    # Colunas de Pedras e Binárias são tratadas pelos Transformers Customizados
    # O restante é numérico
    cols_pedra = ["pedra_20", "pedra_21", "pedra_22"]
    cols_binary = ["indicado", "atingiu_pv", "indicado_bolsa", "ponto_virada"]

    exclude_cols = categorical_features + cols_pedra + cols_binary
    numerical_features = [c for c in X_train.columns if c not in exclude_cols]

    # 2. Pipeline de Transformação Numérica
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 3. Pipeline de Transformação Categórica
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 4. ColumnTransformer
    # Aplica transformações específicas em colunas específicas
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            # As colunas de pedra e binárias passam "passthrough" aqui
            # pois serão tratadas pelos Custom Transformers no início do pipeline principal
            # ou podemos aplicar o scaler nelas se quisermos, mas vamos manter simples.
        ],
        remainder="passthrough",  # Mantém as colunas que já foram tratadas pelos custom transformers
    )

    # 5. Pipeline Principal
    # A ordem é: Limpa Pedras -> Limpa Binários -> Preprocessa (Scale/OneHot) -> Modelo
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

    logger.info("Carregando dados de treino...")
    X_train, y_train = load_train_data(data_dir)

    logger.info("Construindo pipeline...")
    pipeline = create_pipeline(X_train)

    logger.info("Treinando modelo Baseline (Logistic Regression)...")
    pipeline.fit(X_train, y_train)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)

    logger.info(f"Modelo treinado e salvo em: {model_path}")


if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        logger.critical(f"Erro no treinamento: {e}")
