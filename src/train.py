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
from src.feature_engineering import PedraMapper, BinaryCleaner

# Garante que o Scikit-Learn retorne Pandas DataFrames nas transformações
sklearn.set_config(transform_output="pandas")
logger = setup_logger("train")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def create_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Cria o pipeline completo de ML.
    """
    # 1. Definição de Grupos de Colunas
    ideal_categorical = ["genero", "instituicao_de_ensino"]

    # Removemos 'pedra_22' pois é futuro/resultado em relação ao histórico
    ideal_pedra = ["pedra_20", "pedra_21"]

    ideal_binary = ["indicado", "atingiu_pv", "indicado_bolsa", "ponto_virada"]

    # ==============================================================================
    # PREVENÇÃO DE DATA LEAKAGE (CORRIGIDO)
    # ==============================================================================
    # Estas colunas são resultados de 2022 ou proxies diretos do alvo.
    # O modelo deve prever o risco baseado no histórico (2020/2021) e notas parciais.
    forbidden_cols = [
        "ra",
        "nome",
        "turma",
        "alvo",  # Identificadores
        "ian",  # VAZAMENTO: Proxy direto da defasagem
        "fase_ideal",  # VAZAMENTO: Usado no cálculo da defasagem
        "defas",  # VAZAMENTO: Variável original do alvo
        "inde_22",  # VAZAMENTO: Resultado final consolidado de 2022
        "cg",
        "cf",
        "ct",  # VAZAMENTO: Rankings baseados no desempenho de 2022
        "pedra_22",  # VAZAMENTO: Classificação final de 2022
    ]

    # Seleção dinâmica de colunas presentes no DataFrame
    cols_categorical = [c for c in ideal_categorical if c in X_train.columns]
    cols_pedra = [c for c in ideal_pedra if c in X_train.columns]
    cols_binary = [c for c in ideal_binary if c in X_train.columns]

    # Numéricas: Tudo que sobra, exceto as proibidas e as já selecionadas acima
    exclude_cols = cols_categorical + cols_pedra + cols_binary + forbidden_cols

    cols_numerical = [
        c
        for c in X_train.select_dtypes(include=["number"]).columns
        if c not in exclude_cols
    ]

    logger.info(f"Features Numéricas selecionadas: {len(cols_numerical)}")
    logger.info(f"Features Categóricas selecionadas: {len(cols_categorical)}")
    logger.debug(f"Colunas excluídas (Forbidden/Outras): {exclude_cols}")

    # 2. Pipelines de Transformação
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

    # Para colunas que viram números via PedraMapper/BinaryCleaner
    generated_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 3. ColumnTransformer
    # O ColumnTransformer vai buscar as colunas pelos nomes definidos nas listas acima.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cols_numerical),
            ("cat", categorical_transformer, cols_categorical),
            ("gen_num", generated_numeric_transformer, cols_pedra + cols_binary),
        ],
        remainder="drop",  # Descarta explicitamente as colunas forbidden
        verbose_feature_names_out=False,
    )

    # 4. Pipeline Final
    model_pipeline = Pipeline(
        steps=[
            # Passos de Engenharia de Features (aplicados no DF inteiro antes do split de colunas)
            ("pedra_mapper", PedraMapper()),
            ("binary_cleaner", BinaryCleaner()),
            # Pré-processamento (Split + Scaling + Encoding)
            ("preprocessor", preprocessor),
            # Modelo
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

    try:
        # Carrega CSVs processados
        X_train = pd.read_csv(data_dir / "X_train.csv")
        y_train_df = pd.read_csv(data_dir / "y_train.csv")
        y_train = y_train_df.values.ravel()
    except FileNotFoundError:
        logger.error("Arquivos não encontrados. Execute 'src.preprocessing' primeiro.")
        return

    logger.info(f"Dados carregados. X_train: {X_train.shape}")

    logger.info("Construindo pipeline...")
    pipeline = create_pipeline(X_train)

    logger.info("Treinando modelo...")
    pipeline.fit(X_train, y_train)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)

    logger.info(f"Modelo salvo com sucesso em: {model_path}")


if __name__ == "__main__":
    run_training()
