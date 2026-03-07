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
    Constrói o pipeline completo de processamento e modelagem.

    Estratégia de Pré-processamento:
    1. Numéricas: Imputação pela mediana (robusto a outliers) + Padronização (StandardScaler).
    2. Categóricas: Imputação de valor constante + OneHotEncoding.
    3. Customizados:
       - PedraMapper: Mapeamento ordinal das pedras (Quartzo < Ágata < Ametista < Topázio).
       - BinaryCleaner: Padronização de booleanos textuais (Sim/Não).

    Modelo:
    - LogisticRegression com class_weight='balanced' para lidar com o desbalanceamento
      natural das classes de risco.

    Args:
        X_train (pd.DataFrame): DataFrame de treino para inferência de tipos de colunas.

    Returns:
        Pipeline: Pipeline scikit-learn configurado e pronto para treino.
    """
    # 1. Definição de Grupos de Colunas
    ideal_categorical = ["genero", "instituicao_de_ensino"]
    ideal_pedra = ["pedra_20", "pedra_21"]
    ideal_binary = ["indicado", "atingiu_pv", "indicado_bolsa", "ponto_virada"]

    # ==============================================================================
    # PREVENÇÃO DE DATA LEAKAGE (ATUALIZADO)
    # ==============================================================================
    forbidden_cols = [
        "ra",
        "nome",
        "turma",
        "alvo",  # Identificadores
        "ian",
        "fase_ideal",
        "defas",  # Proxies diretos do alvo
        "inde_22",
        "cg",
        "cf",
        "ct",
        "pedra_22",  # Resultados futuros
        # NOVAS REMOÇÕES: Variáveis estruturais que enviesam o modelo
        "fase",
        "idade_22",
        "ano_nasc",
        "ano_ingresso",
    ]

    # Seleção dinâmica de colunas presentes no DataFrame
    cols_categorical = [c for c in ideal_categorical if c in X_train.columns]
    cols_pedra = [c for c in ideal_pedra if c in X_train.columns]
    cols_binary = [c for c in ideal_binary if c in X_train.columns]

    # Numéricas: Tudo que sobra, exceto as proibidas e as já selecionadas
    exclude_cols = cols_categorical + cols_pedra + cols_binary + forbidden_cols

    cols_numerical = [
        c
        for c in X_train.select_dtypes(include=["number"]).columns
        if c not in exclude_cols
    ]

    logger.info(f"Features Numéricas selecionadas: {len(cols_numerical)}")
    logger.info(f"Features Categóricas selecionadas: {len(cols_categorical)}")
    logger.debug(f"Colunas excluídas: {exclude_cols}")

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

    # 4. Pipeline Final
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
    """
    Orquestra o processo de treinamento do modelo.

    Etapas:
    1. Carrega dados processados (X_train, y_train).
    2. Instancia o pipeline via create_pipeline().
    3. Realiza o fit do modelo.
    4. Serializa o artefato final em app/model/pipeline.joblib.
    """
    root = get_project_root()
    data_dir = root / "data" / "processed"
    model_dir = root / "app" / "model"

    logger.info("Iniciando processo de treinamento...")

    try:
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
