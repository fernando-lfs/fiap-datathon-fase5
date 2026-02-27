import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils import setup_logger

logger = setup_logger("feature_engineering")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_processed_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    return pd.read_csv(filepath)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas que causam Data Leakage ou ruído.

    CRITÉRIO DE LEAKAGE:
    Estamos prevendo o risco de defasagem (target derivado de 2022).
    Portanto, qualquer métrica consolidada de 2022 (INDE, Pedra, Notas Finais)
    é uma resposta do futuro ("vazamento") e deve ser removida.
    O modelo deve prever o risco 2022 baseado no histórico (2020, 2021).
    """

    # 1. Identificadores e Metadados
    ids_and_meta = ["nome", "ra", "turma", "cg", "cf", "ct", "ano_nasc"]

    # 2. Data Leakage (Variáveis do Ano Alvo - 2022)
    # Se o aluno já tem INDE_22 ou PEDRA_22, o ano já acabou.
    leakage_2022 = [
        "defas",
        "fase_ideal",
        "fase_22",
        "idade_22",
        "ian_22",
        "inde_22",
        "pedra_22",
        "ipv_22",
        "iaa_22",
        "ieg_22",
        "ips_22",
        "ida_22",
        "ipp_22",
    ]

    # 3. Colunas de Texto Livre (NLP fora do escopo)
    text_cols = [
        c for c in df.columns if "rec_" in c or "destaque_" in c or "avaliador" in c
    ]

    drop_cols = ids_and_meta + leakage_2022 + text_cols

    # Interseção segura (apenas remove o que existe)
    cols_to_drop = [c for c in drop_cols if c in df.columns]

    df = df.drop(columns=cols_to_drop)

    logger.info(f"Colunas removidas: {len(cols_to_drop)}")
    logger.debug(f"Lista de removidas: {cols_to_drop}")
    return df


def run_feature_engineering(input_path: Path):
    logger.info("Iniciando Feature Engineering...")
    df = load_processed_data(input_path)

    # Seleção de Features
    df = select_features(df)

    if "alvo" not in df.columns:
        logger.error("Coluna 'alvo' não encontrada. Verifique o pré-processamento.")
        raise ValueError("Coluna 'alvo' ausente.")

    X = df.drop(columns=["alvo"])
    y = df["alvo"]

    # Split Estratificado (Mantém proporção de risco)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    output_dir = get_project_root() / "data" / "processed"
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    logger.info(f"Split concluído. Treino: {X_train.shape}, Teste: {X_test.shape}")
    return X_train.columns.tolist()


if __name__ == "__main__":
    root = get_project_root()
    input_file = root / "data" / "processed" / "dataset_limpo.csv"
    run_feature_engineering(input_file)
