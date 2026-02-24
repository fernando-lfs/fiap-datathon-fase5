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
    Remove colunas que causam Data Leakage ou não são úteis para predição.
    """
    # Colunas de identificação e vazamento direto da fórmula de defasagem
    drop_cols = [
        "nome",
        "ra",
        "defas",
        "fase_ideal",
        "cg",
        "cf",
        "ct",
        "turma",
        "ian",
        "ian_22",
        "fase",
        "fase_22",
        "idade_22",
        "ano_nasc",
        "inde_22",
        "inde_21",
        "inde_20",  # INDE contém a resposta
    ]

    # Colunas de texto livre (NLP seria necessário, fora do escopo baseline)
    drop_cols += [
        c for c in df.columns if "rec_" in c or "destaque_" in c or "avaliador" in c
    ]

    # Remove apenas o que existe no DF
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    logger.info(f"Colunas removidas para evitar Leakage/Ruído: {len(cols_to_drop)}")
    return df


def run_feature_engineering(input_path: Path):
    logger.info("Iniciando Feature Engineering...")
    df = load_processed_data(input_path)

    # Nota: Não fazemos mais map_pedras ou clean_binary aqui.
    # Isso será feito pelo Pipeline dentro do modelo.

    df = select_features(df)

    if "alvo" not in df.columns:
        logger.error("Coluna 'alvo' não encontrada no dataset.")
        raise ValueError("Coluna 'alvo' não encontrada.")

    X = df.drop(columns=["alvo"])
    y = df["alvo"]

    # Stratify é crucial para classes desbalanceadas
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
