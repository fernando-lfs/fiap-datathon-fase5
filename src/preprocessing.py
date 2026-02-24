import pandas as pd
import numpy as np
import re
from pathlib import Path
from src.utils import setup_logger

# Configuração do Logger
logger = setup_logger("preprocessing")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        logger.error(f"Arquivo não encontrado: {filepath}")
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        logger.info(f"Dataset carregado com sucesso. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.critical(f"Erro ao ler CSV: {e}")
        raise


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    def _clean(name):
        name = str(name).lower()
        name = re.sub(r"\s+", "_", name)
        name = name.replace(".", "").replace("/", "_")
        name = name.replace("ç", "c").replace("ã", "a").replace("õ", "o")
        name = (
            name.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        name = name.replace("ê", "e").replace("â", "a")
        return name

    old_cols = df.columns.tolist()
    df.columns = [_clean(col) for col in df.columns]
    logger.debug("Nomes de colunas padronizados para snake_case.")
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    keywords = [
        "inde",
        "iaa",
        "ieg",
        "ips",
        "ida",
        "ipv",
        "ian",
        "ponto_virada",
        "portug",
        "matem",
        "ingles",
    ]
    cols_to_convert = [c for c in df.columns if any(k in c for k in keywords)]

    for col in cols_to_convert:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[cols_to_convert] = df[cols_to_convert].fillna(0)
    logger.info(f"Conversão numérica aplicada em {len(cols_to_convert)} colunas.")
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    defas_cols = [c for c in df.columns if "defas" in c]

    if not defas_cols:
        logger.warning("Coluna de defasagem não encontrada. Target não criado.")
        return df

    target_col = defas_cols[0]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

    # Regra de Negócio: Defasagem < 0 indica atraso escolar (Risco)
    df["alvo"] = (df[target_col] < 0).astype(int)

    dist = df["alvo"].value_counts(normalize=True).to_dict()
    logger.info(f"Target 'alvo' criado baseada em '{target_col}'. Distribuição: {dist}")
    return df


def preprocess_pipeline(filepath: Path) -> pd.DataFrame:
    logger.info("Iniciando pipeline de pré-processamento...")
    df = load_data(filepath)
    df = clean_column_names(df)
    df = convert_numeric_columns(df)
    df = create_target_variable(df)

    processed_path = get_project_root() / "data" / "processed" / "dataset_limpo.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Dataset processado salvo em: {processed_path}")
    return df


if __name__ == "__main__":
    root = get_project_root()
    raw_data = root / "data" / "raw" / "dataset_pede_passos.csv"
    preprocess_pipeline(raw_data)
