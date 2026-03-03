import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils import setup_logger

logger = setup_logger("preprocessing")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza os nomes das colunas para snake_case e remove acentos comuns.
    """
    replacements = {
        "gênero": "genero",
        "instituição_de_ensino": "instituicao_de_ensino",
        "inglês": "ingles",
        "nº_av": "n_av",
        "matemática": "matem",
        "português": "portug",
        "fase_ideal": "fase_ideal",
        "ano_ingresso": "ano_ingresso",
    }

    new_cols = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.columns = new_cols
    df = df.rename(columns=replacements)
    return df


def convert_brazilian_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas numéricas formatadas como string (PT-BR) para float.
    Ex: "8,5" -> 8.5
    Força a conversão (errors='coerce') para garantir que a coluna vire float,
    transformando valores inválidos em NaN.
    """
    # Lista de colunas que sabemos que são texto e NÃO devem ser convertidas
    text_cols = [
        "ra",
        "nome",
        "turma",
        "genero",
        "instituicao_de_ensino",
        "pedra_20",
        "pedra_21",
        "pedra_22",
        "avaliador1",
        "rec_av1",
        "avaliador2",
        "rec_av2",
        "avaliador3",
        "rec_av3",
        "avaliador4",
        "rec_av4",
        "rec_psicologia",
        "indicado",
        "atingiu_pv",
        "destaque_ieg",
        "destaque_ida",
        "destaque_ipv",
        "fase_ideal",
    ]

    for col in df.columns:
        # Pula se for coluna de texto conhecida ou se já for numérico
        if col in text_cols or pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Tenta converter objetos (strings) para números
        if df[col].dtype == "object":
            try:
                # 1. Converte para string para garantir que métodos .str funcionem
                series_str = df[col].astype(str)

                # 2. Substitui formatação PT-BR
                clean_series = series_str.str.replace(".", "", regex=False).str.replace(
                    ",", ".", regex=False
                )

                # 3. Tenta converter para numérico
                # errors='coerce' é vital: se houver um valor sujo (ex: "-"), vira NaN,
                # mas o resto da coluna vira float com sucesso.
                df[col] = pd.to_numeric(clean_series, errors="coerce")

                logger.debug(f"Coluna processada para numérico: {col}")
            except Exception as e:
                logger.warning(f"Erro ao processar coluna {col}: {e}")

    return df


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Carrega o dataset CSV, normaliza colunas e corrige tipos numéricos.
    """
    if not file_path.exists():
        logger.error(f"Arquivo não encontrado: {file_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    try:
        # Lê o CSV como string inicialmente para evitar conversões erradas automáticas
        df = pd.read_csv(file_path, dtype=str)
        logger.info(f"Dataset carregado: {df.shape[0]} linhas.")

        # 1. Normaliza nomes das colunas
        df = normalize_columns(df)

        # 2. Converte tipos numéricos (PT-BR -> Float)
        df = convert_brazilian_numbers(df)

        # 3. Converte colunas que deveriam ser inteiros
        int_cols = ["fase", "ano_nasc", "idade_22", "ano_ingresso"]
        for col in int_cols:
            if col in df.columns:
                # Primeiro converte para float (para lidar com NaNs) depois para Int se possível
                # Aqui mantemos float se tiver NaN, ou preenchemos com 0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        return df
    except Exception as e:
        logger.critical(f"Erro ao carregar dataset: {e}")
        raise


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável alvo 'ALVO' baseada na coluna 'defas' e REMOVE a origem.
    """
    if "defas" not in df.columns:
        raise ValueError("Coluna 'defas' necessária para criar o target.")

    # Garante que defas seja numérico antes de comparar
    df["defas"] = pd.to_numeric(df["defas"], errors="coerce").fillna(0)

    # 1. Criação do alvo binário
    # Defasagem negativa indica atraso (ex: -1, -2)
    df["ALVO"] = df["defas"].apply(lambda x: 1 if x < 0 else 0)

    # 2. PREVENÇÃO DE DATA LEAKAGE
    df = df.drop(columns=["defas"])
    logger.info("Coluna 'defas' removida para prevenir Data Leakage.")

    return df


def save_split_data(df: pd.DataFrame, data_dir: Path):
    """
    Divide o dataset em Treino e Teste e salva em data/processed.
    """
    logger.info("Iniciando divisão de dados (Split)...")

    if "ALVO" not in df.columns:
        raise ValueError("Coluna ALVO não encontrada para split.")

    X = df.drop(columns=["ALVO"])
    y = df["ALVO"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    logger.info(f"Arquivos salvos em: {processed_dir}")
    logger.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "raw" / "dataset_pede_passos.csv"

    try:
        logger.info("--- Iniciando Preparação dos Dados ---")
        df = load_dataset(raw_path)
        df = create_target(df)
        save_split_data(df, root / "data")
        logger.info("--- Preparação Concluída com Sucesso ---")
    except Exception as e:
        logger.critical(f"Falha na preparação dos dados: {e}")
