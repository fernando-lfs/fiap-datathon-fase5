import pandas as pd
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger("preprocessing")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza os nomes das colunas para snake_case e remove acentos comuns
    para garantir compatibilidade entre o CSV (Treino) e a API (Inferência).

    Ex: 'Instituição de ensino' -> 'instituicao_de_ensino'
        'Pedra 20' -> 'pedra_20'
        'Nº Av' -> 'n_av'
    """
    # Mapeamento manual para casos com acentos ou caracteres especiais
    # As chaves devem corresponder ao estado após o .lower().replace(" ", "_")
    replacements = {
        "gênero": "genero",
        "instituição_de_ensino": "instituicao_de_ensino",
        "inglês": "ingles",
        "nº_av": "n_av",
        "matemática": "matem",
        "português": "portug",
        "fase_ideal": "fase_ideal",  # Garantia
        "ano_ingresso": "ano_ingresso",
    }

    # 1. Strip, Lowercase e substituição de espaços por underscore
    new_cols = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.columns = new_cols

    # 2. Substituição de termos acentuados específicos
    df = df.rename(columns=replacements)

    return df


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Carrega o dataset CSV e realiza limpezas estruturais básicas.
    """
    if not file_path.exists():
        logger.error(f"Arquivo não encontrado: {file_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(
            f"Dataset carregado com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas."
        )

        # Aplica normalização de colunas
        df = normalize_columns(df)

        return df
    except Exception as e:
        logger.critical(f"Erro ao carregar dataset: {e}")
        raise


def split_features_target(df: pd.DataFrame, target_col: str = "ALVO"):
    """
    Separa features (X) e target (y).
    """
    if target_col not in df.columns:
        logger.warning(f"Coluna alvo '{target_col}' não encontrada.")
        return df, None

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ==============================================================================
# BLOCO DE EXECUÇÃO (Adicionado para permitir teste direto via terminal)
# ==============================================================================
if __name__ == "__main__":
    # Define o caminho raiz do projeto
    root = Path(__file__).resolve().parent.parent

    # Define o caminho do arquivo RAW (ajuste o nome se necessário)
    raw_path = root / "data" / "raw" / "dataset_pede_passos.csv"

    # Caso esteja usando a amostra para teste, descomente a linha abaixo:
    # raw_path = root / "data" / "raw" / "dataset_pede_passos_amostra.csv"

    logger.info("--- Iniciando Teste de Pré-processamento ---")

    try:
        # 1. Tenta carregar e normalizar
        df = load_dataset(raw_path)

        # 2. Exibe as colunas para validar a normalização
        print("\n[SUCESSO] Colunas Normalizadas:")
        print(df.columns.tolist())

        print("\n[SUCESSO] Amostra dos dados:")
        print(df.head(3))

        logger.info("Teste de pré-processamento concluído com sucesso.")

    except FileNotFoundError:
        logger.error(f"Certifique-se de que o arquivo existe em: {raw_path}")
    except Exception as e:
        logger.critical(f"Falha no teste: {e}")
