import pandas as pd
import numpy as np
import re
from pathlib import Path
import os


def get_project_root() -> Path:
    """Retorna o caminho absoluto para a raiz do projeto."""
    # __file__ é o caminho deste arquivo (preprocessing.py)
    # .parent é a pasta src
    # .parent.parent é a raiz do projeto
    return Path(__file__).resolve().parent.parent


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Carrega o dataset de um arquivo CSV.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"O arquivo {filepath} não foi encontrado. Verifique se o caminho está correto."
        )

    try:
        # sep=',' é o padrão, mas explicitamos. encoding='utf-8' é boa prática.
        df = pd.read_csv(filepath, encoding="utf-8")
        return df
    except Exception as e:
        raise Exception(f"Erro ao ler o arquivo CSV: {e}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas para snake_case e remove caracteres especiais.
    """

    def _clean(name):
        name = str(name).lower()
        name = re.sub(r"\s+", "_", name)  # Espaços para _
        name = name.replace(".", "").replace("/", "_")
        # Remoção de acentos
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

    df.columns = [_clean(col) for col in df.columns]
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas numéricas que estão como string (com vírgula) para float.
    Detecta automaticamente colunas que parecem números formatados (ex: '5,783').
    """
    # Lista de colunas conhecidas que precisam de conversão baseada no Data Understanding
    # Adicionamos regex para capturar variações como 'inde_2022', 'inde_21', etc.
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
            # Remove pontos de milhar e troca vírgula decimal por ponto
            # Ex: "1.234,56" -> "1234.56"
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preenche NaN gerados pela conversão com 0 (assumindo que falta de nota = 0 para baseline)
    df[cols_to_convert] = df[cols_to_convert].fillna(0)

    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável alvo 'risco_defasagem'.
    Regra PEDE: IAN (Indicador de Adequação de Nível).
    Se defasagem < 0, o aluno está atrasado em relação à fase ideal.
    """
    # Procura por colunas de defasagem (ex: defasagem_2021, defas_2022, ou apenas defas)
    defas_cols = [c for c in df.columns if "defas" in c]

    if not defas_cols:
        print("AVISO: Coluna de defasagem não encontrada para criar o target.")
        return df

    # Prioriza a coluna mais recente ou genérica. Vamos usar a primeira encontrada para o MVP.
    target_col = defas_cols[0]

    # Garante numérico
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

    # Criação do Target Binário: 1 = Risco (Defasagem Negativa), 0 = Sem Risco
    df["alvo"] = (df[target_col] < 0).astype(int)

    return df


def preprocess_pipeline(filepath: Path) -> pd.DataFrame:
    """
    Executa o pipeline completo.
    """
    print(f"Iniciando pré-processamento do arquivo: {filepath}")

    df = load_data(filepath)
    df = clean_column_names(df)
    df = convert_numeric_columns(df)
    df = create_target_variable(df)

    # Salva o dataset processado para uso posterior (Feature Engineering/Treino)
    processed_path = get_project_root() / "data" / "processed" / "dataset_limpo.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Dataset processado salvo em: {processed_path}")

    return df


if __name__ == "__main__":
    # Configuração robusta de caminho
    project_root = get_project_root()
    raw_data_path = project_root / "data" / "raw" / "dataset_pede_passos.csv"

    try:
        df_proc = preprocess_pipeline(raw_data_path)
        print("Shape final:", df_proc.shape)
        print("Colunas:", df_proc.columns.tolist()[:10])  # Mostra as 10 primeiras
        if "alvo" in df_proc.columns:
            print("Distribuição do Alvo:")
            print(df_proc["alvo"].value_counts())
    except Exception as e:
        print(f"Erro fatal no pipeline: {e}")
