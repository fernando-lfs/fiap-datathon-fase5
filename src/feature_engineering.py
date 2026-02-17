import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_processed_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    return pd.read_csv(filepath)


def map_pedras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma as classificações de Pedra (Categorico Ordinal) em Numérico.
    Lógica: Quartzo (1) -> Ágata (2) -> Ametista (3) -> Topázio (4).
    """
    pedra_map = {
        "quartzo": 1,
        "ágata": 2,
        "agata": 2,  # Prevenção contra falta de acento
        "ametista": 3,
        "topázio": 4,
        "topazio": 4,
    }

    cols_pedra = [c for c in df.columns if "pedra" in c]

    for col in cols_pedra:
        # Normaliza para minúsculo antes de mapear
        df[col] = df[col].astype(str).str.lower().map(pedra_map).fillna(0).astype(int)

    return df


def clean_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas Sim/Não para 1/0.
    """
    binary_map = {"sim": 1, "não": 0, "nao": 0, "s": 1, "n": 0}

    # Identifica colunas que contêm 'Sim' ou 'Não'
    # Focamos nas conhecidas do dataset
    target_cols = ["indicado_bolsa", "ponto_virada", "atingiu_pv", "indicado"]

    for col in df.columns:
        if any(x in col for x in target_cols):
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .map(binary_map)
                    .fillna(0)
                    .astype(int)
                )

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas que não serão usadas no modelo (IDs, Textos longos, Vazamentos).
    """
    # Colunas a remover
    drop_cols = [
        "nome",
        "ra",  # Identificadores
        "defas",
        "fase_ideal",  # Origem do Target (Vazamento direto se mantidas)
        "cg",
        "cf",
        "ct",  # Rankings (são consequências, não causas)
        "turma",  # Muito granular para o baseline
    ]

    # Remove colunas de texto livre (Recomendações)
    drop_cols += [
        c for c in df.columns if "rec_" in c or "destaque_" in c or "avaliador" in c
    ]

    # Remove colunas que não existem no DF atual para evitar erro
    drop_cols = [c for c in drop_cols if c in df.columns]

    df = df.drop(columns=drop_cols)

    return df


def run_feature_engineering(input_path: Path):
    print("Iniciando Feature Engineering...")
    df = load_processed_data(input_path)

    # 1. Transformações Determinísticas
    df = map_pedras(df)
    df = clean_binary_features(df)
    df = select_features(df)

    # 2. Separação X (Features) e y (Target)
    if "alvo" not in df.columns:
        raise ValueError("Coluna 'alvo' não encontrada no dataset.")

    X = df.drop(columns=["alvo"])
    y = df["alvo"]

    # 3. Split Treino/Teste (80/20)
    # Stratify é crucial aqui devido ao desbalanceamento (601 vs 259)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Salvamento
    output_dir = get_project_root() / "data" / "processed"
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"Feature Engineering concluído.")
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    print(f"Arquivos salvos em: {output_dir}")

    # Retorna colunas para validação
    return X_train.columns.tolist()


if __name__ == "__main__":
    root = get_project_root()
    input_file = root / "data" / "processed" / "dataset_limpo.csv"

    try:
        cols = run_feature_engineering(input_file)
        print("\nFeatures Selecionadas para o Modelo:")
        print(cols)
    except Exception as e:
        print(f"Erro: {e}")
