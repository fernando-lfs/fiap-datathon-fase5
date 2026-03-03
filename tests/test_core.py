import pandas as pd
import numpy as np
from src.preprocessing import normalize_columns, create_target
from src.feature_engineering import PedraMapper, BinaryCleaner


def test_normalize_columns():
    """Testa a normalização de nomes de colunas (snake_case e acentos)."""
    df = pd.DataFrame(
        {
            "INDE 22": [1],
            "Pedra 20": ["A"],
            "Média Port.": [5],
            "Instituição de Ensino": ["Pub"],
        }
    )
    df_clean = normalize_columns(df)

    # Verifica se as colunas esperadas existem após renomeação
    assert "inde_22" in df_clean.columns
    assert "pedra_20" in df_clean.columns
    # Verifica substituição específica definida no código
    assert "instituicao_de_ensino" in df_clean.columns


def test_pedra_mapper():
    """Testa o Transformer de Pedras isoladamente."""
    df = pd.DataFrame({"pedra_20": ["Ametista", "Quartzo", "Topázio", "Desconhecida"]})
    mapper = PedraMapper()
    df_trans = mapper.transform(df)

    # Ametista=3, Quartzo=1, Topázio=4, Desconhecida=0 (fillna)
    expected = [3, 1, 4, 0]
    assert df_trans["pedra_20"].tolist() == expected


def test_binary_cleaner():
    """Testa o Transformer de Binários isoladamente."""
    df = pd.DataFrame(
        {
            "indicado_bolsa": ["Sim", "Não", "S", "N"],
            "ponto_virada": ["sim", "nao", "s", "n"],
            "outra_coluna": ["Sim", "Não", "S", "N"],  # Não deve mudar (sem keyword)
        }
    )
    cleaner = BinaryCleaner()
    df_trans = cleaner.transform(df)

    # Verifica conversão 1/0
    assert df_trans["indicado_bolsa"].tolist() == [1, 0, 1, 0]
    assert df_trans["ponto_virada"].tolist() == [1, 0, 1, 0]

    # Verifica se não alterou colunas fora da lista de keywords
    assert df_trans["outra_coluna"].tolist() == ["Sim", "Não", "S", "N"]


def test_create_target():
    """Testa a criação da variável ALVO e remoção de Data Leakage."""
    # Defasagem negativa (-1) = Atraso = Alvo 1
    # Defasagem positiva ou zero = Em dia = Alvo 0
    df = pd.DataFrame({"defas": [-1, 0, -2, 1]})
    df_target = create_target(df)

    assert "ALVO" in df_target.columns
    assert df_target["ALVO"].tolist() == [1, 0, 1, 0]

    # Garante que a coluna original 'defas' foi removida (Leakage)
    assert "defas" not in df_target.columns
