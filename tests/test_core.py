import pandas as pd
import pytest
from src.preprocessing import clean_column_names, create_target_variable
from src.transformers import PedraMapper, BinaryCleaner


def test_clean_column_names():
    # Testa conversão de espaços, acentos e caixa alta
    df = pd.DataFrame({"INDE 22": [1], "Pedra 20": ["A"], "Média Port.": [5]})
    df_clean = clean_column_names(df)

    assert "inde_22" in df_clean.columns
    assert "pedra_20" in df_clean.columns
    assert "media_port" in df_clean.columns


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
            "outra_coluna": ["Sim", "Não", "S", "N"],  # Não deve mudar (sem keyword)
        }
    )
    cleaner = BinaryCleaner()
    df_trans = cleaner.transform(df)

    assert df_trans["indicado_bolsa"].tolist() == [1, 0, 1, 0]
    # Verifica se não alterou colunas fora da lista de keywords
    assert df_trans["outra_coluna"].tolist() == ["Sim", "Não", "S", "N"]


def test_create_target_variable():
    # Defasagem negativa (-1) = Atraso = Alvo 1
    # Defasagem positiva ou zero = Em dia = Alvo 0
    df = pd.DataFrame({"defas": [-1, 0, -2, 1]})
    df_target = create_target_variable(df)
    assert df_target["alvo"].tolist() == [1, 0, 1, 0]
