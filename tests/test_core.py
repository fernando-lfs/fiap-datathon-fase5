import pandas as pd
import numpy as np
from src.preprocessing import normalize_columns, create_target
from src.feature_engineering import PedraMapper, BinaryCleaner


def test_normalize_columns():
    """
    Testa a normalização de nomes de colunas.
    Objetivo: Garantir padronização (snake_case) e remoção de acentos/espaços
    para evitar erros de referência no Pandas.
    """
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
    """
    Testa o Transformer de Pedras isoladamente.
    Objetivo: Verificar se o mapeamento ordinal (Quartzo=1 ... Topázio=4)
    está correto e se valores desconhecidos viram 0.
    """
    df = pd.DataFrame({"pedra_20": ["Ametista", "Quartzo", "Topázio", "Desconhecida"]})
    mapper = PedraMapper()
    df_trans = mapper.transform(df)

    # Ametista=3, Quartzo=1, Topázio=4, Desconhecida=0 (fillna)
    expected = [3, 1, 4, 0]
    assert df_trans["pedra_20"].tolist() == expected


def test_binary_cleaner():
    """
    Testa o Transformer de Binários isoladamente.
    Objetivo: Garantir que variações de 'Sim'/'Não' sejam unificadas para 1/0.
    """
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
    """
    Testa a criação da variável ALVO e a prevenção de Data Leakage.

    Regra:
    - Defasagem < 0 (ex: -1) -> ALVO = 1 (Risco)
    - Defasagem >= 0 (ex: 0) -> ALVO = 0 (Sem Risco)
    - A coluna original 'defas' DEVE ser removida.
    """
    df = pd.DataFrame({"defas": [-1, 0, -2, 1]})
    df_target = create_target(df)

    assert "ALVO" in df_target.columns
    assert df_target["ALVO"].tolist() == [1, 0, 1, 0]

    # Garante que a coluna original 'defas' foi removida (Leakage)
    assert "defas" not in df_target.columns


def test_create_target_drops_nan():
    """
    Testa o tratamento de dados nulos no Target.
    Objetivo: Garantir que não treinamos o modelo com targets imputados ou incertos.
    Linhas com 'defas' nulo devem ser descartadas.
    """
    df = pd.DataFrame({"defas": [-1, 0, np.nan, None, ""]})

    # A função deve converter para numeric (coercing errors) e dropar NaNs
    df_target = create_target(df)

    # Esperamos apenas 2 linhas válidas (-1 e 0)
    assert len(df_target) == 2
    assert df_target["ALVO"].tolist() == [1, 0]
