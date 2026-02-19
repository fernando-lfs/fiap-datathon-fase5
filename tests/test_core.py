import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.preprocessing import clean_column_names, create_target_variable
from src.feature_engineering import map_pedras, clean_binary_features
from src.train import create_pipeline


def test_clean_column_names():
    """Testa a padronização dos nomes das colunas (snake_case)."""
    # Cria um DataFrame sujo de exemplo
    df = pd.DataFrame({"INDE 22": [1], "Pedra 20": ["A"], "Nome do Aluno": ["X"]})
    df_clean = clean_column_names(df)

    assert "inde_22" in df_clean.columns
    assert "pedra_20" in df_clean.columns
    assert "nome_do_aluno" in df_clean.columns


def test_map_pedras():
    """Testa a conversão de Pedras (texto) para números."""
    df = pd.DataFrame({"pedra_20": ["Ametista", "Quartzo", "Topázio"]})
    df_mapped = map_pedras(df)

    # Verifica se converteu para números: Ametista=3, Quartzo=1, Topázio=4
    assert df_mapped["pedra_20"].tolist() == [3, 1, 4]


def test_clean_binary_features():
    """Testa a conversão de Sim/Não para 1/0."""
    df = pd.DataFrame({"indicado_bolsa": ["Sim", "Não", "S", "N"]})
    df_clean = clean_binary_features(df)

    assert df_clean["indicado_bolsa"].tolist() == [1, 0, 1, 0]


def test_create_target_variable():
    """Testa a criação da variável alvo baseada na defasagem."""
    # Simula dados para target: defasagem negativa = risco (1)
    df = pd.DataFrame({"defas": [-1, 0, -2, 1]})
    df_target = create_target_variable(df)

    # -1 e -2 são risco (1), 0 e 1 não são (0)
    assert df_target["alvo"].tolist() == [1, 0, 1, 0]


def test_create_pipeline_structure():
    """
    Testa se a função create_pipeline monta a estrutura correta do Scikit-Learn.
    Isso aumenta a cobertura do src/train.py sem precisar treinar o modelo.
    """
    # Cria um DataFrame dummy com as colunas esperadas pelo pipeline
    # É importante ter as colunas categóricas e algumas numéricas
    df_dummy = pd.DataFrame(
        {
            "genero": ["Menina", "Menino"],
            "instituicao_de_ensino": ["Publica", "Privada"],
            "iaa": [5.0, 6.0],
            "ieg": [6.0, 7.0],
            "ips": [7.0, 8.0],
            "ida": [8.0, 9.0],
            "matem": [5.0, 6.0],
            "portug": [5.0, 6.0],
            "ingles": [5.0, 6.0],
            "ipv": [5.0, 6.0],
            "pedra_20": [1, 2],  # Numéricas após feature engineering
            "pedra_21": [1, 2],
            "pedra_22": [1, 2],
        }
    )

    # Chama a função que cria o pipeline
    pipeline = create_pipeline(df_dummy)

    # 1. Verifica se o objeto retornado é um Pipeline do sklearn
    assert isinstance(pipeline, Pipeline)

    # 2. Verifica se tem os passos esperados (Preprocessor e Classificador)
    steps = [step[0] for step in pipeline.steps]
    assert "preprocessor" in steps
    assert "classifier" in steps

    # 3. Verifica se o classificador é o esperado (LogisticRegression)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
