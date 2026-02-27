import pandas as pd
import pytest
from pathlib import Path
import sklearn
import joblib

# Importamos os módulos para poder fazer o patch das funções de caminho
import src.feature_engineering
import src.train
import src.evaluate
from src.feature_engineering import run_feature_engineering
from src.train import run_training
from src.evaluate import evaluate_model

# Garante configuração consistente nos testes
sklearn.set_config(transform_output="pandas")


@pytest.fixture(scope="module")
def setup_test_environment(tmp_path_factory):
    """
    Cria um ambiente isolado com dados fake para testar o pipeline completo.
    """
    # Cria diretórios temporários
    temp_dir = tmp_path_factory.mktemp("data_test")
    raw_dir = temp_dir / "data" / "raw"
    processed_dir = temp_dir / "data" / "processed"
    model_dir = temp_dir / "app" / "model"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Cria CSV dummy simulando o dataset_limpo.csv
    # IMPORTANTE: O dataset deve conter colunas que NÃO são removidas pelo feature_engineering
    # e a coluna 'alvo' já criada pelo preprocessing.
    df_dummy = pd.DataFrame(
        {
            "genero": ["Menina", "Menino"] * 25,
            "instituicao_de_ensino": ["Publica", "Privada"] * 25,
            "pedra_20": ["Ametista", "Quartzo"] * 25,
            "pedra_21": ["Ametista", "Quartzo"] * 25,
            # pedra_22 removida propositalmente pois é leakage
            "indicado_bolsa": ["Sim", "Não"] * 25,
            "ponto_virada": ["Sim", "Não"] * 25,
            "atingiu_pv": ["Sim", "Não"] * 25,
            "indicado": ["Sim", "Não"] * 25,
            "n_av": [3, 4] * 25,
            "iaa": [5.5, 8.5] * 25,
            "ieg": [6.0, 9.0] * 25,
            "ips": [7.5, 7.5] * 25,
            "ida": [5.0, 8.0] * 25,
            "matem": [5.0, 8.0] * 25,
            "portug": [5.0, 8.0] * 25,
            "ingles": [5.0, 8.0] * 25,
            "ipv": [5.0, 8.0] * 25,
            "alvo": [0, 1] * 25,  # Target balanceado
        }
    )

    input_file = processed_dir / "dataset_limpo.csv"
    df_dummy.to_csv(input_file, index=False)

    # --- MONKEY PATCHING ---
    # Substituímos a função get_project_root dos módulos para apontar para o temp_dir
    # Isso impede que o teste escreva na pasta real do projeto

    orig_root_fe = src.feature_engineering.get_project_root
    orig_root_tr = src.train.get_project_root
    orig_root_ev = src.evaluate.get_project_root

    src.feature_engineering.get_project_root = lambda: temp_dir
    src.train.get_project_root = lambda: temp_dir
    src.evaluate.get_project_root = lambda: temp_dir

    yield temp_dir, input_file

    # Restaura os caminhos originais após o teste (Teardown)
    src.feature_engineering.get_project_root = orig_root_fe
    src.train.get_project_root = orig_root_tr
    src.evaluate.get_project_root = orig_root_ev


def test_full_pipeline_execution(setup_test_environment):
    """
    Smoke Test: Roda o pipeline inteiro (Feature Eng -> Train -> Evaluate)
    Verifica se os arquivos são gerados e se não há erros de execução.
    """
    temp_dir, input_file = setup_test_environment

    # 1. Feature Engineering
    # Deve gerar X_train, X_test, etc.
    cols = run_feature_engineering(input_file)
    assert (temp_dir / "data" / "processed" / "X_train.csv").exists()
    assert len(cols) > 0

    # 2. Treinamento
    # Deve gerar o pipeline.joblib
    run_training()
    model_path = temp_dir / "app" / "model" / "pipeline.joblib"
    assert model_path.exists()

    # Verifica se o modelo salvo é carregável
    pipeline = joblib.load(model_path)
    assert pipeline is not None

    # 3. Avaliação
    # Não deve lançar exceção
    try:
        evaluate_model()
    except Exception as e:
        pytest.fail(f"Evaluate falhou: {e}")
