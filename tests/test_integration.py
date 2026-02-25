import pandas as pd
import pytest
from pathlib import Path
import shutil
from src.feature_engineering import run_feature_engineering
from src.train import run_training
from src.evaluate import evaluate_model
from app.config import settings


# Fixture para criar um ambiente temporário de teste
@pytest.fixture(scope="module")
def setup_test_environment(tmp_path_factory):
    # Cria diretórios temporários para não sujar o projeto real
    temp_dir = tmp_path_factory.mktemp("data_test")
    raw_dir = temp_dir / "data" / "raw"
    processed_dir = temp_dir / "data" / "processed"
    model_dir = temp_dir / "app" / "model"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Cria um CSV dummy simulando o dataset_limpo.csv (pós-preprocessing)
    # Precisamos de dados suficientes para o train_test_split não falhar
    df_dummy = pd.DataFrame(
        {
            "genero": ["Menina", "Menino"] * 10,
            "instituicao_de_ensino": ["Publica", "Privada"] * 10,
            "pedra_20": ["Ametista", "Quartzo"] * 10,
            "pedra_21": ["Ametista", "Quartzo"] * 10,
            "pedra_22": ["Ametista", "Quartzo"] * 10,
            "indicado_bolsa": ["Sim", "Não"] * 10,
            "ponto_virada": ["Sim", "Não"] * 10,
            "atingiu_pv": ["Sim", "Não"] * 10,
            "indicado": ["Sim", "Não"] * 10,
            "iaa": [5.5, 8.5] * 10,
            "ieg": [6.0, 9.0] * 10,
            "ips": [7.5, 7.5] * 10,
            "ida": [5.0, 8.0] * 10,
            "matem": [5.0, 8.0] * 10,
            "portug": [5.0, 8.0] * 10,
            "ingles": [5.0, 8.0] * 10,
            "ipv": [5.0, 8.0] * 10,
            "alvo": [0, 1] * 10,  # Target balanceado
        }
    )

    input_file = processed_dir / "dataset_limpo.csv"
    df_dummy.to_csv(input_file, index=False)

    # Mockar (substituir) as funções que retornam caminhos no código original
    # para usarem nosso diretório temporário
    import src.feature_engineering
    import src.train
    import src.evaluate

    # Guarda as funções originais
    orig_root_fe = src.feature_engineering.get_project_root
    orig_root_tr = src.train.get_project_root
    orig_root_ev = src.evaluate.get_project_root

    # Substitui por lambda que retorna nosso temp_dir
    src.feature_engineering.get_project_root = lambda: temp_dir
    src.train.get_project_root = lambda: temp_dir
    src.evaluate.get_project_root = lambda: temp_dir

    yield temp_dir, input_file

    # Teardown: Restaura funções originais
    src.feature_engineering.get_project_root = orig_root_fe
    src.train.get_project_root = orig_root_tr
    src.evaluate.get_project_root = orig_root_ev


def test_full_pipeline_execution(setup_test_environment):
    """
    Teste de Fumaça (Smoke Test): Roda o pipeline inteiro para garantir
    que os scripts conversam entre si e geram os arquivos esperados.
    """
    temp_dir, input_file = setup_test_environment

    # 1. Testar Feature Engineering
    cols = run_feature_engineering(input_file)
    assert (temp_dir / "data" / "processed" / "X_train.csv").exists()
    assert (temp_dir / "data" / "processed" / "y_train.csv").exists()
    assert len(cols) > 0

    # 2. Testar Treinamento
    run_training()
    model_path = temp_dir / "app" / "model" / "pipeline.joblib"
    assert model_path.exists()

    # 3. Testar Avaliação
    # O evaluate apenas imprime no console, mas se não der erro, passou.
    try:
        evaluate_model()
    except Exception as e:
        pytest.fail(f"Evaluate falhou: {e}")
