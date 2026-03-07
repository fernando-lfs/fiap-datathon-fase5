import pandas as pd
import pytest
from pathlib import Path

# Imports dos módulos do projeto
import src.preprocessing
import src.train
import src.evaluate


@pytest.fixture
def mock_project_root(tmp_path, monkeypatch):
    """
    Fixture de isolamento de ambiente.

    Objetivo:
    Redireciona as operações de arquivo para um diretório temporário (`tmp_path`),
    garantindo que o teste de integração não suje a pasta real `data/` ou `app/model`
    do projeto, nem sobrescreva o modelo de produção.
    """
    # Cria estrutura de pastas no diretório temporário
    d = tmp_path
    (d / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (d / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (d / "app" / "model").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)

    # Função que retorna o caminho temporário
    def mock_return():
        return d

    # Aplica o monkeypatch nas funções que definem caminhos absolutos
    monkeypatch.setattr(src.train, "get_project_root", mock_return)
    monkeypatch.setattr(src.evaluate, "get_project_root", mock_return)

    return d


def test_full_pipeline_execution(mock_project_root):
    """
    Teste de Integração End-to-End (Smoke Test).

    Cenário:
    Simula o ciclo de vida completo do MLOps em um ambiente controlado:
    1. Ingestão (Criação de CSV Raw dummy).
    2. Processamento (Limpeza, Feature Engineering, Split).
    3. Treinamento (Fit do modelo, Serialização .joblib).
    4. Avaliação (Carga do modelo, Predição em lote).

    Critério de Sucesso:
    O pipeline deve rodar do início ao fim sem lançar exceções, gerando
    os artefatos esperados (arquivos csv e joblib).
    """
    root = mock_project_root
    raw_file = root / "data" / "raw" / "dataset_pede_passos.csv"

    # 1. CRIAÇÃO DE DADOS DUMMY (RAW)
    # Criamos dados com variação suficiente para evitar warnings de variância zero no Scaler
    df_dummy = pd.DataFrame(
        {
            "RA": [f"RA-{i}" for i in range(20)],
            "Fase": [1] * 20,
            "Turma": ["A"] * 20,
            "Nome": [f"Aluno {i}" for i in range(20)],
            "Gênero": ["Menina", "Menino"] * 10,
            "Instituição de ensino": ["Publica", "Privada"] * 10,
            "Pedra 20": ["Ametista", "Quartzo"] * 10,
            "Pedra 21": ["Ametista", "Quartzo"] * 10,
            "Pedra 22": ["Ametista"] * 20,  # Será ignorada (forbidden)
            "Indicado": ["Sim", "Não"] * 10,
            "Atingiu PV": ["Sim", "Não"] * 10,
            "Ponto Virada": ["Sim", "Não"] * 10,
            "Indicado Bolsa": ["Sim", "Não"] * 10,
            "IAA": [8.5, 5.5] * 10,
            "IEG": [7.0, 6.0] * 10,
            "IPS": [7.5, 7.5] * 10,
            "IDA": [6.0, 5.0] * 10,
            "IPP": [7.0, 7.0] * 10,
            "IPV": [7.0, 7.0] * 10,
            "Matemática": ["8,5", "5,0"] * 10,  # String PT-BR para testar conversão
            "Português": ["8,5", "5,0"] * 10,
            "Inglês": ["8,5", "5,0"] * 10,
            "Defas": [-1, 0] * 10,  # Gera ALVO 1 e 0
        }
    )
    df_dummy.to_csv(raw_file, index=False)

    # 2. EXECUÇÃO DO PREPROCESSING
    try:
        df = src.preprocessing.load_dataset(raw_file)
        df = src.preprocessing.create_target(df)
        src.preprocessing.save_split_data(df, root / "data")
    except Exception as e:
        pytest.fail(f"Falha no Preprocessing: {e}")

    # Verifica se arquivos processados foram criados
    assert (root / "data" / "processed" / "X_train.csv").exists()
    assert (root / "data" / "processed" / "y_train.csv").exists()

    # 3. EXECUÇÃO DO TREINAMENTO
    try:
        src.train.run_training()
    except Exception as e:
        pytest.fail(f"Falha no Training: {e}")

    # Verifica se o modelo foi salvo
    model_path = root / "app" / "model" / "pipeline.joblib"
    assert model_path.exists()

    # 4. EXECUÇÃO DA AVALIAÇÃO
    try:
        # Apenas executa para garantir que não quebra (Smoke Test)
        src.evaluate.evaluate_model()
    except Exception as e:
        pytest.fail(f"Falha no Evaluate: {e}")
