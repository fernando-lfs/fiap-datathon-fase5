import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import setup_logger

logger = setup_logger("feature_engineering")


class PedraMapper(BaseEstimator, TransformerMixin):
    """
    Transformer customizado para converter a variável ordinal 'Pedra' em valores numéricos.

    Regra de Negócio (PEDE 2022):
    A classificação 'Pedra' segue uma hierarquia de desempenho educacional e engajamento:
    1. Quartzo (Iniciante/Baixo desempenho)
    2. Ágata (Em desenvolvimento)
    3. Ametista (Bom desempenho)
    4. Topázio (Alto desempenho/Referência)

    Este transformer preserva essa ordinalidade (1 < 2 < 3 < 4) para que modelos
    lineares e de árvore possam capturar a evolução do aluno.
    """

    def __init__(self):
        self.pedra_map = {
            "quartzo": 1,
            "ágata": 2,
            "agata": 2,  # Tratamento de variação sem acento
            "ametista": 3,
            "topázio": 4,
            "topazio": 4,  # Tratamento de variação sem acento
        }
        # Lista de colunas conhecidas de Pedra no dataset
        self.cols_pedra = ["pedra_20", "pedra_21", "pedra_22"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Aplica o mapeamento nas colunas de Pedra identificadas.
        Valores nulos ou desconhecidos são preenchidos com 0 (Sem classificação).
        """
        X = X.copy()
        for col in self.cols_pedra:
            if col in X.columns:
                # Converte para string, lowercase, mapeia e preenche nulos com 0
                X[col] = (
                    X[col]
                    .astype(str)
                    .str.lower()
                    .map(self.pedra_map)
                    .fillna(0)
                    .astype(int)
                )
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class BinaryCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer para padronização de variáveis booleanas textuais.

    Motivação:
    O dataset original possui inconsistências na entrada de dados manuais,
    apresentando variações como 'Sim', 'S', 's', 'Não', 'N', 'n'.

    Ação:
    Converte todas as variações para binário numérico:
    - 'Sim', 'S' -> 1
    - 'Não', 'N' -> 0
    """

    def __init__(self):
        self.binary_map = {"sim": 1, "não": 0, "nao": 0, "s": 1, "n": 0}
        # Keywords para identificar colunas binárias dinamicamente no dataset
        self.target_keywords = ["indicado", "ponto_virada", "atingiu_pv", "bolsa"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Varre as colunas do DataFrame e aplica a conversão naquelas que contêm
        as palavras-chave definidas em target_keywords.
        """
        X = X.copy()
        for col in X.columns:
            # Verifica se a coluna contém alguma das palavras-chave
            if any(k in col for k in self.target_keywords):
                # Aplica apenas se não for numérico (evita re-processar se já for int)
                if not pd.api.types.is_numeric_dtype(X[col]):
                    try:
                        X[col] = (
                            X[col]
                            .astype(str)
                            .str.lower()
                            .map(self.binary_map)
                            .fillna(0)
                            .astype(int)
                        )
                    except Exception as e:
                        logger.warning(f"Falha ao converter coluna binária {col}: {e}")
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features
