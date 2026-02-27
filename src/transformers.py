import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PedraMapper(BaseEstimator, TransformerMixin):
    """
    Transformer customizado para converter a variável ordinal 'Pedra' em valores numéricos.
    Preserva a hierarquia: Quartzo (1) < Ágata (2) < Ametista (3) < Topázio (4).
    """

    def __init__(self):
        self.pedra_map = {
            "quartzo": 1,
            "ágata": 2,
            "agata": 2,
            "ametista": 3,
            "topázio": 4,
            "topazio": 4,
        }
        self.cols_pedra = ["pedra_20", "pedra_21", "pedra_22"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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


class BinaryCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer para padronização de variáveis booleanas textuais.
    Converte: 'Sim', 'S' -> 1 e 'Não', 'N' -> 0.
    """

    def __init__(self):
        self.binary_map = {"sim": 1, "não": 0, "nao": 0, "s": 1, "n": 0}
        # Keywords para identificar colunas binárias dinamicamente
        self.target_keywords = ["indicado", "ponto_virada", "atingiu_pv", "bolsa"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            # Verifica se a coluna contém alguma das palavras-chave
            if any(k in col for k in self.target_keywords):
                # Aplica apenas se não for numérico (evita re-processar se já for int)
                if not pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = (
                        X[col]
                        .astype(str)
                        .str.lower()
                        .map(self.binary_map)
                        .fillna(0)
                        .astype(int)
                    )
        return X
