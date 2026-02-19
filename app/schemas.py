from pydantic import BaseModel, ConfigDict
from typing import Optional


class AlunoInput(BaseModel):
    genero: str
    ano_ingresso: int
    instituicao_de_ensino: str
    pedra_20: str
    pedra_21: str
    pedra_22: str
    n_av: int
    iaa: float
    ieg: float
    ips: float
    ida: float
    matem: float
    portug: float
    ingles: float
    indicado: str
    atingiu_pv: str
    ipv: float

    # Nova sintaxe do Pydantic V2
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "genero": "Menina",
                "ano_ingresso": 2018,
                "instituicao_de_ensino": "Escola Pública",
                "pedra_20": "Ametista",
                "pedra_21": "Ágata",
                "pedra_22": "Quartzo",
                "n_av": 4,
                "iaa": 8.5,
                "ieg": 7.2,
                "ips": 6.8,
                "ida": 5.5,
                "matem": 6.0,
                "portug": 7.5,
                "ingles": 5.0,
                "indicado": "Não",
                "atingiu_pv": "Não",
                "ipv": 7.2,
            }
        }
    )


class PredicaoOutput(BaseModel):
    risco_defasagem: bool
    probabilidade_risco: float
    mensagem: str
