from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class AlunoInput(BaseModel):
    genero: str = Field(..., description="Gênero do aluno (ex: Menina, Menino)")
    ano_ingresso: int = Field(..., description="Ano de ingresso na associação")
    instituicao_de_ensino: str = Field(
        ..., description="Tipo de instituição (ex: Escola Pública)"
    )
    pedra_20: str = Field(..., description="Classificação Pedra em 2020")
    pedra_21: str = Field(..., description="Classificação Pedra em 2021")
    pedra_22: str = Field(..., description="Classificação Pedra em 2022")
    n_av: int = Field(..., description="Número de avaliações realizadas")
    iaa: float = Field(..., description="Indicador de Auto Avaliação")
    ieg: float = Field(..., description="Indicador de Engajamento")
    ips: float = Field(..., description="Indicador Psicossocial")
    ida: float = Field(..., description="Indicador de Aprendizagem")
    matem: float = Field(..., description="Nota de Matemática")
    portug: float = Field(..., description="Nota de Português")
    ingles: float = Field(..., description="Nota de Inglês")
    indicado: str = Field(..., description="Indicado para bolsa? (Sim/Não)")
    atingiu_pv: str = Field(..., description="Atingiu Ponto de Virada? (Sim/Não)")
    ipv: float = Field(..., description="Indicador de Ponto de Virada")

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
    risco_defasagem: bool = Field(..., description="True se houver risco de defasagem")
    probabilidade_risco: float = Field(
        ..., description="Probabilidade calculada pelo modelo (0-1)"
    )
    mensagem: str = Field(..., description="Mensagem explicativa para o usuário")
