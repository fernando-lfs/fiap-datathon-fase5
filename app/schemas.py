from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class AlunoInput(BaseModel):
    # Identificadores e Categóricas
    genero: str = Field(..., description="Gênero do aluno (ex: Menina, Menino)")
    instituicao_de_ensino: str = Field(
        ..., description="Tipo de instituição (ex: Escola Pública)"
    )

    # Histórico (Pedras)
    pedra_20: str = Field(..., description="Classificação Pedra em 2020 (ex: Ametista)")
    pedra_21: str = Field(..., description="Classificação Pedra em 2021 (ex: Ágata)")
    # pedra_22 removida pois não usamos no treino para evitar leakage futuro,
    # mas se o front-end enviar, o pipeline ignora. Vamos manter opcional ou remover.

    # Indicadores e Notas (Floats)
    # O modelo espera floats. O JSON deve vir com ponto (ex: 8.5)
    iaa: float = Field(..., description="Indicador de Auto Avaliação")
    ieg: float = Field(..., description="Indicador de Engajamento")
    ips: float = Field(..., description="Indicador Psicossocial")
    ida: float = Field(..., description="Indicador de Aprendizagem")
    ipp: float = Field(..., description="Indicador Psicopedagógico")
    ipv: float = Field(..., description="Indicador de Ponto de Virada")

    matem: float = Field(..., description="Nota de Matemática")
    portug: float = Field(..., description="Nota de Português")
    ingles: float = Field(None, description="Nota de Inglês (pode ser nulo)")

    # Binários (Texto 'Sim'/'Não' ou 'S'/'N')
    indicado: str = Field(..., description="Indicado para bolsa? (Sim/Não)")
    atingiu_pv: str = Field(..., description="Atingiu Ponto de Virada? (Sim/Não)")
    ponto_virada: str = Field(..., description="Ponto de Virada 2020/2021 (Sim/Não)")
    indicado_bolsa: str = Field(..., description="Indicado Bolsa 2022 (Sim/Não)")

    # Campos extras que podem vir, mas o modelo ignora (definidos como opcionais para não quebrar)
    ian: Optional[float] = Field(
        None, description="Indicador de Adequação ao Nível (Ignorado pelo modelo)"
    )
    fase: Optional[int] = Field(None, description="Fase atual")
    turma: Optional[str] = Field(None, description="Turma")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "genero": "Menina",
                "instituicao_de_ensino": "Escola Pública",
                "pedra_20": "Ametista",
                "pedra_21": "Ágata",
                "iaa": 8.5,
                "ieg": 7.2,
                "ips": 6.8,
                "ida": 5.5,
                "ipp": 7.0,
                "ipv": 7.2,
                "matem": 6.0,
                "portug": 7.5,
                "ingles": 5.0,
                "indicado": "Não",
                "atingiu_pv": "Não",
                "ponto_virada": "Não",
                "indicado_bolsa": "Não",
            }
        }
    )


class PredicaoOutput(BaseModel):
    risco_defasagem: bool = Field(..., description="True se houver risco de defasagem")
    probabilidade_risco: float = Field(
        ..., description="Probabilidade calculada pelo modelo (0-1)"
    )
    mensagem: str = Field(..., description="Mensagem explicativa para o usuário")
