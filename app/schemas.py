from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal


class AlunoInput(BaseModel):
    """
    Schema de entrada de dados para a API de Predição.

    Define os contratos de dados esperados pelo modelo de Machine Learning.
    Campos históricos (como Pedra 2020/2021) são definidos como opcionais
    para permitir a avaliação de alunos recém-ingressos na associação,
    onde o modelo assumirá valores padrão (0/Sem Pedra).
    """

    # Identificadores e Categóricas
    genero: Literal["Menina", "Menino"] = Field(..., description="Gênero do aluno.")

    instituicao_de_ensino: str = Field(
        ..., description="Tipo de instituição (ex: Escola Pública, Rede Decisão)"
    )

    # Histórico de Classificação (Opcionais para suportar alunos novos)
    pedra_20: Optional[Literal["Quartzo", "Ágata", "Ametista", "Topázio"]] = Field(
        None, description="Classificação Pedra em 2020 (Deixar nulo se aluno novo)"
    )
    pedra_21: Optional[Literal["Quartzo", "Ágata", "Ametista", "Topázio"]] = Field(
        None, description="Classificação Pedra em 2021 (Deixar nulo se aluno novo)"
    )

    # Indicadores Psicossociais e Acadêmicos (Obrigatórios - Ano Corrente/Recente)
    iaa: float = Field(
        ..., ge=0, le=10, description="Indicador de Auto Avaliação (0-10)"
    )
    ieg: float = Field(..., ge=0, le=10, description="Indicador de Engajamento (0-10)")
    ips: float = Field(..., ge=0, le=10, description="Indicador Psicossocial (0-10)")
    ida: float = Field(..., ge=0, le=10, description="Indicador de Aprendizagem (0-10)")
    ipp: float = Field(..., ge=0, le=10, description="Indicador Psicopedagógico (0-10)")
    ipv: float = Field(
        ..., ge=0, le=10, description="Indicador de Ponto de Virada (0-10)"
    )

    # Notas Escolares
    matem: float = Field(..., ge=0, le=10, description="Nota de Matemática (0-10)")
    portug: float = Field(..., ge=0, le=10, description="Nota de Português (0-10)")
    ingles: Optional[float] = Field(
        None, ge=0, le=10, description="Nota de Inglês (0-10, opcional)"
    )

    # Variáveis Binárias e de Processo
    indicado: Literal["Sim", "Não"] = Field(
        ..., description="Indicado para bolsa? (Sim/Não)"
    )
    atingiu_pv: Literal["Sim", "Não"] = Field(
        ..., description="Atingiu Ponto de Virada? (Sim/Não)"
    )
    ponto_virada: Literal["Sim", "Não"] = Field(
        ..., description="Ponto de Virada 2020/2021 (Sim/Não)"
    )
    indicado_bolsa: Literal["Sim", "Não"] = Field(
        ..., description="Indicado Bolsa 2022 (Sim/Não)"
    )

    # Campos Estruturais Opcionais (Metadados não utilizados na inferência direta)
    ian: Optional[float] = Field(None, description="Indicador de Adequação ao Nível")
    fase: Optional[int] = Field(None, description="Fase atual")
    turma: Optional[str] = Field(None, description="Turma")
    ano_ingresso: Optional[int] = Field(2022, description="Ano de ingresso")
    ano_nasc: Optional[int] = Field(0, description="Ano de nascimento")
    ra: Optional[str] = Field("API_REQ", description="Registro do Aluno (RA)")
    nome: Optional[str] = Field("API_REQ", description="Nome do Aluno")
    n_av: Optional[int] = Field(0, description="Número de avaliações realizadas")
    pedra_22: Optional[str] = Field(None, description="Pedra 2022 (Target/Futuro)")
    inde_22: Optional[float] = Field(None, description="INDE 2022 (Target/Futuro)")
    cg: Optional[int] = Field(None, description="Classificação Geral")
    cf: Optional[int] = Field(None, description="Classificação Fase")
    ct: Optional[int] = Field(None, description="Classificação Turma")

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
    """
    Schema de saída da API de Predição.
    Retorna o risco calculado e uma mensagem pedagógica contextualizada.
    """

    risco_defasagem: bool = Field(
        ..., description="True se houver risco de defasagem, False caso contrário."
    )
    probabilidade_risco: float = Field(
        ..., description="Probabilidade calculada pelo modelo (0.0 a 1.0)."
    )
    mensagem: str = Field(
        ..., description="Mensagem explicativa com recomendação pedagógica."
    )
