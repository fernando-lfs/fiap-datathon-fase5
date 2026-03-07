from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Centraliza as configurações da aplicação e variáveis de ambiente.

    Padrão: 12-Factor App.
    As configurações são carregadas prioritariamente de variáveis de ambiente (OS),
    seguidas pelo arquivo .env, e por fim os valores padrão definidos aqui.
    """

    # Metadados da API
    PROJECT_NAME: str = "API Passos Mágicos"
    VERSION: str = "1.0.0"

    # Caminhos Absolutos (Baseados na localização deste arquivo)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = BASE_DIR / "app" / "model" / "pipeline.joblib"

    # Configuração de Observabilidade
    LOG_LEVEL: str = "INFO"

    # Configuração do Pydantic V2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # 'ignore': Permite rodar em ambientes (ex: Kubernetes/Docker) que injetam
        # variáveis extras sem quebrar a aplicação.
        extra="ignore",
    )


# Singleton: Instância única importada por toda a aplicação
settings = Settings()
