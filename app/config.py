from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # Definição de caminhos e configurações globais
    PROJECT_NAME: str = "API Passos Mágicos"
    VERSION: str = "1.0.0"

    # Caminho base do projeto
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Caminho do Modelo
    MODEL_PATH: Path = BASE_DIR / "app" / "model" / "pipeline.joblib"

    # Configuração de Logs
    LOG_LEVEL: str = "INFO"

    # NOVA SINTAXE PYDANTIC V2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignora variáveis extras no .env
    )


# Instância única para ser importada
settings = Settings()
