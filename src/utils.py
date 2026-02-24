import logging
import sys
from pathlib import Path


def setup_logger(
    name: str, log_file: str = "app.log", level=logging.INFO
) -> logging.Logger:
    """
    Configura um logger padronizado para a aplicação.
    """
    # Formato do log: Data - Nome do Modulo - Nivel - Mensagem
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Handler para Console (Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Handler para Arquivo
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evita duplicidade de handlers se a função for chamada múltiplas vezes
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
