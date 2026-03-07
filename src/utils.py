import logging
import sys
from pathlib import Path


def setup_logger(
    name: str, log_file: str = "app.log", level=logging.INFO
) -> logging.Logger:
    """
    Configura e retorna um logger padronizado para a aplicação.

    Estratégia de Observabilidade:
    - Console (StreamHandler): Para visualização em tempo real (stdout) e logs de container (Docker).
    - Arquivo (FileHandler): Para persistência e auditoria histórica em 'logs/'.

    Args:
        name (str): Nome do logger (geralmente __name__ do módulo).
        log_file (str): Nome do arquivo de saída.
        level (int): Nível de log (INFO, DEBUG, WARNING, etc).

    Returns:
        logging.Logger: Objeto logger configurado.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 1. Handler para Console (Docker logs / Stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 2. Handler para Arquivo (Persistência)
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evita duplicação de handlers se a função for chamada múltiplas vezes
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
