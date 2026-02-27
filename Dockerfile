# Imagem base leve do Python
FROM python:3.11-slim

# Metadados do projeto
LABEL maintainer="Time Passos Mágicos <datathon@fiap.com.br>"
LABEL description="API de Predição de Risco Escolar - Datathon FIAP"

# Define diretório de trabalho
WORKDIR /app

# Variáveis de ambiente
# PYTHONDONTWRITEBYTECODE: Evita criar arquivos .pyc desnecessários
# PYTHONUNBUFFERED: Garante que os logs da API apareçam instantaneamente no console
# PYTHONPATH: Adiciona o diretório raiz ao path para imports absolutos funcionarem
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Instala dependências do sistema necessárias para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalação de dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY app/ ./app/
COPY src/ ./src/

# Configuração de Permissões e Segurança
# 1. Cria usuário não-root (appuser) para não rodar a aplicação como root
# 2. Cria diretório de logs
# 3. Ajusta permissões recursivamente
RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p logs && \
    chown -R appuser:appuser /app
# Muda para o usuário seguro
USER appuser

# Expõe a porta padrão da API
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]