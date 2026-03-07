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

# 1. Instala dependências do sistema necessárias para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Cria usuário não-root (appuser) ANTES de copiar arquivos
RUN adduser --disabled-password --gecos "" appuser

# 3. Instalação de dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copia o código fonte JÁ com as permissões corretas
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser src/ ./src/

# 5. Cria diretório de logs e ajusta permissão
RUN mkdir -p logs && chown -R appuser:appuser logs

# 6. Muda para o usuário seguro
USER appuser

# Expõe a porta padrão da API
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]