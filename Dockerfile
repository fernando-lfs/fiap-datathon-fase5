# Imagem base leve do Python
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema necessárias para compilação (se houver libs C)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python
# Fazemos isso ANTES de copiar o código para aproveitar o cache de camadas do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY app/ ./app/
COPY src/ ./src/

# Cria diretório de logs e ajusta permissões
# Isso é necessário porque o código tenta escrever em logs/app.log e logs/drift_data.csv
RUN mkdir -p logs

# Cria um usuário não-root para segurança
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expõe a porta
EXPOSE 8000

# Comando de execução
# --proxy-headers é importante se for rodar atrás de um Nginx/AWS ALB
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]