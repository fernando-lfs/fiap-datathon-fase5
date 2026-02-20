# Imagem base leve do Python
FROM python:3.11-slim

# Define diretório de trabalho dentro do container
WORKDIR /app

# Variáveis de ambiente para evitar arquivos .pyc e logs em buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema (se necessário)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte do projeto para o container
# Copiamos pastas específicas para manter a estrutura
COPY app/ ./app/
COPY src/ ./src/
# Opcional: Copiar dados se necessário, mas idealmente o modelo já está treinado em app/model

# Expõe a porta 8000 (padrão do FastAPI/Uvicorn)
EXPOSE 8000

# Comando para iniciar a API quando o container rodar
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]