# 🎓 Datathon: Passos Mágicos - Previsão de Risco Escolar

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Container-2496ed?style=for-the-badge&logo=docker)
![Pytest](https://img.shields.io/badge/Pytest-Testing-yellow?style=for-the-badge&logo=pytest)

> **Pós Tech - Machine Learning Engineering | FIAP**

Este projeto apresenta uma solução de **Machine Learning** desenvolvida para a **Associação Passos Mágicos**, visando identificar precocemente alunos com alto risco de defasagem escolar.

A arquitetura implementa um pipeline robusto de classificação, desde a engenharia de features focada em indicadores psicossociais até o deploy produtivo via **FastAPI**, garantindo intervenções pedagógicas proativas e baseadas em dados.

---

## 🚀 Funcionalidades e Diferenciais

*   **Pipeline Anti-Leakage:** Estratégia rigorosa de engenharia de features que remove variáveis do ano corrente (2022) para evitar vazamento de dados, garantindo que o modelo aprenda apenas com o histórico (2020-2021).
*   **Monitoramento de Drift:** Implementação de logs dedicados (`drift_data.csv`) na API para monitorar as entradas em produção, facilitando a detecção de mudanças no perfil dos alunos.
*   **API Inteligente:** Endpoint de inferência construído com **FastAPI**, utilizando validação estrita de tipos e intervalos (0-10) via **Pydantic**, além de fornecer mensagens de retorno com contexto pedagógico.
*   **Qualidade de Código:** Suíte de testes unitários e de integração (`pytest`) cobrindo desde a limpeza de dados até a resposta da API, com cobertura superior a 80%.
*   **Containerização Segura:** Dockerfile otimizado utilizando usuário não-root (`appuser`) e imagem base `slim`, seguindo as melhores práticas de segurança em MLOps.
*   **Reprodutibilidade:** Gerenciamento de dependências via **Poetry** e serialização do pipeline completo (incluindo pré-processamento) com `joblib`.

---

## 🏗️ Arquitetura e Decisões Técnicas (ADR)

| Componente | Escolha Técnica | Justificativa (Why?) |
| :--- | :--- | :--- |
| **Modelo Baseline** | **Regressão Logística** | Escolha mandatória para estabelecimento de baseline. Oferece alta interpretabilidade dos pesos das features (ex: impacto do `IEG` no risco) e eficiência computacional. |
| **Métrica Principal** | **Recall (Sensibilidade)** | No contexto social, o custo de um Falso Negativo (não identificar um aluno em risco) é crítico. Priorizamos cobrir a maioria dos casos vulneráveis (~92% de Recall). |
| **Pipeline** | **Scikit-Learn Pipeline** | Garante que o pré-processamento (imputação, scaling, one-hot encoding) aplicado no treino seja idêntico na inferência, eliminando erros de transformação. |
| **API** | **FastAPI** | Performance assíncrona e geração automática de documentação (Swagger UI), essencial para consumo por outros sistemas da ONG. |
| **Feature Eng.** | **Transformers Customizados** | Criação de classes como `PedraMapper` para tratar a ordinalidade das classificações (Quartzo < Ágata < Ametista < Topázio) sem perder a hierarquia. |
| **Estratégia de Dados** | **Foco no Dataset Principal** | Durante o *Data Understanding*, optou-se por não utilizar arquivos do diretório "Bases Antigas", interpretando-os como legados/depreciados. A prioridade foi garantir a integridade dos dados (consistência 2020-2022) e a robustez da arquitetura MLOps, em vez de aumentar a complexidade de engenharia de dados com fontes não-padronizadas. |

---

## 📘 Conexão com o Negócio (PEDE)

A seleção de variáveis do modelo não foi aleatória; ela reflete os insights dos **Relatórios PEDE (2020-2022)** contidos no dataset principal consolidado da Associação Passos Mágicos:

1.  **IEG (Indicador de Engajamento):** Priorizado como feature chave, pois o relatório de 2022 aponta o engajamento (entrega de lições, participação) como o "termômetro" mais sensível para prever a queda de desempenho acadêmico.
2.  **Histórico de Pedras:** A evolução da classificação (ex: queda de Ametista para Ágata) foi modelada para capturar tendências de longo prazo, alinhando-se à visão longitudinal da ONG.
3.  **Indicadores Psicossociais (IPS/IPP):** Incluídos para garantir que o modelo considere não apenas notas, mas o bem-estar emocional do aluno, respeitando a abordagem holística da Passos Mágicos.

---

## ⚡ Guia de Instalação e Execução

### Pré-requisitos
*   **Docker** (Recomendado para execução isolada).
*   **Python 3.11+** e **Poetry** (Para desenvolvimento local).
*   Arquivo `.env` na raiz (opcional, ver `app/config.py` para defaults).

### 1. Clonar o Repositório
O primeiro passo é obter o código-fonte em sua máquina local.

```bash
git clone <url-do-repositorio>
cd passos-magicos-datathon
```

### 2. Configuração do Ambiente
Você pode executar o projeto via **Docker** (Recomendado para avaliação rápida) ou **Localmente** (Para desenvolvimento).

#### Opção A: Via Docker (Produção/Avaliação)
A solução é agnóstica ao ambiente. Para rodar a API containerizada:

```bash
# 1. Construir a Imagem
docker build -t passos-magicos-api .

# 2. Rodar o Container
docker run -p 8000:8000 passos-magicos-api
```
*Acesse a documentação interativa em:* [http://localhost:8000/docs](http://localhost:8000/docs)

#### Opção B: Execução Local (Desenvolvimento)
Recomendado se você deseja rodar o pipeline de treinamento passo a passo.

**Passo 1: Instalar Dependências**
```bash
# Instala as dependências definidas no pyproject.toml
poetry install
```

**Passo 2: Executar o Pipeline de Dados e Treino**
Utilize o comando `poetry run` para garantir a execução dentro do ambiente virtual isolado:

```bash
# 1. Pré-processamento -> Gera data/processed/*.csv
poetry run python -m src.preprocessing

# 2. Treinamento -> Gera app/model/pipeline.joblib
poetry run python -m src.train

# 3. Avaliação -> Exibe métricas no console
poetry run python -m src.evaluate
```

**Passo 3: Iniciar a API**
```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Dica:** Caso prefira ativar o shell manualmente (sem usar `poetry run` repetidamente), ative o ambiente virtual criado pelo Poetry (ex: `.venv\Scripts\activate` no Windows ou `source .venv/bin/activate` no Linux/Mac).

---

## 🔌 Documentação da API

Abaixo, uma visão geral de todos os endpoints disponíveis.

| Método | Endpoint | Descrição |
| :--- | :--- | :--- |
| `POST` | **/predict** | **Principal:** Recebe dados históricos do aluno e retorna a probabilidade de risco de defasagem com interpretação pedagógica. |
| `GET` | **/model/info** | Retorna metadados do modelo (versão, tipo, features) para auditoria. |
| `GET` | **/health** | Health Check para monitoramento de disponibilidade da aplicação. |
| `GET` | **/** | Redireciona para a documentação Swagger UI. |

### Detalhamento do Endpoint de Predição

#### Predição de Risco (`POST /predict`)
Recebe indicadores acadêmicos e psicossociais dos anos anteriores para prever o risco no ano corrente.

**Exemplo de Requisição (Body):**
```json
{
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
  "indicado_bolsa": "Não"
}
```

**Exemplo de Resposta (Sucesso):**
```json
{
  "risco_defasagem": true,
  "probabilidade_risco": 0.8245,
  "mensagem": "CRÍTICO: Risco muito alto de defasagem. Intervenção pedagógica imediata recomendada."
}
```

---

## 📂 Estrutura do Projeto

```text
project-root/
├── app/                        # Aplicação API
│   ├── main.py                 # Endpoint e ciclo de vida da API
│   ├── schemas.py              # Contratos de dados (Pydantic)
│   ├── config.py               # Configurações globais
│   └── model/                  # Pipeline serializado (.joblib)
├── src/                        # Core de Machine Learning
│   ├── preprocessing.py        # Limpeza e tratamento inicial
│   ├── feature_engineering.py  # Transformers customizados (PedraMapper, BinaryCleaner)
│   ├── train.py                # Treinamento do modelo
│   ├── evaluate.py             # Avaliação de métricas
│   └── utils.py                # Utilitários de Log
├── tests/                      # Testes Unitários e de Integração
├── data/                       # Dados (Raw e Processed - ignorados no git)
├── logs/                       # Logs de aplicação e drift
├── Dockerfile                  # Receita da imagem Docker
├── pyproject.toml              # Configuração do Poetry
└── README.md                   # Documentação do Projeto
```

---

## 📈 Resultados Obtidos

## 📈 Resultados Obtidos (Baseline)

Os resultados abaixo refletem o **Baseline** do projeto (Regressão Logística). Diferente de abordagens que utilizam dados do futuro (notas de 2022) para inflar métricas artificialmente, este projeto priorizou a **prevenção rigorosa de Data Leakage**, utilizando apenas o histórico (2020-2021) para prever o risco em 2022.

| Métrica | Valor (Teste) | Análise de Negócio |
| :--- | :--- | :--- |
| **Recall (Classe de Risco)** | **67%** | **KPI Principal:** O modelo identifica corretamente 2 em cada 3 alunos que realmente precisam de ajuda. No contexto social, priorizamos minimizar os Falsos Negativos (alunos deixados para trás). |
| **Precision (Classe de Risco)** | **74%** | Quando o modelo alerta para risco, ele está correto em 74% das vezes, garantindo que a equipe pedagógica não desperdice recursos excessivos com alarmes falsos. |
| **Acurácia Global** | **60%** | A métrica reflete a dificuldade de separar classes linearmente apenas com dados históricos. É o ponto de partida ideal para futuras iterações com modelos não-lineares (ex: XGBoost). |

> **Nota Técnica:** O *trade-off* atual favorece o Recall da classe de risco (1), alinhado à missão da Associação Passos Mágicos de atuar preventivamente.

---

## ☁️ Próximos Passos

Para evolução do projeto visando maior escala e robustez:

1.  **Experimentação de Modelos:** Testar algoritmos baseados em árvores (Random Forest, XGBoost) para capturar relações não-lineares complexas entre os indicadores psicossociais.
2.  **Cloud Deployment:** Implantar a imagem Docker em serviços gerenciados (AWS ECS ou Google Cloud Run) para alta disponibilidade.
3.  **Dashboard de Monitoramento:** Conectar os logs de drift (`drift_data.csv`) a uma ferramenta de visualização (Streamlit ou Grafana) para acompanhar a distribuição das notas e indicadores em tempo real.