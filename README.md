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
*   **API Performática:** Endpoint de inferência construído com **FastAPI**, utilizando validação estrita de tipos via **Pydantic** para garantir a integridade dos dados de entrada.
*   **Qualidade de Código:** Suíte de testes unitários e de integração (`pytest`) cobrindo desde a limpeza de dados até a resposta da API.
*   **Containerização Segura:** Dockerfile otimizado utilizando usuário não-root (`appuser`) e imagem base `slim`, seguindo as melhores práticas de segurança em MLOps.
*   **Reprodutibilidade:** Gerenciamento de dependências via **Poetry** e serialização do pipeline completo (incluindo pré-processamento) com `joblib`.

---

## 🏗️ Arquitetura e Decisões Técnicas (ADR)

| Componente | Escolha Técnica | Justificativa (Why?) |
| :--- | :--- | :--- |
| **Modelo Baseline** | **Regressão Logística** | Escolha mandatória para estabelecimento de baseline. Oferece alta interpretabilidade dos pesos das features (ex: impacto do `IEG` no risco) e eficiência computacional. |
| **Métrica Principal** | **Recall (Sensibilidade)** | No contexto social, o custo de um Falso Negativo (não identificar um aluno em risco) é crítico. Priorizamos cobrir a maioria dos casos vulneráveis (~83% de Recall). |
| **Pipeline** | **Scikit-Learn Pipeline** | Garante que o pré-processamento (imputação, scaling, one-hot encoding) aplicado no treino seja idêntico na inferência, eliminando erros de transformação. |
| **API** | **FastAPI** | Performance assíncrona e geração automática de documentação (Swagger UI), essencial para consumo por outros sistemas da ONG. |
| **Feature Eng.** | **Transformers Customizados** | Criação de classes como `PedraMapper` para tratar a ordinalidade das classificações (Quartzo < Ágata < Ametista < Topázio) sem perder a hierarquia. |

---

## ⚡ Guia de Instalação e Execução

### Pré-requisitos
*   **Docker** (Recomendado para execução isolada).
*   **Python 3.11+** e **Poetry** (Para desenvolvimento local).

### 1. Clonar o Repositório
```bash
git clone <url-do-repositorio>
cd passos-magicos-datathon
```

### 2. Configuração do Ambiente

#### Opção A: Via Docker (Recomendado)
A solução é agnóstica ao ambiente. Para rodar a API containerizada:

```bash
# 1. Construir a Imagem
docker build -t passos-magicos-api .

# 2. Rodar o Container
docker run -p 8000:8000 passos-magicos-api
```
*Acesse a documentação interativa em:* [http://localhost:8000/docs](http://localhost:8000/docs)

#### Opção B: Execução Local (Desenvolvimento)
Para rodar o pipeline e a API diretamente na máquina:

```bash
# 1. Instalar dependências
poetry install

# 2. Ativar ambiente virtual
poetry shell

# 3. Executar API
uvicorn app.main:app --reload
```

---

## 🔌 Documentação da API

Abaixo, os endpoints disponíveis na aplicação.

| Método | Endpoint | Descrição |
| :--- | :--- | :--- |
| `POST` | **/predict** | **Principal:** Recebe dados históricos do aluno e retorna a probabilidade de risco de defasagem. |
| `GET` | **/health** | Health Check para monitoramento de disponibilidade da aplicação. |
| `GET` | **/** | Redireciona para a documentação Swagger UI. |

### Detalhamento do Endpoint de Predição

#### Predição de Risco (`POST /predict`)
Recebe indicadores acadêmicos e psicossociais dos anos anteriores para prever o risco no ano corrente.

**Exemplo de Requisição (Body):**
```json
{
  "genero": "Menina",
  "ano_ingresso": 2018,
  "instituicao_de_ensino": "Escola Pública",
  "pedra_20": "Ametista",
  "pedra_21": "Ágata",
  "pedra_22": "Quartzo",
  "n_av": 4,
  "iaa": 8.5,
  "ieg": 7.2,
  "ips": 6.8,
  "ida": 5.5,
  "matem": 6.0,
  "portug": 7.5,
  "ingles": 5.0,
  "indicado": "Não",
  "atingiu_pv": "Não",
  "ipv": 7.2
}
```

**Exemplo de Resposta (Sucesso):**
```json
{
  "risco_defasagem": true,
  "probabilidade_risco": 0.7845,
  "mensagem": "ALERTA: Alto risco de defasagem. Intervenção pedagógica recomendada."
}
```

---

## 📂 Estrutura do Projeto

```text
project-root/
├── app/                        # Aplicação API
│   ├── main.py                 # Endpoint e ciclo de vida da API
│   ├── schemas.py              # Contratos de dados (Pydantic)
│   └── model/                  # Pipeline serializado (.joblib)
├── src/                        # Core de Machine Learning
│   ├── preprocessing.py        # Limpeza e tratamento inicial
│   ├── feature_engineering.py  # Seleção de features e prevenção de Leakage
│   ├── train.py                # Treinamento do modelo
│   ├── evaluate.py             # Avaliação de métricas
│   ├── transformers.py         # Transformers customizados (PedraMapper, BinaryCleaner)
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

O modelo Baseline (Regressão Logística) foi otimizado para maximizar a detecção de alunos em situação de vulnerabilidade educacional.

| Métrica | Valor Aprox. | Descrição |
| :--- | :--- | :--- |
| **Recall (Risco)** | **~83%** | Capacidade do modelo de identificar corretamente os alunos que realmente terão defasagem. |
| **Precision** | **Variável** | Mantida em nível aceitável, equilibrando o número de falsos alertas. |

> **Nota de Negócio:** O foco em Recall garante que a Associação Passos Mágicos atue preventivamente na maioria dos casos críticos, cumprindo sua missão social de não deixar nenhum aluno para trás.

---

## ☁️ Próximos Passos

Para evolução do projeto visando maior escala e robustez:

1.  **Experimentação de Modelos:** Testar algoritmos baseados em árvores (Random Forest, XGBoost) para capturar relações não-lineares complexas entre os indicadores psicossociais.
2.  **Cloud Deployment:** Implantar a imagem Docker em serviços gerenciados (AWS ECS ou Google Cloud Run) para alta disponibilidade.
3.  **Dashboard de Monitoramento:** Conectar os logs de drift (`drift_data.csv`) a uma ferramenta de visualização (Streamlit ou Grafana) para acompanhar a distribuição das notas e indicadores em tempo real.