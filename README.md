# Datathon: Passos Mágicos - Previsão de Risco de Defasagem Escolar

## 1. Visão Geral do Projeto
**Objetivo:** Desenvolver uma solução de Machine Learning capaz de identificar precocemente alunos da Associação Passos Mágicos com alto risco de defasagem escolar (queda de desempenho ou desengajamento), permitindo intervenções pedagógicas proativas.

**Solução Proposta:** 
Um modelo preditivo (Regressão Logística) treinado com dados históricos (2020-2022), focado em indicadores comportamentais e psicossociais, exposto via API REST (FastAPI) e empacotado em Docker para fácil distribuição.

**Impacto de Negócio:**
O modelo prioriza a **Sensibilidade (Recall)**, garantindo que a maioria dos alunos em risco seja identificada (Recall de ~83% no baseline), minimizando o erro de deixar um aluno vulnerável sem assistência.

**Stack Tecnológica:**
*   **Linguagem:** Python 3.11
*   **ML:** Scikit-learn, Pandas, Numpy
*   **API:** FastAPI, Pydantic
*   **Gerenciamento:** Poetry
*   **Testes:** Pytest (Cobertura de testes unitários implementada)
*   **Container:** Docker

## 2. Estrutura do Projeto
```bash
project-root/
│
├── app/                        # Aplicação API
│   ├── main.py                 # Endpoint e ciclo de vida da API
│   ├── schemas.py              # Validação de dados (Pydantic)
│   └── model/                  # Pipeline treinado (.joblib)
│
├── src/                        # Pipeline de Machine Learning
│   ├── preprocessing.py        # Limpeza e tratamento inicial
│   ├── feature_engineering.py  # Seleção de features e prevenção de Leakage
│   ├── train.py                # Treinamento do modelo
│   ├── evaluate.py             # Avaliação de métricas de negócio
│   └── utils.py                # Utilitários gerais
│
├── tests/                      # Testes Unitários
├── data/                       # Dados (ignorados no git)
├── Dockerfile                  # Receita da imagem Docker
├── pyproject.toml              # Configuração do Poetry
└── README.md                   # Esta documentação
```

## 3. Instruções de Instalação e Deploy

### Pré-requisitos
*   Docker instalado
*   Git

### Executando com Docker (Recomendado)
A solução é agnóstica ao ambiente. Para rodar:

1.  **Clone o repositório:**
    ```bash
    git clone <seu-repo-url>
    cd passos-magicos-datathon
    ```

2.  **Construa a imagem:**
    ```bash
    docker build -t passos-magicos-api .
    ```

3.  **Execute o container:**
    ```bash
    docker run -p 8000:8000 passos-magicos-api
    ```

A API estará disponível em: `http://localhost:8000/docs`

### Executando Localmente (Desenvolvimento)
1.  Instale o Poetry: `pip install poetry`
2.  Instale as dependências: `poetry install`
3.  Ative o ambiente: `poetry shell`
4.  Execute a API: `uvicorn app.main:app --reload`

## 4. Pipeline de Machine Learning

O pipeline foi desenhado para evitar **Data Leakage** e focar em causalidade:

1.  **Pré-processamento:** Limpeza de nomes de colunas, conversão de tipos numéricos (tratamento de vírgula decimal PT-BR).
2.  **Feature Engineering:**
    *   Remoção de variáveis que compõem matematicamente o alvo (ex: `IAN`, `Fase Ideal`) para evitar vazamento.
    *   Mapeamento ordinal de Pedras (Quartzo=1 a Topázio=4).
    *   Seleção de features comportamentais (`IEG`, `IPS`, `IAA`) e notas (`Matemática`, `Português`).
3.  **Modelo:** Pipeline com `SimpleImputer`, `StandardScaler`, `OneHotEncoder` e `LogisticRegression` (com `class_weight='balanced'` para lidar com o desbalanceamento).

## 5. Exemplos de Uso da API

**Endpoint:** `POST /predict`

**Exemplo de Request (JSON):**
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

**Exemplo de Response:**
```json
{
  "risco_defasagem": true,
  "probabilidade_risco": 0.7845,
  "mensagem": "ALERTA: Alto risco de defasagem. Intervenção recomendada."
}
```

## 6. Defesa Técnica e Métricas

Optamos por priorizar o **Recall (Sensibilidade)** da classe positiva (Risco).
*   **Justificativa:** No contexto social, o custo de um Falso Negativo (não identificar um aluno que precisa de ajuda) é muito maior do que um Falso Positivo (oferecer ajuda extra a quem não precisa).
*   **Performance Atual:** O modelo atinge ~83% de Recall, garantindo alta cobertura dos alunos vulneráveis.
