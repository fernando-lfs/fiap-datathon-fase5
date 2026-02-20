# Guia de Contribuição

Obrigado pelo interesse em contribuir para o projeto **Datathon: Case Passos Mágicos**. Este documento define as diretrizes técnicas para garantir a qualidade, consistência e reprodutibilidade do código em um ambiente de MLOps focado em impacto social.

## 1. Código de Conduta

Este projeto adota um ambiente de respeito e colaboração técnica. Nosso objetivo é entregar valor real para a Associação Passos Mágicos. Críticas construtivas são bem-vindas; desrespeito ou discriminação não serão tolerados.

## 2. Fluxo de Desenvolvimento (Git Flow)

Para manter a integridade da *branch* principal (`main`), siga este fluxo:

1.  **Fork** o repositório.
2.  Crie uma **Feature Branch** descritiva:
    ```bash
    git checkout -b feat/nova-feature-engenharia
    # ou
    git checkout -b fix/correcao-leakage
    ```
3.  Implemente suas mudanças.
4.  Realize o **Commit** seguindo as convenções (veja seção 6).
5.  Abra um **Pull Request (PR)** para a branch `main`.

## 3. Padrões de Engenharia de Software

*   **Linguagem:** Python 3.11+.
*   **Formatação:** O código deve seguir a **PEP 8**.
*   **Tipagem (Type Hinting):** É **obrigatório** o uso de tipagem estática nas assinaturas de funções e métodos, especialmente nos Schemas do Pydantic (`app/schemas.py`).
    *   *Correto:* `def preprocess_pipeline(filepath: Path) -> pd.DataFrame:`
    *   *Incorreto:* `def preprocess_pipeline(filepath):`
*   **Modularização:** Não coloque lógica de negócio em Notebooks. Toda a lógica deve residir em `src/` ou `app/`. Notebooks servem apenas para exploração (EDA).
*   **Configuração:** Não utilize caminhos *hardcoded* absolutos (ex: `C:/Users/...`). Utilize sempre `pathlib` e caminhos relativos à raiz do projeto.

## 4. Gerenciamento de Dependências

Este projeto utiliza **Poetry** como fonte da verdade.

1.  Para adicionar uma lib: `poetry add <nome-da-lib>`.
2.  **Nunca edite o `requirements.txt` manualmente.** Ele é um artefato gerado para o Docker.
3.  Se você alterou as dependências, **você deve atualizar o arquivo de requisitos**:
    ```bash
    poetry export --without-hashes --format=requirements.txt > requirements.txt
    ```

## 5. Diretrizes de MLOps e Data Science

Se sua contribuição envolve alterações no modelo ou nos dados, atente-se rigorosamente a estes pontos:

*   **Prevenção de Data Leakage (CRÍTICO):**
    *   É estritamente **proibido** incluir features que componham matematicamente a variável alvo (Defasagem).
    *   **Features Proibidas:** `IAN`, `INDE` (geral), `Fase Ideal`, `Ano Nascimento` (quando usado para calcular fase).
    *   O modelo deve ser **comportamental** e não matemático.
*   **Pipeline Scikit-Learn:**
    *   Qualquer transformação de dados (Imputer, Scaler, Encoder) deve estar dentro do `Pipeline` em `src/train.py`. Isso garante que o pré-processamento seja idêntico no treino e na inferência (API).
*   **Reprodutibilidade:**
    *   Mantenha a semente aleatória fixa (`random_state=42`) em divisões de treino/teste e inicialização de modelos.

## 6. Mensagens de Commit (Conventional Commits)

Siga o padrão: `<tipo>: <descrição breve no imperativo>`

*   `feat:` Nova funcionalidade (ex: `feat: adiciona endpoint de healthcheck`).
*   `fix:` Correção de bug (ex: `fix: remove feature com vazamento de dados`).
*   `docs:` Alterações na documentação.
*   `refactor:` Melhoria de código sem mudança de comportamento.
*   `test:` Adição ou correção de testes unitários.
*   `chore:` Configurações, dependências ou CI/CD.

## 7. Checklist de Validação (Antes do PR)

Garanta que o pipeline completo funciona localmente e que a qualidade do código foi mantida:

1.  **Pipeline de Dados:** O pré-processamento e a engenharia de features rodam sem erros?
    ```bash
    python -m src.preprocessing
    python -m src.feature_engineering
    ```
2.  **Treino e Avaliação:** O modelo treina e gera métricas honestas (Recall < 100%)?
    ```bash
    python -m src.train
    python -m src.evaluate
    ```
3.  **Testes Unitários:** A suíte de testes passa com cobertura aceitável?
    ```bash
    pytest --cov=src --cov=app tests/
    ```
4.  **Docker:** A imagem constrói sem erros?
    ```bash
    docker build -t passos-magicos-api .
    ```