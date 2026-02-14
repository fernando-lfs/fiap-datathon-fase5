import os
from pathlib import Path

def create_structure():
    # Definição da estrutura baseada no requisitos.md
    directories = [
        "app/model",
        "src",
        "tests",
        "notebooks",
        "data/raw",       # Para guardar o csv original
        "data/processed"  # Para guardar dados limpos (feature engineering)
    ]

    files = [
        "app/main.py",
        "app/routes.py",
        "src/preprocessing.py",
        "src/feature_engineering.py",
        "src/train.py",
        "src/evaluate.py",
        "src/utils.py",
        "tests/test_preprocessing.py",
        "tests/test_model.py",
        "Dockerfile",
        "README.md",
        ".gitignore"
    ]

    # Criação dos diretórios
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        # Cria __init__.py em pastas de código python (app, src, tests)
        if directory.startswith(("app", "src", "tests")):
            (path / "__init__.py").touch()
        print(f"Diretório criado: {directory}")

    # Criação dos arquivos vazios
    for file in files:
        path = Path(file)
        if not path.exists():
            path.touch()
            print(f"Arquivo criado: {file}")
        else:
            print(f"Arquivo já existe: {file}")

    # Conteúdo básico do .gitignore para Python e ML
    gitignore_content = """
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.venv/
dist/
*.pkl
*.joblib
data/
!data/.gitkeep
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("Arquivo .gitignore configurado.")

if __name__ == "__main__":
    create_structure()