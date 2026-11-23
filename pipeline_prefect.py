import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

from prefect import flow, task

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

FILES_TO_CHECK = [
    "model_pipeline.py",
    "main.py",
    "pipeline_prefect.py",
]


def run_command(command):
    """Execute une commande et signale les erreurs."""
    print("\nCommande :", " ".join(map(str, command)))
    subprocess.run(command, cwd=BASE_DIR, check=True)  # nosec B603


@task(name="Installation des dependances")
def install_dependencies():
    """Installe les dependances du projet."""
    run_command([PYTHON, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    """Formate les fichiers Python du projet."""
    run_command([PYTHON, "-m", "black", *FILES_TO_CHECK])


@task(name="Qualite du code")
def check_code_quality():
    """Verifie la qualite du code avec Pylint."""
    run_command([PYTHON, "-m", "pylint", "--errors-only", *FILES_TO_CHECK])


@task(name="Securite du code")
def check_security():
    """Analyse les vulnerabilites potentielles avec Bandit."""
    run_command([PYTHON, "-m", "bandit", "-q", *FILES_TO_CHECK])


@task(name="Tests unitaires")
def run_unit_tests():
    """Execute les tests unitaires avec Pytest."""
    run_command([PYTHON, "-m", "pytest", "-v"])


@flow(name="Code")
def code_flow():
    """Flow de controle et de validation du code."""
    install_dependencies()
    format_code()
    check_code_quality()
    check_security()
    run_unit_tests()


@task(name="Preparation des donnees")
def prepare_data():
    """Lance la preparation des donnees."""
    run_command([PYTHON, "main.py", "--step", "prepare"])


@task(name="Entrainement du modele")
def train_model():
    """Lance l'entrainement des deux modeles."""
    run_command([PYTHON, "main.py", "--step", "train", "--model", "both"])


@task(name="Sauvegarde du modele")
def save_model():
    """Verifie que les modeles entraines ont ete sauvegardes."""
    models_dir = BASE_DIR / "models"
    expected = ["logistic.joblib", "random_forest.joblib"]

    for filename in expected:
        path = models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Modele introuvable : {path}")
        print(f"Modele sauvegarde : {path}")


@task(name="Chargement du modele")
def load_model():
    """Verifie la presence des modeles avant evaluation."""
    models_dir = BASE_DIR / "models"
    expected = ["logistic.joblib", "random_forest.joblib"]

    for filename in expected:
        path = models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Impossible de charger : {path}")
        print(f"Modele disponible : {path}")


@task(name="Evaluation du modele")
def evaluate_model():
    """Lance l'evaluation des modeles."""
    run_command([PYTHON, "main.py", "--step", "evaluate", "--model", "both"])


@flow(name="All")
def all_flow():
    """Execute le pipeline complet de controle et de Machine Learning."""
    code_flow()
    prepare_data()
    train_model()
    save_model()
    load_model()
    evaluate_model()


@flow(name="Train")
def train_flow():
    """Prepare les donnees puis entraine les modeles."""
    prepare_data()
    train_model()
    save_model()


@flow(name="Evaluate")
def evaluate_flow():
    """Charge les modeles puis les evalue."""
    load_model()
    evaluate_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Adult Income avec Prefect")
    parser.add_argument(
        "--flow",
        choices=["all", "train", "entrainement", "evaluate", "code"],
        default="all",
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow in ["train", "entrainement"]:
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
    elif args.flow == "code":
        code_flow()
