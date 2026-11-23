
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def test_model_pipeline_exists():
    """Verifie la presence du module de Machine Learning."""
    assert (BASE_DIR / "model_pipeline.py").exists()


def test_main_exists():
    """Verifie la presence du script principal."""
    assert (BASE_DIR / "main.py").exists()


def test_dataset_exists():
    """Verifie la presence du jeu de donnees nettoye."""
    assert (BASE_DIR / "adult_cleaned.csv").exists()
