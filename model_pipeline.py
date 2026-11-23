import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==================================================
# 1. PREPARATION DES DONNEES
# ==================================================


def prepare_data(file_path):
    """Charge et prépare le dataset Adult Census Income."""

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Harmonisation de la variable cible
    df["income"] = df["income"].str.replace(".", "", regex=False)

    # Suppression de education si elle existe
    if "education" in df.columns:
        df = df.drop(columns=["education"])

    X = df.drop(columns=["income"])
    y = df["income"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    capital_features = [
        col for col in ["capital-gain", "capital-loss"] if col in numeric_features
    ]

    other_numeric_features = [
        col for col in numeric_features if col not in capital_features
    ]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    capital_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = [
        ("numeric", numeric_pipeline, other_numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]

    if capital_features:
        transformers.append(("capital", capital_pipeline, capital_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Préparation terminée.")
    print("X_train :", X_train.shape)
    print("X_test :", X_test.shape)

    return X_train, X_test, y_train, y_test, preprocessor


# ==================================================
# 2. SAUVEGARDE ET CHARGEMENT DES DONNEES PREPAREES
# ==================================================


def save_prepared_data(data, file_path):
    """Sauvegarde les données préparées."""

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(data, file_path)

    print("Données préparées sauvegardées :", file_path)


def load_prepared_data(file_path):
    """Charge les données préparées."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Données préparées introuvables. " "Exécute d'abord --step prepare."
        )

    return joblib.load(file_path)


# ==================================================
# 3. ENTRAINEMENT DU MODELE
# ==================================================


def train_model(X_train, y_train, preprocessor, model_name):
    """Entraîne Logistic Regression ou Random Forest."""

    if model_name == "logistic":
        classifier = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        )

    else:
        raise ValueError("Modèle non reconnu.")

    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    model.fit(X_train, y_train)

    print("Entraînement terminé :", model_name)

    return model


# ==================================================
# 4. EVALUATION DU MODELE
# ==================================================


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle de classification."""

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=">50K", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=">50K", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, pos_label=">50K", zero_division=0),
    }

    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred, labels=["<=50K", ">50K"]))

    print("\nMétriques :")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results


# ==================================================
# 5. SAUVEGARDE ET CHARGEMENT DU MODELE
# ==================================================


def save_model(model, file_path):
    """Sauvegarde le modèle avec joblib."""

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)

    print("Modèle sauvegardé :", file_path)


def load_model(file_path):
    """Charge un modèle sauvegardé."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {file_path}. " "Exécute d'abord --step train."
        )

    model = joblib.load(file_path)

    print("Modèle chargé :", file_path)

    return model
