import argparse
import os
import pandas as pd

from model_pipeline import (
    prepare_data,
    save_prepared_data,
    load_prepared_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_FILE = "adult_cleaned.csv"
PREPARED_FILE = "data/processed/prepared_data.joblib"


def main():

    parser = argparse.ArgumentParser(description="Adult Census Income - Pipeline ML")

    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "all"],
        default="all",
        help="Étape à exécuter",
    )

    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest", "both"],
        default="both",
        help="Modèle à utiliser",
    )

    parser.add_argument("--data", default=DATA_FILE, help="Chemin du fichier CSV")

    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------
    # ETAPE 1 : PREPARATION
    # ------------------------------------------

    if args.step in ["prepare", "all"]:

        print("\n===== PREPARATION DES DONNEES =====")

        prepared_data = prepare_data(args.data)

        save_prepared_data(prepared_data, PREPARED_FILE)

        print("Préparation terminée avec succès.")

        if args.step == "prepare":
            return

    # ------------------------------------------
    # CHARGEMENT DES DONNEES PREPAREES
    # ------------------------------------------

    X_train = None
    X_test = None
    y_train = None
    y_test = None
    preprocessor = None

    if args.step in ["train", "evaluate", "all"]:

        prepared_data = load_prepared_data(PREPARED_FILE)

        X_train, X_test, y_train, y_test, preprocessor = prepared_data

    # ------------------------------------------
    # ETAPE 2 : ENTRAINEMENT
    # ------------------------------------------

    if args.step in ["train", "all"]:

        print("\n===== ENTRAINEMENT =====")

        if args.model == "both":
            models_to_train = ["logistic", "random_forest"]
        else:
            models_to_train = [args.model]

        for model_name in models_to_train:

            print(f"\nEntraînement : {model_name}")

            model = train_model(X_train, y_train, preprocessor, model_name)

            save_model(model, f"models/{model_name}.joblib")

        print("\nEntraînement terminé.")

        if args.step == "train":
            return

    # ------------------------------------------
    # ETAPE 3 : EVALUATION
    # ------------------------------------------

    if args.step in ["evaluate", "all"]:

        print("\n===== EVALUATION =====")

        if args.model == "both":
            models_to_evaluate = ["logistic", "random_forest"]
        else:
            models_to_evaluate = [args.model]

        all_results = []

        for model_name in models_to_evaluate:

            print(f"\nEvaluation : {model_name}")

            model = load_model(f"models/{model_name}.joblib")

            results = evaluate_model(model, X_test, y_test)

            results["model"] = model_name
            all_results.append(results)

        results_df = pd.DataFrame(all_results)

        results_df.to_csv("results/evaluation_results.csv", index=False)

        print("\n===== COMPARAISON DES MODELES =====")
        print(results_df.to_string(index=False))

        print("\nRésultats sauvegardés dans results/.")


if __name__ == "__main__":
    main()
