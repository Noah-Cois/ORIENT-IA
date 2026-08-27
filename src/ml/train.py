"""
src/ml/train.py
================
Entraînement du modèle d'orientation ISPM (RandomForest + OneHotEncoder).

Ce module expose entrainer_modele(), utilisée aussi bien :
- en ligne de commande (python -m src.ml.train), pour un entraînement manuel,
- automatiquement par predict.py au premier appel si le .pkl est absent
  (ex: déploiement Streamlit Cloud, où models/*.pkl n'est pas versionné sur Git).
"""

import os
import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def _localiser_racine_projet() -> str:
    """Remonte de src/ml -> src -> racine du projet, quel que soit le point d'appel."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def entrainer_modele(project_root: str = None, verbose: bool = True) -> str:
    """
    Entraîne le pipeline ML complet (OneHotEncoder + RandomForest) sur le
    dataset synthétique ISPM, et sauvegarde le modèle sur disque.

    Args:
        project_root: racine du projet. Si None, déduite automatiquement de
            l'emplacement de ce fichier (utile pour un appel externe depuis
            predict.py, où le calcul serait sinon relatif à predict.py).
        verbose: si True, affiche la progression (logs utiles en local ;
            peuvent être coupés lors d'un entraînement silencieux au démarrage
            d'une app Streamlit).

    Returns:
        str: chemin absolu du fichier .pkl généré.

    Raises:
        FileNotFoundError: si le dossier ou le fichier de données source
            sont introuvables.
    """
    if project_root is None:
        project_root = _localiser_racine_projet()

    def _log(msg):
        if verbose:
            print(msg)

    # 1. Localisation du dataset
    synthetic_dir = os.path.join(project_root, 'data', 'synthetic')
    if not os.path.exists(synthetic_dir):
        synthetic_dir = os.path.join(project_root, 'synthetic')

    if not os.path.exists(synthetic_dir):
        raise FileNotFoundError(
            f"Le dossier 'synthetic' est introuvable. Vérifie son emplacement par rapport à {project_root}"
        )

    all_csv_files = glob.glob(os.path.join(synthetic_dir, '*.csv'))
    _log(f"Fichiers détectés dans synthetic : {len(all_csv_files)} fichiers trouvés.")

    # 2. Chargement du dataset principal
    target_df_path = os.path.join(synthetic_dir, 'profils_etudiants_synthetiques.csv')
    if not os.path.exists(target_df_path):
        target_df_path = os.path.join(synthetic_dir, 'candidat_filiere.csv')

    if not os.path.exists(target_df_path):
        raise FileNotFoundError(f"Le fichier de données principal est introuvable dans {synthetic_dir}")

    df = pd.read_csv(target_df_path)
    _log(f"Dataset principal chargé : {os.path.basename(target_df_path)} ({len(df)} lignes)")

    # 3. Définition de la cible et des features exploitables
    target_col = (
        'filiere_recommandee' if 'filiere_recommandee' in df.columns
        else ('filiere_associee' if 'filiere_associee' in df.columns else df.columns[-1])
    )

    features = [
        'serie', 'moyenne_generale', 'matieres_fortes',
        'matieres_faibles', 'centres_interet', 'competences', 'serie_ou_filiere'
    ]
    available_features = [col for col in features if col in df.columns]

    df_clean = df.dropna(subset=available_features + [target_col]).copy()
    _log(f"Lignes exploitables après nettoyage des valeurs manquantes : {len(df_clean)}")

    X = df_clean[available_features]
    y = df_clean[target_col]

    # NOTE : le test `dtype == 'object'` ne fonctionne pas ici car pandas charge
    # les colonnes texte en dtype 'string' (backend pyarrow), pas 'object'.
    # On utilise donc is_numeric_dtype pour distinguer proprement numérique / catégoriel.
    categorical_cols = [col for col in available_features if not pd.api.types.is_numeric_dtype(df_clean[col])]
    numeric_cols = [col for col in available_features if pd.api.types.is_numeric_dtype(df_clean[col])]

    _log(f"\nColonnes détectées comme catégorielles : {categorical_cols}")
    _log(f"Colonnes détectées comme numériques : {numeric_cols}")

    # 4. Pipeline de Machine Learning
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    # 5. Entraînement et évaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    _log("--- Rapport de performance du modèle global ---")
    if verbose:
        print(classification_report(y_test, y_pred, zero_division=0))

    # 6. Sauvegarde du modèle
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ispm_orientation_model.pkl')

    joblib.dump(model_pipeline, model_path)
    _log(f"Modèle global sauvegardé avec succès dans {model_path} !")

    return model_path


if __name__ == "__main__":
    entrainer_modele()