import os
import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np

# Permet d'importer src.ml.train quel que soit le point d'entrée du script
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ml.train import entrainer_modele


def _assurer_modele_ml(model_path: str, project_root: str) -> None:
    """
    Vérifie que le modèle .pkl existe sur disque. S'il est absent (ex: premier
    démarrage sur Streamlit Cloud, où models/*.pkl n'est pas versionné sur Git
    car généré par train.py), on lance l'entraînement automatiquement au lieu
    de planter avec un FileNotFoundError.
    """
    if os.path.exists(model_path):
        return

    print(
        f"[INFO] Modèle introuvable à {model_path} — entraînement automatique en cours "
        "(premier démarrage ou déploiement sans .pkl versionné)..."
    )
    entrainer_modele(project_root=project_root, verbose=True)

    if not os.path.exists(model_path):
        # Sécurité : si entrainer_modele a réussi mais a écrit ailleurs (chemin
        # différent), on préfère un message explicite à un échec silencieux.
        raise FileNotFoundError(
            f"L'entraînement automatique s'est terminé mais le modèle reste introuvable à {model_path}."
        )


def predire_orientation_top3(profil_etudiant: dict) -> list:
    """
    Fonction généralisée pour l'Agent IA.
    Prend un dictionnaire contenant le profil de l'étudiant et retourne
    le Top 3 des filières recommandées avec leurs pourcentages de confiance.

    Exemple de profil attendu :
    {
        'serie': 'D',
        'moyenne_generale': 14.5,
        'matieres_fortes': 'Mathématiques, SVT',
        'matieres_faibles': 'Histoire-Géographie',
        'centres_interet': 'Nouvelles technologies',
        'competences': 'Analyse'
    }
    """
    # 1. Localisation dynamique robuste du modèle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    model_path = os.path.join(project_root, 'models', 'ispm_orientation_model.pkl')

    # 2. Vérifie l'existence du modèle, l'entraîne automatiquement sinon
    _assurer_modele_ml(model_path, project_root)

    # 3. Chargement du pipeline
    model_pipeline = joblib.load(model_path)

    # 4. Validation et préparation des données pour le modèle
    df_input = pd.DataFrame([profil_etudiant])

    # 5. Extraction des prédictions et probabilités
    classifier = model_pipeline.named_steps['classifier']
    classes = classifier.classes_

    if hasattr(model_pipeline, "predict_proba"):
        probas = model_pipeline.predict_proba(df_input)[0]
        top_indices = np.argsort(probas)[::-1]

        top_3_results = []
        for i in range(min(3, len(classes))):
            idx = top_indices[i]
            top_3_results.append({
                "filiere": str(classes[idx]),
                "confiance": round(float(probas[idx] * 100), 2)
            })

        return top_3_results
    else:
        filiere_unique = model_pipeline.predict(df_input)[0]
        return [{"filiere": str(filiere_unique), "confiance": 100.0}]


if __name__ == "__main__":
    test_profil = {
        'serie': 'D',
        'moyenne_generale': 14.5,
        'matieres_fortes': 'Sciences Physiques et Chimiques; SVT / Biologie-Géologie',
        'matieres_faibles': 'Histoire-Géographie; Anglais',
        'centres_interet': 'Robotique; Musique',
        'competences': 'Analyse en laboratoire; Pharmacologie; Résolution de problèmes'
    }
    print("Résultat structuré pour l'agent :", predire_orientation_top3(test_profil))