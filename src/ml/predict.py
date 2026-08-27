"""
src/ml/predict.py
=================
Fonction de prédiction pour l'agent IA, optimisée pour Streamlit.
"""

import os
import sys
import threading
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
import streamlit as st

# Permet d'importer src.ml.train quel que soit le point d'entrée du script
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ml.train import entrainer_modele

# Verrou global pour éviter que deux utilisateurs ne lancent l'entraînement en même temps
_TRAINING_LOCK = threading.Lock()


def _assurer_modele_ml(model_path: str, project_root: str) -> None:
    """
    Vérifie que le modèle .pkl existe. S'il est absent, l'entraîne automatiquement
    de manière sécurisée (Thread-Safe) pour un déploiement web (Streamlit).
    """
    # 1. Vérification rapide
    if os.path.exists(model_path):
        return

    # 2. On bloque l'accès aux autres processus si le modèle n'existe pas
    with _TRAINING_LOCK:
        # 3. Double vérification au cas où l'entraînement vient juste de se terminer
        if os.path.exists(model_path):
            return

        print(
            f"[INFO] Modèle introuvable à {model_path} — entraînement automatique en cours "
            "(premier démarrage ou déploiement sans .pkl versionné)..."
        )
        
        # 4. Feedback visuel pour l'utilisateur sur l'interface web
        with st.spinner("⚙️ Premier démarrage : Entraînement du modèle d'orientation en cours (quelques secondes)..."):
            entrainer_modele(project_root=project_root, verbose=True)

        if not os.path.exists(model_path):
            st.error("Erreur critique : Impossible de sauvegarder le modèle.")
            raise FileNotFoundError(
                f"L'entraînement automatique s'est terminé mais le modèle reste introuvable à {model_path}."
            )

@st.cache_resource(show_spinner=False)
def _charger_modele(model_path: str):
    """
    Charge le modèle en mémoire UNE SEULE FOIS pour toute la durée de vie de l'application.
    Cela évite de lire le disque lourdement à chaque question posée à l'agent.
    """
    return joblib.load(model_path)


def predire_orientation_top3(profil_etudiant: dict) -> list:
    """
    Prend un dictionnaire contenant le profil de l'étudiant et retourne
    le Top 3 des filières recommandées avec leurs pourcentages de confiance.
    """
    # 1. Localisation dynamique robuste du modèle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    model_path = os.path.join(project_root, 'models', 'ispm_orientation_model.pkl')

    # 2. Vérifie l'existence du modèle, l'entraîne de façon sécurisée sinon
    _assurer_modele_ml(model_path, project_root)

    # 3. Chargement du pipeline (depuis le cache Streamlit !)
    model_pipeline = _charger_modele(model_path)

    # 4. Validation et préparation des données
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