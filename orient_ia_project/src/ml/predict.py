import os
import pandas as pd
import joblib

def predire_orientation(profil_etudiant):
    """
    Prend en paramètre un dictionnaire représentant le profil d'un étudiant
    et retourne la filière recommandée par le modèle de Machine Learning.
    """
    # 1. Localisation dynamique du modèle enregistré
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
    model_path = os.path.join(project_root, 'models', 'ispm_orientation_model.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle est introuvable à l'emplacement : {model_path}. Lance d'abord train.py !")

    # 2. Chargement du pipeline entraîné
    model_pipeline = joblib.load(model_path)

    # 3. Conversion du dictionnaire du profil en DataFrame (le format attendu par scikit-learn)
    df_input = pd.DataFrame([profil_etudiant])

    # 4. Prédiction de la filière
    filiere_predite = model_pipeline.predict(df_input)[0]

    # Récupération des probabilités si disponibles (pour donner un indice de confiance)
    confiance = None
    if hasattr(model_pipeline.named_steps['classifier'], "predict_proba"):
        probas = model_pipeline.predict_proba(df_input)
        confiance = max(probas[0]) * 100

    return filiere_predite, confiance

# --- Exemple de test direct si on exécute ce script dans le terminal ---
if __name__ == "__main__":
    # Un exemple de profil étudiant à tester
    nouveau_profil = {
        'serie': 'D',
        'moyenne_generale': 14.5,
        'matieres_fortes': 'Mathématiques, SVT',
        'matieres_faibles': 'Histoire-Géographie',
        'centres_interet': 'Nouvelles technologies, Biologie',
        'competences': 'Analyse, Logique'
    }

    print("Test de prédiction pour le profil :", nouveau_profil)
    filiere, score_conf = predire_orientation(nouveau_profil)
    
    print("\n--- Résultat de la prédiction ---")
    print(f"Filière ISPM recommandée : {filiere}")
    if score_conf:
        print(f"Indice de confiance estimé : {score_conf:.2f}%")