import os
import pandas as pd
import joblib
import numpy as np

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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle est introuvable à l'emplacement : {model_path}. Lance d'abord train.py !")

    # 2. Chargement du pipeline
    model_pipeline = joblib.load(model_path)

    # 3. Validation et préparation des données pour le modèle
    df_input = pd.DataFrame([profil_etudiant])

    # 4. Extraction des prédictions et probabilités
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

# Test local optionnel
if __name__ == "__main__":
    test_profil = {
        'serie': 'D',
        'moyenne_generale': 14.5,
        'matieres_fortes': 'Mathématiques, SVT',
        'matieres_faibles': 'Histoire-Géographie',
        'centres_interet': 'Nouvelles technologies, Biologie',
        'competences': 'Analyse, Logique'
    }
    print("Résultat structuré pour l'agent :", predire_orientation_top3(test_profil))