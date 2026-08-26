import os
import pandas as pd
import joblib
import numpy as np

def predire_orientation_top3(profil_etudiant):
    """
    Prend en paramètre un dictionnaire représentant le profil d'un étudiant
    et retourne le Top 3 des filières recommandées avec leurs scores de confiance.
    """
    # 1. Localisation dynamique robuste (src/ml -> src -> orient_ia_project)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    
    model_path = os.path.join(project_root, 'models', 'ispm_orientation_model.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle est introuvable à l'emplacement : {model_path}. Lance d'abord train.py !")

    # 2. Chargement du pipeline entraîné
    model_pipeline = joblib.load(model_path)

    # 3. Conversion du dictionnaire du profil en DataFrame
    df_input = pd.DataFrame([profil_etudiant])

    # 4. Récupération des probabilités pour toutes les classes
    classifier = model_pipeline.named_steps['classifier']
    classes = classifier.classes_
    
    if hasattr(model_pipeline, "predict_proba"):
        probas = model_pipeline.predict_proba(df_input)[0]
        
        # Associer chaque classe à sa probabilité et trier par ordre décroissant
        top_indices = np.argsort(probas)[::-1]
        
        top_3_results = []
        for i in range(min(3, len(classes))):
            idx = top_indices[i]
            filiere = classes[idx]
            confiance = probas[idx] * 100
            top_3_results.append((filiere, confiance))
            
        return top_3_results
    else:
        # Fallback si le modèle ne supporte pas predict_proba
        filiere_unique = model_pipeline.predict(df_input)[0]
        return [(filiere_unique, 100.0)]

# --- Exemple de test direct ---
if __name__ == "__main__":
    nouveau_profil = {
        'serie': 'D',
        'moyenne_generale': 14.5,
        'matieres_fortes': 'Mathématiques, SVT',
        'matieres_faibles': 'Histoire-Géographie',
        'centres_interet': 'Nouvelles technologies, Biologie',
        'competences': 'Analyse, Logique'
    }

    print("Test de prédiction Top 3 pour le profil :", nouveau_profil)
    top_3 = predire_orientation_top3(nouveau_profil)
    
    print("\n--- Top 3 des filières ISPM recommandées ---")
    for rang, (filiere, score) in enumerate(top_3, 1):
        print(f"{rang}. Filière : **{filiere}** (Confiance : {score:.2f}%)")