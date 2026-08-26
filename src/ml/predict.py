import os
import joblib
import pandas as pd

def predict_ispm_mention(student_profile: dict):
    model_path = 'models/ispm_orientation_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} est introuvable. Exécute d'abord train.py")
        
    # Charger le modèle entraîné
    model = joblib.load(model_path)
    
    # Transformer en DataFrame avec l'ordre exact des features
    df_input = pd.DataFrame([student_profile])
    
    # Prédiction et probabilités
    prediction = model.predict(df_input)[0]
    probas = model.predict_proba(df_input)[0]
    classes = model.classes_
    
    scores = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probas)}
    return prediction, scores

if __name__ == "__main__":
    # Exemple de profil de test (Ex : Élève en Bac C avec de bons résultats)
    profil_test = {
        'serie_bac': 'Bac C',
        'note_maths': 15.0,
        'note_physique': 14.0,
        'note_francais': 11.0,
        'note_malagasy': 11.5,
        'score_sci_pondere': 14.5, # Calculé selon les coefficients (ex: moyenne pondérée maths/physique)
        'niveau_prog': 4,
        'interet_elec': 3,
        'appetence_design': 1,
        'interet_gestion': 2
    }
    
    mention, probas = predict_ispm_mention(profil_test)
    print(f"\nProfil testé : {profil_test['serie_bac']} (Score sci pondéré : {profil_test['score_sci_pondere']})")
    print(f"Mention ISPM recommandée : --> {mention} <--")
    print("Probabilités par mention :", probas)