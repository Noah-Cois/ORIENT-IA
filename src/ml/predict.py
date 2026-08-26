import joblib
import pandas as pd

def predict_ispm_mention(student_profile: dict):
    # Charger le modèle entraîné
    model = joblib.load('models/ispm_orientation_model.pkl')
    
    # Transformer en DataFrame
    df_input = pd.DataFrame([student_profile])
    
    # Prédiction
    prediction = model.predict(df_input)[0]
    probas = model.predict_proba(df_input)[0]
    classes = model.classes_
    
    scores = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probas)}
    return prediction, scores

if __name__ == "__main__":
    # Profil de test (Exemple : Bac C fort en maths et code)
    profil_test = {
        'serie_bac': 'Bac C',
        'note_maths': 16.0,
        'note_physique': 14.5,
        'niveau_prog': 4,
        'interet_elec': 2,
        'appetence_design': 1,
        'interet_gestion': 2
    }
    
    mention, probas = predict_ispm_mention(profil_test)
    print(f"\nMention recommandée : {mention}")
    print("Probabilités par mention :", probas)