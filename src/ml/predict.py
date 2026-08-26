import joblib
import pandas as pd
import os

# Construction des chemins absolus pour éviter les erreurs peu importe d'où on lance le script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'reco_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

def recommander_parcours(serie_bac, note_maths, note_physique, niveau_prog, appetence_comm):
    """
    Fonction qui sera appelée par l'Agent LLM (Dev 4) pour obtenir une recommandation
    à partir du profil de l'étudiant.
    """
    try:
        # 1. Charger le modèle et l'encodeur
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)

        # 2. Encodage de la série du Bac (ex: transforme "S" en un chiffre compris par la machine)
        # Gère le cas où l'utilisateur entre une série inconnue
        if serie_bac not in le.classes_:
            serie_bac = le.classes_[0] # Valeur par défaut pour éviter le crash
            
        serie_encoded = le.transform([serie_bac])[0]

        # 3. Préparer les données exactement comme lors de l'entraînement
        input_data = pd.DataFrame([[
            serie_encoded, note_maths, note_physique, niveau_prog, appetence_comm
        ]], columns=['serie_bac_encoded', 'note_maths', 'note_physique', 'niveau_prog', 'appetence_comm'])

        # 4. Faire la prédiction
        prediction = model.predict(input_data)[0]
        
        return prediction

    except FileNotFoundError:
        return "Erreur : Le modèle n'a pas été trouvé. As-tu bien exécuté train.py avant ?"
    except Exception as e:
        return f"Erreur lors de la prédiction : {str(e)}"

# Zone de test : pour vérifier que ton code marche dans ton terminal
if __name__ == "__main__":
    print("--- Test du modèle de recommandation ---")
    
    # Test 1 : Profil plutôt technique/IA
    reco_1 = recommander_parcours(serie_bac='S', note_maths=16, note_physique=15, niveau_prog=5, appetence_comm=2)
    print(f"Profil 1 (Maths 16, Prog 5) -> Recommandation : {reco_1}")
    
    # Test 2 : Profil plutôt Management
    reco_2 = recommander_parcours(serie_bac='Litteraire', note_maths=10, note_physique=8, niveau_prog=1, appetence_comm=5)
    print(f"Profil 2 (Prog 1, Comm 5) -> Recommandation : {reco_2}")