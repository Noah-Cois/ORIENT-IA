import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# --- NOUVEAU : Chemins dynamiques sécurisés ---
# Récupère le dossier actuel (src/ml)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Remonte à la racine du projet (orient_ia_project)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# 1. Génération de données synthétiques
def generate_mock_data(n_samples=500):
    np.random.seed(42)
    data = {
        'serie_bac': np.random.choice(['S', 'Technique', 'Litteraire', 'OSE'], n_samples),
        'note_maths': np.random.normal(12, 3, n_samples).clip(0, 20),
        'note_physique': np.random.normal(11, 3, n_samples).clip(0, 20),
        'niveau_prog': np.random.randint(1, 6, n_samples),
        'appetence_comm': np.random.randint(1, 6, n_samples)
    }
    df = pd.DataFrame(data)
    
    conditions = [
        (df['niveau_prog'] >= 4) & (df['note_maths'] >= 12),
        (df['note_physique'] >= 13) & (df['niveau_prog'] <= 3),
        (df['appetence_comm'] >= 4)
    ]
    parcours = ['Intelligence Artificielle', 'Réseaux & Systèmes', 'Management / SI']
    df['parcours_cible'] = np.select(conditions, parcours, default='Génie Logiciel')
    
    return df

# 2. Préparation des données
print("Génération des données...")
df = generate_mock_data()

# Utilisation du chemin absolu sécurisé
os.makedirs(DATA_DIR, exist_ok=True)
df.to_csv(os.path.join(DATA_DIR, 'mock_profiles.csv'), index=False)

# Encodage
le = LabelEncoder()
df['serie_bac_encoded'] = le.fit_transform(df['serie_bac'])

X = df[['serie_bac_encoded', 'note_maths', 'note_physique', 'niveau_prog', 'appetence_comm']]
y = df['parcours_cible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraînement du Modèle
print("Entraînement du modèle RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Évaluation
print("\n--- Rapport de Classification ---")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# 5. Sauvegarde
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODELS_DIR, 'reco_model.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
print(f"Modèle sauvegardé avec succès dans {MODELS_DIR}")