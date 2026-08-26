import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Chargement des données enrichies
data_path = 'data/processed/mock_profiles.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Le fichier {data_path} est introuvable. Exécute d'abord generate_mock_data.py")

df = pd.read_csv(data_path)

# Définition des features (entrées) et de la cible (sortie)
# On inclut les notes pondérées et les auto-évaluations
features = [
    'serie_bac', 'note_maths', 'note_physique', 'note_francais', 
    'note_malagasy', 'score_sci_pondere', 'niveau_prog', 
    'interet_elec', 'appetence_design', 'interet_gestion'
]

X = df[features]
y = df['parcours_cible']

# 2. Préparation du pipeline de prétraitement et du modèle
categorical_cols = ['serie_bac']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Laisse passer les colonnes numériques telles quelles
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42))
])

# 3. Séparation train/test, entraînement et évaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print("--- Rapport de performance du modèle (Dataset enrichi) ---")
print(classification_report(y_test, y_pred))

# 4. Sauvegarde de l'artefact du modèle
os.makedirs('models', exist_ok=True)
model_path = 'models/ispm_orientation_model.pkl'
joblib.dump(model_pipeline, model_path)
print(f"Modèle sauvegardé avec succès dans {model_path} !")