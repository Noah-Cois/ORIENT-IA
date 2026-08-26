import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Chargement dynamique et sécurisé (basé sur l'emplacement de train.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))

# Recherche du fichier dans data/synthetic ou directement dans synthetic
data_path = os.path.join(project_root, 'data', 'synthetic', 'profils_etudiants_synthetiques.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(project_root, 'synthetic', 'profils_etudiants_synthetiques.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Le fichier des profils étudiants synthétiques est introuvable. Vérifie son emplacement.")

df = pd.read_csv(data_path)
print(f"Données chargées avec succès depuis : {data_path}")
print(f"Aperçu des colonnes disponibles : {list(df.columns)}")

# 2. Définition de la cible et des features pertinentes
# On cible la filière recommandée et on sélectionne les colonnes exploitables pour le ML
target_col = 'filiere_recommandee' if 'filiere_recommandee' in df.columns else df.columns[-1]

# Sélection des features en écartant les ID, noms et textes longs (comme 'justification')
features = [
    'serie', 'moyenne_generale', 'matieres_fortes', 
    'matieres_faibles', 'centres_interet', 'competences'
]
available_features = [col for col in features if col in df.columns]

X = df[available_features]
y = df[target_col]

# Identification des colonnes catégorielles à encoder
categorical_cols = [col for col in ['serie', 'matieres_fortes', 'matieres_faibles', 'centres_interet', 'competences'] if col in available_features]

# 3. Préparation du pipeline de prétraitement et du modèle
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

# 4. Séparation train/test, entraînement et évaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print("--- Rapport de performance du modèle (Dataset ISPM) ---")
print(classification_report(y_test, y_pred, zero_division=0))

# 5. Sauvegarde de l'artefact du modèle
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'ispm_orientation_model.pkl')

joblib.dump(model_pipeline, model_path)
print(f"Modèle sauvegardé avec succès dans {model_path} !")