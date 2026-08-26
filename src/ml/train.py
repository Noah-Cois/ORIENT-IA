import os
import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Chargement dynamique de tous les fichiers du dossier synthetic
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))

synthetic_dir = os.path.join(project_root, 'data', 'synthetic')
if not os.path.exists(synthetic_dir):
    synthetic_dir = os.path.join(project_root, 'synthetic')

all_csv_files = glob.glob(os.path.join(synthetic_dir, '*.csv'))
print(f"Fichiers détectés dans synthetic : {len(all_csv_files)} fichiers trouvés.")

# 2. Stratégie d'intégration globale : 
# On cherche en priorité le fichier principal qui contient les profils et les cibles d'orientation,
# tout en pouvant enrichir avec les autres tables si nécessaire.
# Le fichier le plus complet pour l'apprentissage direct reste 'profils_etudiants_synthetiques.csv' ou 'candidat_filiere.csv'.

target_df_path = os.path.join(synthetic_dir, 'profils_etudiants_synthetiques.csv')
if not os.path.exists(target_df_path):
    target_df_path = os.path.join(synthetic_dir, 'candidat_filiere.csv')

df = pd.read_csv(target_df_path)
print(f"Dataset principal chargé : {os.path.basename(target_df_path)} ({len(df)} lignes)")

# 3. Définition propre de la cible et des features exploitables
target_col = 'filiere_recommandee' if 'filiere_recommandee' in df.columns else ('filiere_associee' if 'filiere_associee' in df.columns else df.columns[-1])

# Sélection des features disponibles dans le dataset principal
features = [
    'serie', 'moyenne_generale', 'matieres_fortes', 
    'matieres_faibles', 'centres_interet', 'competences', 'serie_ou_filiere'
]
available_features = [col for col in features if col in df.columns]

# Nettoyage strict des NaN uniquement sur les colonnes utiles pour éviter le plantage du modèle
df_clean = df.dropna(subset=available_features + [target_col]).copy()

print(f"Lignes exploitables après nettoyage des valeurs manquantes : {len(df_clean)}")

X = df_clean[available_features]
y = df_clean[target_col]

# Identification des colonnes catégorielles à encoder
categorical_cols = [col for col in available_features if df_clean[col].dtype == 'object']

# 4. Pipeline de Machine Learning
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, random_state=42))
])

# 5. Entraînement et Évaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print("--- Rapport de performance du modèle global ---")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Sauvegarde du modèle
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'ispm_orientation_model.pkl')

joblib.dump(model_pipeline, model_path)
print(f"Modèle global sauvegardé avec succès dans {model_path} !")