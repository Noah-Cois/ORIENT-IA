import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Chargement des données
df = pd.read_csv('data/processed/mock_profiles.csv')

X = df.drop(columns=['parcours_cible'])
y = df['parcours_cible']

# 2. Préparation des pipelines d'encodage et de modèle
categorical_cols = ['serie_bac']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 3. Entraînement et Évaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print("--- Rapport de performance du modèle ---")
print(classification_report(y_test, y_pred))

# 4. Sauvegarde de l'artefact
os.makedirs('models', exist_ok=True)
joblib.dump(model_pipeline, 'models/ispm_orientation_model.pkl')
print("Modèle sauvegardé avec succès dans models/ispm_orientation_model.pkl !")