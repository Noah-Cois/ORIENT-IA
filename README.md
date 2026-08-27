# ORIENT-IA

## Examen clinique

**ORIENT-IA** est un projet d’orientation académique basé sur l’intelligence artificielle, développé dans le cadre de l’examen clinique.

### Membres

- Josia ISAIA 5 — N°01
- Harimanantsoa ISAIA 5 — N°03
- Tsilavina ISAIA 5 — N°05
- Lovatiana ISAIA 5 — N°06
- Noah ISAIA 5 — N°13

## Application déployée

L’application est accessible en ligne :

**[Ouvrir ORIENT-IA](https://orient-ia-2bxqbpgsrddqwg5f6co79u.streamlit.app/)**

## Lancement local

### 1. Installer les dépendances

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

### 2. Configurer les variables d’environnement

Créer un fichier `.env` à la racine du projet et renseigner les tokens nécessaires :

```env
GEMINI_API_KEYS=...
HUGGINGFACEHUB_API_TOKEN=...
```

### 3. Lancer l’application

```bash
python -m streamlit run app.py
```

## Architecture du projet

```text
ORIENT-IA/
├── app.py
├── requirements.txt
│
├── data/
│   ├── corpus/              # Données ISPM au format Markdown
│   │   ├── biotechnologie_et_agronomie/
│   │   ├── genie_industriel_et_genie_civil/
│   │   ├── informatique_telecommunication/
│   │   ├── techniques_des_affaires/
│   │   ├── technique_du_tourisme/
│   │   └── fichiers de référence ISPM
│   │
│   ├── synthetic/           # Données synthétiques utilisées par le ML
│   ├── raw/                 # Données brutes
│   └── processed/           # Données traitées
│
├── models/
│   └── ispm_orientation_model.pkl
│
├── notebooks/
│   └── ml_exploration.ipynb
│
└── src/
    ├── agent/
    │   ├── chatbot.py       # Agent conversationnel
    │   └── tools.py         # Outils utilisés par l’agent
    │
    ├── db/
    │   ├── database.py      # Gestion de la base de données
    │   └── schema.sql       # Schéma de la base
    │
    ├── ml/
    │   ├── train.py         # Entraînement du modèle ML
    │   └── predict.py       # Prédictions du modèle
    │
    ├── rag/
    │   ├── ingest.py        # Ingestion et génération de la base vectorielle
    │   └── search.py        # Recherche dans la base vectorielle
    │
    └── symbolic/
        └── logic.py         # Logique symbolique
```

## Données

### Corpus ISPM

Les données provenant de l’ISPM sont stockées dans `data/corpus/`.

Elles sont organisées par filière et enregistrées au format **Markdown (`.md`)**, afin de faciliter leur traitement et leur transformation en embeddings pour le système RAG.

### Données synthétiques

Les données synthétiques utilisées pour le développement et l’entraînement des modèles sont stockées dans :

```text
data/synthetic/
```

## Intelligence artificielle

Le projet combine plusieurs composants :

- **Machine Learning (`src/ml/`)** : entraînement et prédiction du modèle d’orientation.
- **RAG (`src/rag/`)** : ingestion des données, génération et recherche dans la base vectorielle.
- **Agent (`src/agent/`)** : liaison entre l’agent conversationnel, les outils et les différents composants IA.
- **Logique symbolique (`src/symbolic/`)** : gestion de règles et de logique complémentaire.
- **Base de données (`src/db/`)** : gestion des données persistantes du projet.

Le modèle entraîné est sauvegardé dans :

```text
models/ispm_orientation_model.pkl
```

## Flux simplifié

```text
Données ISPM (.md)
        │
        ▼
   src/rag/ingest.py
        │
        ▼
 Base vectorielle
        │
        ▼
   src/rag/search.py
        │
        ├──────────────┐
        ▼              ▼
   src/agent/      src/ml/
   chatbot.py      predict.py
        │              │
        └──────┬───────┘
               ▼
          ORIENT-IA
               │
               ▼
        Interface Streamlit
```
