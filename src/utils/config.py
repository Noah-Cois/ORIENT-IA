import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ajoute la racine du projet au PYTHONPATH de manière dynamique
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Charge le fichier .env situé à la racine
load_dotenv(ROOT_DIR / ".env")

def get_hf_token() -> str:
    """Récupère la clé API Hugging Face (compatible Local & Streamlit Cloud)."""
    # 1. Vérification dans le fichier .env local
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return token

    # 2. Vérification dans Streamlit Secrets (Déploiement Cloud)
    try:
        import streamlit as st
        if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
            return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except Exception:
        pass

    raise ValueError(
        "❌ Clé HUGGINGFACEHUB_API_TOKEN introuvable. "
        "Ajoutez-la dans .env (local) ou dans les Secrets Streamlit (cloud)."
    )

def get_gemini_token() -> str:
    """Récupère la clé API Google Gemini (compatible Local & Streamlit Cloud)."""
    # 1. Vérification dans le fichier .env local
    token = os.getenv("GEMINI_API_KEY")
    if token:
        return token

    # 2. Vérification dans Streamlit Secrets (Déploiement Cloud)
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    raise ValueError(
        "❌ Clé GEMINI_API_KEY introuvable. "
        "Ajoutez-la dans .env (local) ou dans les Secrets Streamlit (cloud)."
    )