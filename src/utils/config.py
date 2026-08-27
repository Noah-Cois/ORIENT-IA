import os
import sys
import threading
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

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
        if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
            return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except Exception:
        pass

    raise ValueError(
        "❌ Clé HUGGINGFACEHUB_API_TOKEN introuvable. "
        "Ajoutez-la dans .env (local) ou dans les Secrets Streamlit (cloud)."
    )

# =====================================================================
# GESTIONNAIRE DE ROTATION DES CLÉS GEMINI (KEY POOL)
# =====================================================================

class APIKeyManager:
    def __init__(self, keys: list):
        self.keys = keys
        self.current_index = 0
        self.lock = threading.Lock()
        self.exhausted = False

    def get_current_key(self) -> str | None:
        """Retourne la clé active, ou None si toutes sont épuisées."""
        with self.lock:
            if self.exhausted or not self.keys:
                return None
            return self.keys[self.current_index]

    def rotate_key(self, failed_key: str) -> bool:
        """Passe à la clé suivante. Retourne False s'il n'y a plus de clés."""
        with self.lock:
            # Si une autre requête a déjà fait tourner la clé entre temps
            if self.keys[self.current_index] != failed_key:
                return not self.exhausted
            
            self.current_index += 1
            print(f"🔄 [KEY ROTATION] Passage à la clé {self.current_index + 1}/{len(self.keys)}")
            
            if self.current_index >= len(self.keys):
                self.exhausted = True
                return False
            return True


@st.cache_resource(show_spinner=False)
def get_api_key_manager() -> APIKeyManager:
    """
    Initialise le gestionnaire de clés UNE SEULE FOIS pour toute l'application Streamlit.
    Supporte Streamlit Secrets (Cloud) et le fichier .env (Local).
    """
    keys_str = ""
    
    # 1. Tenter de lire GEMINI_API_KEYS (liste) depuis Streamlit Cloud Secrets
    try:
        keys_str = st.secrets.get("GEMINI_API_KEYS", "")
    except Exception:
        pass
    
    # 2. Repli sur le fichier .env (Local) pour GEMINI_API_KEYS
    if not keys_str:
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        
    # 3. Repli de compatibilité : si l'ancienne variable GEMINI_API_KEY (unique) est utilisée dans Streamlit
    if not keys_str:
        try:
            keys_str = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
            
    # 4. Repli de compatibilité : si l'ancienne variable GEMINI_API_KEY (unique) est utilisée en local
    if not keys_str:
        keys_str = os.getenv("GEMINI_API_KEY", "")

    # Nettoyage et création de la liste de clés (sépare par des virgules et enlève les espaces)
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    
    if not keys:
        raise ValueError(
            "❌ AUCUNE CLÉ GEMINI TROUVÉE ! "
            "Ajoutez GEMINI_API_KEYS (séparées par des virgules) dans .env ou st.secrets."
        )
        
    return APIKeyManager(keys)