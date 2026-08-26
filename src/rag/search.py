import sys
from pathlib import Path

# 1. Fixation dynamique de la racine
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from src.utils.config import get_hf_token

CHROMA_DB_DIR = ROOT_DIR / "data" / "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Liste des filières gérées par l'ISPM
LISTE_FILIERES = ["ISAIA", "EMII", "IGGLIA", "GCA", "IAA", "TIM", "MB"]

def detecter_filiere(question: str) -> str | None:
    """Détecte la présence d'un acronyme de filière dans la question."""
    for filiere in LISTE_FILIERES:
        if re.search(rf"\b{filiere}\b", question, re.IGNORECASE):
            return filiere.upper()
    return None

def tester_recherche(question: str, top_k: int = 3):
    HF_TOKEN = get_hf_token()
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        huggingfacehub_api_token=HF_TOKEN
    )
    
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DB_DIR), 
        embedding_function=embeddings
    )
    
    filiere_target = detecter_filiere(question)
    search_kwargs = {"k": top_k}
    
    # 2. Si une filière est détectée, on filtre par métadonnée et on enrichit le texte
    if filiere_target:
        print(f"\n🎯 Filière détectée : {filiere_target} (Filtrage ChromaDB activé)")
        search_kwargs["filter"] = {"filiere": filiere_target}
        query_text = f"Filière {filiere_target} : {question}"
    else:
        query_text = question

    print(f"🔍 Requête exécutée : '{query_text}'")
    resultats = vectorstore.similarity_search(query_text, **search_kwargs)
    
    # Repli (Fallback) si le filtre strict ne renvoie aucun résultat
    if not resultats and filiere_target:
        print("⚠️ Aucun chunk trouvé avec le filtre strict. Recherche sans filtre...")
        resultats = vectorstore.similarity_search(query_text, k=top_k)

    for i, doc in enumerate(resultats, 1):
        print(f"\n--- Résultat {i} ---")
        print(f"Fichier  : {doc.metadata.get('chemin_fichier', 'N/A')}")
        print(f"Filière  : {doc.metadata.get('filiere', 'Général')}")
        print(f"Extrait  : {doc.page_content[:250]}...")

if __name__ == "__main__":
    # Test 1 : Avec acronyme spécifique (Filtre + Enrichissement)
    tester_recherche("Quels sont les prérequis et débouchés de la filière ISAIA ?")

    # Test 2 : Sans acronyme (Recherche globale)
    tester_recherche("Combien coûtent l'écolage annuel et le droit d'inscription en Licence 1 ?")