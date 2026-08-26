import glob
import frontmatter
from dotenv import load_dotenv
import sys
from pathlib import Path

# Fixe le chemin vers la racine du projet (orient_ia_project/)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import get_hf_token
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

load_dotenv()
HF_TOKEN = get_hf_token()

CORPUS_DIR = ROOT_DIR / "data" / "corpus"
CHROMA_DB_DIR = ROOT_DIR / "data" / "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def charger_fichiers_markdown(directory: str) -> list[Document]:
    documents = []
    fichiers_md = glob.glob(f"{directory}/**/*.md", recursive=True)
    print(f"{CORPUS_DIR}")
    print(f"📄 {len(fichiers_md)} fichiers Markdown trouvés dans {directory}.")

    for path_fichier in fichiers_md:
        post = frontmatter.load(path_fichier)
        contenu_texte = post.content
        metadonnees = post.metadata
        metadonnees["chemin_fichier"] = path_fichier
        
        # Formater les listes YAML en chaînes de caractères pour ChromaDB
        for key, value in metadonnees.items():
            if isinstance(value, list):
                metadonnees[key] = ", ".join(map(str, value))

        documents.append(Document(page_content=contenu_texte, metadata=metadonnees))
        
    return documents

def executer_ingestion():
    if not HF_TOKEN:
        raise ValueError("❌ Clé HUGGINGFACEHUB_API_TOKEN non trouvée dans le fichier .env")

    # 1. Chargement
    docs_bruts = charger_fichiers_markdown(CORPUS_DIR)

    # 2. Découpage par titres Markdown puis par taille
    headers_to_split = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    
    chunks_finaux = []
    for doc in docs_bruts:
        sections = markdown_splitter.split_text(doc.page_content)
        for section in sections:
            metadonnees_fusionnees = {**doc.metadata, **section.metadata}
            sub_chunks = text_splitter.split_text(section.page_content)
            for chunk in sub_chunks:
                chunks_finaux.append(
                    Document(page_content=chunk, metadata=metadonnees_fusionnees)
                )

    print(f"🧩 {len(chunks_finaux)} chunks générés.")

    # 3. Vectorisation et sauvegarde
    print(f"🌐 Vectorisation via l'API Hugging Face ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        huggingfacehub_api_token=HF_TOKEN
    )
    
    Chroma.from_documents(
        documents=chunks_finaux,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"✅ Indexation réussie ! Base stockée dans `{CHROMA_DB_DIR}`.")

if __name__ == "__main__":
    executer_ingestion()