import sys
from pathlib import Path

import streamlit as st

# Permet d'importer les modules src.* quel que soit le dossier de lancement
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.chatbot import OrientIAAgent
from src.agent.tools import traduire_profil_vers_vocabulaire_ml, analyser_profil_ml
from src.rag.search import CHROMA_DB_DIR
from src.rag.ingest import executer_ingestion


def _base_vectorielle_existe() -> bool:
    """
    Vérifie si la base Chroma a déjà été générée et contient des données.
    Nécessaire car data/chroma_db n'est pas versionné sur Git (trop volumineux),
    donc absent lors d'un déploiement Streamlit Cloud frais.
    """
    if not CHROMA_DB_DIR.exists():
        return False
    # Chroma persiste au moins un fichier sqlite dans ce dossier une fois indexé
    return any(CHROMA_DB_DIR.iterdir())


@st.cache_resource(show_spinner=False)
def assurer_base_vectorielle():
    """
    Exécuté une seule fois par session (grâce au cache) : si la base vectorielle
    n'existe pas encore sur le disque, on l'ingère depuis data/corpus avant de
    laisser le reste de l'application démarrer.
    """
    if not _base_vectorielle_existe():
        executer_ingestion()
    return True

# Configuration de la page en mode "wide" pour profiter de tout l'écran côte à côte
st.set_page_config(
    page_title="ORIENT'IA - Assistant d'Orientation",
    page_icon="logo_ispm.png",
    layout="wide"
)

# CSS personnalisé pour embellir l'interface (Champs de saisie + Logo)
st.markdown("""
    <style>
    /* 1. Cache les flèches haut/bas des champs numériques (number_input) */
    input[type=number]::-webkit-inner-spin-button, 
    input[type=number]::-webkit-outer-spin-button { 
        -webkit-appearance: none; 
        margin: 0; 
    }
    
    /* 2. Arrondit les coins des champs de saisie pour un look moderne */
    div.stNumberInput input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    /* 3. Boîte pour faire ressortir le logo s'il a un fond sombre */
    .logo-container {
        background: radial-gradient(circle, #f8fafc 0%, #e2e8f0 100%);
        padding: 10px;
        border-radius: 15px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)


# --- INITIALISATION DE L'AGENT (une seule fois par session, pas à chaque rerun) ---
@st.cache_resource(show_spinner=False)
def charger_agent() -> OrientIAAgent:
    return OrientIAAgent()


with st.spinner("Préparation de la base de connaissances (première utilisation uniquement)..."):
    assurer_base_vectorielle()

agent = charger_agent()

# Initialisation de l'historique du chat dans la session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Une question sur les filières de l'ISPM ? Posez-la-moi ici."}
    ]

# Stocke le dernier profil analysé, pour que le chat en tienne compte
if "dernier_profil" not in st.session_state:
    st.session_state.dernier_profil = None


# --- MAPPING SÉRIE DU FORMULAIRE -> CODE UTILISÉ PAR LE MODÈLE ---
MAPPING_SERIE = {
    "Série C": "C",
    "Série D": "D",
    "Série A1": "A1",
    "Série A2": "A2",
    "Série S": "S",
    "Série Tertiaire/Gestion": "G",
    "Série Technique": "Technique",
}


def construire_profil_libre(donnees_profil: dict) -> dict:
    """
    Transforme les données brutes du formulaire (notes par matière + série +
    centre d'intérêt) en un profil "libre" prêt à être traduit vers le
    vocabulaire exact du modèle ML via traduire_profil_vers_vocabulaire_ml.
    """
    notes = {
        "Mathématiques": donnees_profil["note_maths"],
        "Physique-Chimie": donnees_profil["note_pc"],
        "SVT": donnees_profil["note_svt"],
        "Français": donnees_profil["note_francais"],
        "Anglais": donnees_profil["note_anglais"],
        "Gestion/Eco/Philo": donnees_profil["note_gestion"],
    }
    matieres_triees = sorted(notes.items(), key=lambda kv: kv[1], reverse=True)
    matieres_fortes = [nom for nom, _ in matieres_triees[:2]]
    matieres_faibles = [nom for nom, _ in matieres_triees[-2:]]
    moyenne_generale = sum(notes.values()) / len(notes)

    return {
        "serie": MAPPING_SERIE.get(donnees_profil["serie"], donnees_profil["serie"]),
        "moyenne_generale": moyenne_generale,
        "notes": notes,
        "matieres_fortes": matieres_fortes,
        "matieres_faibles": matieres_faibles,
        "centres_interet": [donnees_profil["interet"].replace("_", " ")],
        "competences": [],
    }


def analyser_profil_complet(donnees_profil: dict) -> dict:
    """
    Pipeline complet : profil libre -> traduction LLM vers vocabulaire exact
    -> appel au vrai modèle ML entraîné. Retourne le contrat de analyser_profil_ml.
    """
    profil_libre = construire_profil_libre(donnees_profil)

    if agent.llm:
        profil_traduit = traduire_profil_vers_vocabulaire_ml(profil_libre, agent.llm)
    else:
        # Mode dégradé : pas de LLM disponible, on tente avec le profil brut
        profil_traduit = profil_libre

    return analyser_profil_ml.invoke({"profil": profil_traduit})


# --- FONCTION D'AFFICHAGE DES DONNÉES D'ENTRÉE ET DES RÉSULTATS ---
def afficher_resultats_orientation(donnees_profil, t1, s1, t2, s2, t3, s3):
    """
    Affiche le récapitulatif des données d'entrée saisies par l'utilisateur 
    ainsi que les résultats de l'orientation (métriques).
    """
    st.markdown("---")
    st.markdown("### Récapitulatif de votre profil")
    
    # Affichage des données d'entrée sous forme de colonnes / texte
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.write(f"- **Série du Bac :** {donnees_profil['serie']}")
        st.write(f"- **Centre d'intérêt :** {donnees_profil['interet']}")
        st.write(f"- **Mathématiques :** {donnees_profil['note_maths']}/20")
        st.write(f"- **Physique-Chimie :** {donnees_profil['note_pc']}/20")
    with col_p2:
        st.write(f"- **SVT :** {donnees_profil['note_svt']}/20")
        st.write(f"- **Français :** {donnees_profil['note_francais']}/20")
        st.write(f"- **Anglais :** {donnees_profil['note_anglais']}/20")
        st.write(f"- **Gestion / Éco / Philo :** {donnees_profil['note_gestion']}/20")

    st.markdown("---")
    st.markdown("### Résultat de l'orientation")
    st.info(f"**Filière principale recommandée :** {t1}")

    st.markdown("#### Top 3 des parcours suggérés :")
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric(label="1er Choix", value=t1, delta=s1)
    with col_res2:
        st.metric(label="2ème Choix", value=t2, delta=s2)
    with col_res3:
        st.metric(label="3ème Choix", value=t3, delta=s3)


# --- FONCTION DE GESTION DU CHAT ---
def gerer_interface_chat():
    """
    Gère l'affichage de l'historique des messages et la saisie utilisateur.
    """
    st.subheader("Assistant Conversationnel (IA)")
    
    chat_container = st.container(height=500)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Posez votre question à l'assistant..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("ORIENT'IA réfléchit..."):
            resultat = agent.process_query(prompt, st.session_state.dernier_profil)
            reponse_assistant = resultat["response"]

        st.session_state.messages.append({"role": "assistant", "content": reponse_assistant})
        st.rerun()


# --- EN-TÊTE ---
col_logo, col_titre = st.columns([0.1, 0.9])

with col_logo:
    st.image("logo_ispm.png", width=130)
    st.markdown('</div>', unsafe_allow_html=True)

with col_titre:
    st.title("ORIENT'IA - Plateforme d'Orientation & Assistant")
    st.markdown("Institut Supérieur Polytechnique de Madagascar (ISPM)")
    
    st.markdown(
        """
        <div style='background-color: #262730; padding: 8px 12px; border-radius: 6px; border-left: 4px solid #F60000; font-size: 20px; color: #fff; margin-top: 8px; margin-bottom: 12px;'>
            <b>Avertissement :</b> ORIENT'IA constitue un outil d'aide à l'orientation. Ses recommandations ne remplacent ni l'avis d'un conseiller pédagogique ni une décision officielle d'admission.
        </div>
        """,
        unsafe_allow_html=True
    )

# --- DIVISION DE L'ÉCRAN EN 2 COLONNES ---
col_form, col_chat = st.columns([1.2, 1], gap="large")

# Colonne de gauche (Formulaire de profil)
with col_form:
    st.subheader(" Votre Profil Académique")
    
    with st.form("formulaire_profil_utilisateur"):
        serie = st.selectbox(
            "Série du Baccalauréat",
            ['Série C', 'Série D', 'Série A1', 'Série A2', 'Série S', 'Série Tertiaire/Gestion', 'Série Technique']
        )
        
        interet = st.selectbox(
            "Centre d'intérêt principal",
            [
                'Informatique_Logiciel', 'Electronique_Robotique', 'Finance_Comptabilite',
                'Droit_Affaires', 'Biologie_Pharmacie', 'Agronomie_Elevage', 
                'Tourisme_Environnement', 'Genie_Civil'
            ]
        )

        st.markdown("**Vos Notes au Bac (sur 20)**")
        c1, c2 = st.columns(2)
        with c1:
            note_maths = st.number_input("Mathématiques", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
            note_pc = st.number_input("Physique-Chimie", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
            note_svt = st.number_input("SVT", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
        with c2:
            note_francais = st.number_input("Français", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
            note_anglais = st.number_input("Anglais", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
            note_gestion = st.number_input("Gestion / Éco / Philo", min_value=0.0, max_value=20.0, value=12.0, step=0.5)

        soumis = st.form_submit_button("Analyser mon profil", use_container_width=True)
        
    if soumis:
        # Collecte des données d'entrée dans un dictionnaire propre
        profil_saisi = {
            "serie": serie,
            "interet": interet,
            "note_maths": note_maths,
            "note_pc": note_pc,
            "note_svt": note_svt,
            "note_francais": note_francais,
            "note_anglais": note_anglais,
            "note_gestion": note_gestion
        }

        with st.spinner("Analyse du profil en cours (traduction + modèle ML)..."):
            try:
                ml_result = analyser_profil_complet(profil_saisi)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse du profil : {e}")
                ml_result = None

        if ml_result and ml_result.get("status") == "SUCCESS":
            st.success("Analyse du profil terminée.")

            probas_triees = sorted(
                ml_result["predict_proba"].items(), key=lambda kv: kv[1], reverse=True
            )
            # Complète à 3 entrées si le modèle en renvoie moins
            while len(probas_triees) < 3:
                probas_triees.append(("N/A", 0.0))

            (t1, p1), (t2, p2), (t3, p3) = probas_triees[:3]
            s1, s2, s3 = f"{p1 * 100:.1f}%", f"{p2 * 100:.1f}%", f"{p3 * 100:.1f}%"

            # Mémorise le profil pour que le chat puisse en tenir compte ensuite
            st.session_state.dernier_profil = construire_profil_libre(profil_saisi)

            afficher_resultats_orientation(profil_saisi, t1, s1, t2, s2, t3, s3)
        elif ml_result:
            st.warning(f"Analyse incomplète : {', '.join(ml_result.get('facteurs_cles', ['Erreur inconnue']))}")

# Colonne de droite (Assistant Conversationnel)
with col_chat:
    gerer_interface_chat()

# --- FOOTER DE LA PAGE ---
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; padding: 15px; font-size: 14px;'>
         <b>ORIENT'IA</b> — <b>Institut Supérieur Polytechnique de Madagascar (ISPM)</b>   <b >  — ISAIA 5</b>
         | Plateforme d'Aide à l'Orientation Intelligente
    </div>
    """,
    unsafe_allow_html=True
)