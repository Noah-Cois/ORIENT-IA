import streamlit as st
import time

# Configuration de la page Streamlit
st.set_page_config(
    page_title="ORIENT’IA - Assistant d'Orientation ISPM",
    page_icon="🎓",
    layout="wide"
)

# 1. Mention légale obligatoire (Exigence stricte du cahier des charges)
st.warning(
    "⚠️ **Avertissement obligatoire :** *ORIENT’IA constitue un outil d’aide à l’orientation. "
    "Ses recommandations ne remplacent ni l’avis d’un conseiller pédagogique ni une décision officielle d’admission.*"
)

st.title("🎓 ORIENT’IA - Système d'Aide à l'Orientation ISPM")

# Initialisation de l'état de la session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# 2. Barre latérale : Saisie explicite du profil (Aucune inférence implicite/psychologique)
with st.sidebar:
    st.header("👤 Profil déclaré par l'étudiant")
    st.caption("Saisie explicite des préférences et du niveau académique.")
    
    parcours = st.selectbox(
        "Parcours d'origine :",
        ["Baccalauréat Scientifique (C/D/S)", "Baccalauréat Technique", "Licence L1/L2", "Autre"]
    )
    domaines_interet = st.multiselect(
        "Centres d'intérêt déclarés :",
        ["Informatique & Génie Logiciel", "Électronique & Automatisme", "Réseaux & Télécoms", "Management & Digital"]
    )
    niveau_math_info = st.slider("Niveau auto-évalué en Math/Info (0 à 20) :", 0, 20, 12)
    
    if st.button("💾 Mettre à jour le profil"):
        st.session_state.user_profile = {
            "parcours": parcours,
            "domaines": domaines_interet,
            "niveau_math_info": niveau_math_info
        }
        st.success("Profil mis à jour !")

    st.divider()
    
    # Toggle d'observabilité pour la soutenance / le suivi des tests
    st.header("⚙️ Observabilité")
    show_traces = st.checkbox("Afficher le panneau d'observabilité (Traces & Outils)", value=True)

# 3. Zone Conversationnelle Principale
# Affichage de l'historique du chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Affichage structuré des métadonnées (Confiance, Sources, Traces)
        if "metadata" in message:
            meta = message["metadata"]
            
            # Affichage de l'incertitude et de la confiance
            if "confiance" in meta:
                st.metric(
                    label="Niveau de confiance de la recommandation", 
                    value=f"{meta['confiance']}%", 
                    delta=f"Incertitude déclarée: {meta['incertitude']}"
                )
            
            # Affichage des sources citées RAG
            if "sources" in meta and meta["sources"]:
                with st.expander("📚 Sources citées (Corpus ISPM)"):
                    for src in meta["sources"]:
                        st.markdown(f"- `{src}`")
            
            # Panneau technique des traces capturées
            if show_traces and "traces" in meta:
                with st.expander("🔍 Traces d'observabilité (Audit & Logs)"):
                    st.json(meta["traces"])

# 4. Saisie utilisateur et boucle d'exécution de l'Agent
if prompt := st.chat_input("Posez une question sur les formations ISPM ou demandez une recommandation..."):
    # Stockage du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'Agent (Simulée pour le composant UI, à relier à l'agent backend)
    with st.chat_message("assistant"):
        with st.spinner("Traitement par l'agent : interrogation des outils RAG et prédictions ML..."):
            time.sleep(1) # Latence simulée des appels d'outils
            
            # Exemple de réponse argumentée, traçable et prudente
            response_text = (
                "Sur la base de votre profil explicitement déclaré et de l'analyse du corpus ISPM :\n\n"
                "### Recommandation\n"
                "Nous vous orientons vers la filière **Génie Logiciel & Systèmes d'Information (GLSI)**.\n\n"
                "### Argumentation & Prudence\n"
                "* **Adéquation avec votre profil :** Vos centres d'intérêt pour le développement et votre score auto-évalué correspondent aux prérequis académiques de la filière.\n"
                "* **Remarque prudentielle :** Le volume d'heures en algorithmique au 1er semestre nécessite une bonne maîtrise préalable des bases de la logique mathématique."
            )
            
            # Structure des données d'observabilité capturées
            metadata_example = {
                "confiance": 88,
                "incertitude": "Faible (0.12)",
                "sources": [
                    "Maquette_Pedagogique_ISPM_2025.pdf (Page 12, Section GLSI)",
                    "Brochure_Admissions_ISPM_2024-2025.pdf (Page 4)"
                ],
                "traces": {
                    "question_initiale": prompt,
                    "profil_transmis": st.session_state.user_profile,
                    "outils_appeles": ["analyser_profil_ml", "rechercher_formation", "verifier_prerequis"],
                    "scores_rag": [0.94, 0.87],
                    "entrees_sorties_ml": {
                        "input_vector": [st.session_state.user_profile.get("niveau_math_info", 0)],
                        "output_prediction": "GLSI",
                        "proba": 0.88
                    },
                    "temps_execution_ms": 520
                }
            }
            
            # Affichage dans l'UI
            st.markdown(response_text)
            
            st.metric(
                label="Niveau de confiance", 
                value=f"{metadata_example['confiance']}%", 
                delta=f"Incertitude: {metadata_example['incertitude']}"
            )
            
            with st.expander("📚 Sources citées (Corpus ISPM)"):
                for src in metadata_example["sources"]:
                    st.markdown(f"- `{src}`")
                    
            if show_traces:
                with st.expander("🔍 Traces d'observabilité (Audit & Logs)"):
                    st.json(metadata_example["traces"])
            
            # Enregistrement dans la session
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "metadata": metadata_example
            })