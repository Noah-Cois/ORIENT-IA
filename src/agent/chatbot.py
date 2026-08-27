"""
src/agent/chatbot.py
====================
Agent Autonome ORIENT'IA propulsé par LangGraph (avec Key Rotation)
"""

import sys
from pathlib import Path

# Fixe le chemin vers la racine
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import json
import uuid
from typing import Dict, Any, Optional, Union, List

# Import de LangGraph
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    # Exception spécifique renvoyée par langchain-google-genai en cas de
    # quota/rate-limit dépassé (429 RESOURCE_EXHAUSTED).
    from langchain_google_genai.chat_models import GoogleRateLimitError
except ImportError:
    GoogleRateLimitError = None

# /!\ NOUVEL IMPORT : On remplace get_gemini_token par notre manager
from src.utils.config import get_api_key_manager 
from src.agent.tools import (
    rechercher_documentation_ispm,
    analyser_profil_ml,
    verifier_prerequis,
    comparer_parcours,
    traduire_profil_vers_vocabulaire_ml
)

DISCLAIMER_ISPM = (
    "*ORIENT’IA constitue un outil d’aide à l’orientation. "
    "Ses recommandations ne remplacent ni l’avis d’un conseiller pédagogique.*"
)

def _extraire_texte_reponse(content: Union[str, List[Any]]) -> str:
    """Normalise le contenu d'un message LangChain."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        morceaux = []
        for bloc in content:
            if isinstance(bloc, dict) and bloc.get("type") == "text":
                morceaux.append(bloc.get("text", ""))
            elif isinstance(bloc, str):
                morceaux.append(bloc)
        return "\n\n".join(m for m in morceaux if m)
    return str(content)

def _est_erreur_quota(e: Exception) -> bool:
    """Détecte un dépassement de quota/rate-limit Gemini."""
    if GoogleRateLimitError is not None and isinstance(e, GoogleRateLimitError):
        return True
    texte = str(e)
    return "RESOURCE_EXHAUSTED" in texte or "429" in texte or "quota" in texte.lower()

def _construire_reponse_fallback(ml_resultat: Optional[Dict[str, Any]], rag_resultats: List[Dict[str, Any]], raison: str) -> str:
    """Construit une réponse structurée de repli."""
    if ml_resultat and ml_resultat.get("status") == "SUCCESS":
        filiere = ml_resultat.get("filiere_recommandee", "N/A")
        score = ml_resultat.get("score_adequation", 0.0) * 100
        synth = f"D'après l'analyse de votre profil, la filière recommandée est **{filiere}** (score d'adéquation : {score:.0f}%)."
        probas = ml_resultat.get("predict_proba", {})
        proba_str = ", ".join(f"{k} : {v * 100:.0f}%" for k, v in probas.items())
        justif = f"Probabilités par filière : {proba_str}." if proba_str else "Détails du score indisponibles."
    else:
        synth = "Je ne peux pas générer de recommandation personnalisée pour le moment."
        justif = "Le service de génération de texte est temporairement indisponible ou surchargé."

    if rag_resultats:
        citations = "\n".join(
            f"- *{doc.get('source_title', 'Document ISPM')}* ({doc.get('section', '')})"
            for doc in rag_resultats
        )
    else:
        citations = "- Aucune source documentaire disponible pour cette requête."

    return (
        f"**Recommandation / Synthèse**\n{synth}\n\n"
        f"**Justification & Outils**\n{justif}\n\n"
        f"_Réponse générée en mode dégradé ({raison}), sans reformulation par l'IA générative._\n\n"
        f"**Sources & Citations**\n{citations}\n\n"
        f"{DISCLAIMER_ISPM}"
    )

class OrientIAAgent:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.model_name = model_name
        self._injection_keywords = ["ignore previous", "system prompt", "jailbreak"]
        self._psycho_keywords = ["analyse ma personnalité", "profil psychologique"]

        # /!\ CHANGEMENT : On ne crée plus self.llm ici. Il sera créé dynamiquement 
        # dans process_query avec la clé API valide.
        self.tools = [
            rechercher_documentation_ispm,
            verifier_prerequis,
            comparer_parcours,
        ]
    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """
        Génère dynamiquement l'instance LLM avec la clé Gemini active 
        provenant du KeyManager (permet la rétrocompatibilité avec agent.llm).
        """
        key_manager = get_api_key_manager()
        current_key = key_manager.get_current_key()
        
        if not current_key:
            raise ValueError("❌ Aucune clé API Gemini valide n'est disponible (quotas épuisés).")
            
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=current_key,
            temperature=0.2
        )

    def _check_security(self, text: str) -> Dict[str, bool]:
        """Détecte les tentatives de contournement."""
        text_lower = text.lower()
        return {
            "prompt_injection": any(kw in text_lower for kw in self._injection_keywords),
            "psycho_profiling": any(kw in text_lower for kw in self._psycho_keywords),
        }

    def process_query(self, user_input: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Traite la requête avec le moteur LangGraph et la rotation de clés."""
        start_time = time.time()
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"
        profile = user_profile or {}
        tool_calls_history = []

        # 1. ÉTAPE DE SÉCURITÉ & DÉONTOLOGIE
        sec_status = self._check_security(user_input)

        if sec_status["prompt_injection"]:
            response_text = (
                "**Recommandation / Synthèse**\nAction bloquée. ORIENT’IA ne peut pas exécuter d'instructions modifiant ses règles.\n\n"
                "**Justification & Score ML**\nTentative de contournement d'instructions détectée.\n\n"
                "**Sources & Citations**\n- Politique de Sécurité du Système IT ISPM 2026\n\n"
                f"{DISCLAIMER_ISPM}"
            )
            return {
                "response": response_text,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "trace": {"trace_id": trace_id, "tool_calls": [], "security_flags": {"prompt_injection_blocked": True}},
            }

        if sec_status["psycho_profiling"]:
            response_text = (
                "**Recommandation / Synthèse**\nDemande refusée. ORIENT’IA ne réalise aucune évaluation psychologique.\n\n"
                "**Justification & Score ML**\nConformité éthique : critères académiques uniquement.\n\n"
                "**Sources & Citations**\n- Charte Éthique et Protection des Données ISPM\n\n"
                f"{DISCLAIMER_ISPM}"
            )
            return {
                "response": response_text,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "trace": {"trace_id": trace_id, "tool_calls": [], "security_flags": {"psycho_refusal": True}},
            }

        # /!\ NOUVEAU : INITIALISATION DE LA ROTATION DES CLÉS
        key_manager = get_api_key_manager()
        max_retries = len(key_manager.keys) if key_manager.keys else 1
        tentatives = 0

        # Variables pour garder une trace en cas d'erreur fatale non liée au quota
        ml_resultat = None
        rag_resultats = []

        while tentatives <= max_retries:
            current_key = key_manager.get_current_key()

            # Mode dégradé si toutes les clés sont grillées ou absentes
            if not current_key:
                return {
                    "response": _construire_reponse_fallback(None, [], "quotas épuisés pour toutes les clés API"),
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                    "trace": {"trace_id": trace_id, "tool_calls": tool_calls_history, "security_flags": {"llm_unavailable": True}},
                }

            try:
                # 2. Instanciation dynamique du LLM avec la clé actuelle
                llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=current_key,
                    temperature=0.2
                )
                
                # On réinitialise l'historique des outils à chaque tentative pour ne pas faire de doublons
                tool_calls_history = [] 

                # 3. ANALYSE ML
                ml_resultat = None
                if profile.get("notes") or profile.get("competences"):
                    try:
                        profil_traduit = traduire_profil_vers_vocabulaire_ml(profile, llm)
                        tool_calls_history.append({"tool_name": "traduire_profil_vers_vocabulaire_ml", "tool_arguments": profile})
                        
                        ml_resultat = analyser_profil_ml.invoke({"profil": profil_traduit})
                        tool_calls_history.append({"tool_name": "analyser_profil_ml", "tool_arguments": profil_traduit})
                    except Exception as e:
                        # /!\ IMPORTANT : Si c'est un problème de quota sur la traduction ML, on propage l'erreur
                        if _est_erreur_quota(e): raise e
                        
                        print(f"[ERREUR ML] Analyse du profil échouée : {e}")
                        ml_resultat = {
                            "filiere_recommandee": "INCONNUE",
                            "score_adequation": 0.0,
                            "predict_proba": {},
                            "facteurs_cles": [f"Erreur lors de l'analyse : {e}"],
                            "status": "ERROR",
                        }


                # 5. Préparation du prompt système
                profil_json = json.dumps(profile, ensure_ascii=False)
                ml_json = json.dumps(ml_resultat, ensure_ascii=False) if ml_resultat else "null"

                system_prompt = (
                    "Tu es l'assistant académique ORIENT'IA de l'ISPM. Tu es autonome.\n"
                    f"Voici le profil de l'étudiant avec qui tu parles : {profil_json}\n\n"
                    f"RÉSULTAT DE L'ANALYSE ML (déjà calculé, NE PAS recalculer) :\n{ml_json}\n\n"
                    "RÈGLE IMPÉRATIVE : Base ta synthèse sur le résultat ML et les résultats RAG ci-dessus. Utilise tes autres outils "
                    "uniquement si des informations complémentaires sont nécessaires.\n"
                    "Formate ta réponse finale selon ces 3 sections :\n"
                    "**Recommandation / Synthèse**\n[Ta synthèse]\n\n"
                    "**Justification & Outils**\n[Explications basées sur le score ML et les outils utilisés]\n\n"
                    "**Sources & Citations**\n[Les documents cités]\n\n"
                    f"{DISCLAIMER_ISPM}"
                )

                # 6. Création et exécution de l'agent LangGraph
                agent_executor = create_agent(
                    model=llm,
                    tools=self.tools,
                    system_prompt=system_prompt
                )

                response_state = agent_executor.invoke({
                    "messages": [("user", user_input)]
                })

                # Extraction de l'historique des outils de l'agent
                messages = response_state.get("messages", [])
                for msg in messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_calls_history.append({
                                "tool_name": tool_call.get("name"),
                                "tool_arguments": tool_call.get("args")
                            })

                reponse_finale = _extraire_texte_reponse(messages[-1].content) if messages else ""
                
                # SUCCÈS : On retourne directement le résultat (Sortie de la boucle)
                return {
                    "response": reponse_finale,
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                    "trace": {
                        "trace_id": trace_id,
                        "tool_calls": tool_calls_history,
                        "ml_resultat": ml_resultat,
                        "erreur_agent": None,
                    }
                }

            except Exception as e:
                # 7. GESTION DES ERREURS DANS LA BOUCLE
                if _est_erreur_quota(e):
                    print(f"⚠️ [QUOTA DÉPASSÉ] Rotation demandée. Tentative {tentatives + 1}/{max_retries}")
                    key_manager.rotate_key(current_key)
                    tentatives += 1
                    continue # On boucle pour réessayer avec la nouvelle clé
                
                else:
                    # Erreur inconnue de l'IA (pas un quota) : On déclenche le mode dégradé normal
                    print(f"[ERREUR AGENT] {e}")
                    return {
                        "response": _construire_reponse_fallback(ml_resultat, rag_resultats, "erreur inattendue du moteur IA"),
                        "latency_ms": round((time.time() - start_time) * 1000, 2),
                        "trace": {
                            "trace_id": trace_id,
                            "tool_calls": tool_calls_history,
                            "ml_resultat": ml_resultat,
                            "erreur_agent": str(e),
                        }
                    }

# =====================================================================
# ZONE DE TEST (MAIN) reste inchangée
# =====================================================================