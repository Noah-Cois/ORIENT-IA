"""
src/agent/chatbot.py
====================
Agent Autonome ORIENT'IA propulsé par LangGraph
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

from src.utils.config import get_gemini_token
from src.agent.tools import (
    rechercher_documentation_ispm,
    analyser_profil_ml,
    rechercher_formation,
    verifier_prerequis,
    comparer_parcours,
    traduire_profil_vers_vocabulaire_ml
)

DISCLAIMER_ISPM = (
    "*ORIENT’IA constitue un outil d’aide à l’orientation. "
    "Ses recommandations ne remplacent ni l’avis d’un conseiller pédagogique.*"
)


def _extraire_texte_reponse(content: Union[str, List[Any]]) -> str:
    """Normalise le contenu d'un message LangChain, qui peut être
    une string simple ou une liste de blocs (cas Gemini avec métadonnées
    de citation/grounding, ex: [{'type': 'text', 'text': '...', 'extras': {...}}]).
    """
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


class OrientIAAgent:
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.model_name = model_name
        self._injection_keywords = ["ignore previous", "system prompt", "jailbreak"]
        self._psycho_keywords = ["analyse ma personnalité", "profil psychologique"]

        try:
            api_key = get_gemini_token()
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.2
            )
            # Outils laissés à la discrétion de l'agent (dépendent vraiment du
            # contexte de la question — analyser_profil_ml et rechercher_formation
            # sont volontairement RETIRÉS de cette liste : ils sont désormais
            # exécutés de façon déterministe dans process_query, voir plus bas).
            self.tools = [
                rechercher_documentation_ispm,
                verifier_prerequis,
                comparer_parcours,
            ]

        except ValueError as e:
            self.llm = None
            self.tools = []
            print(f"[ERREUR] Impossible d'initialiser Gemini : {e}")

    def _check_security(self, text: str) -> Dict[str, bool]:
        """Détecte les tentatives de contournement d'instructions (prompt
        injection) et les demandes de profilage psychologique, de façon
        distincte, pour permettre des messages de refus différenciés.
        """
        text_lower = text.lower()
        return {
            "prompt_injection": any(kw in text_lower for kw in self._injection_keywords),
            "psycho_profiling": any(kw in text_lower for kw in self._psycho_keywords),
        }

    def process_query(self, user_input: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Traite la requête avec le moteur LangGraph."""
        start_time = time.time()
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"
        profile = user_profile or {}
        tool_calls_history = []

        # 1. ÉTAPE DE SÉCURITÉ & DÉONTOLOGIE
        sec_status = self._check_security(user_input)

        if sec_status["prompt_injection"]:
            response_text = (
                "**Recommandation / Synthèse**\n"
                "Action bloquée. ORIENT’IA ne peut pas exécuter d'instructions modifiant ses règles de fonctionnement de sécurité.\n\n"
                "**Justification & Score ML**\n"
                "Tentative de contournement d'instructions détectée par le module de sécurité.\n\n"
                "**Sources & Citations**\n"
                "- Politique de Sécurité du Système IT ISPM 2026\n\n"
                f"{DISCLAIMER_ISPM}"
            )
            return {
                "response": response_text,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "trace": {
                    "trace_id": trace_id,
                    "tool_calls": [],
                    "security_flags": {"prompt_injection_blocked": True},
                },
            }

        if sec_status["psycho_profiling"]:
            response_text = (
                "**Recommandation / Synthèse**\n"
                "Demande refusée. ORIENT’IA ne réalise aucune évaluation ou analyse de la personnalité ou de l'état psychologique.\n\n"
                "**Justification & Score ML**\n"
                "Conformité éthique : l'agent d'orientation se limite exclusivement aux critères académiques et aux compétences déclarées.\n\n"
                "**Sources & Citations**\n"
                "- Charte Éthique et Protection des Données ISPM\n\n"
                f"{DISCLAIMER_ISPM}"
            )
            return {
                "response": response_text,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "trace": {
                    "trace_id": trace_id,
                    "tool_calls": [],
                    "security_flags": {"psycho_refusal": True},
                },
            }

        # 2. Mode dégradé : Gemini indisponible (clé absente/invalide)
        if not self.llm:
            return {
                "response": (
                    "**Recommandation / Synthèse**\n"
                    "Le service est momentanément indisponible (échec d'initialisation du moteur IA).\n\n"
                    "**Justification & Score ML**\n"
                    "Aucune clé API valide n'a pu être chargée pour Gemini.\n\n"
                    "**Sources & Citations**\n"
                    "- N/A\n\n"
                    f"{DISCLAIMER_ISPM}"
                ),
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "trace": {"trace_id": trace_id, "tool_calls": [], "security_flags": {"llm_unavailable": True}},
            }

        # 3. ANALYSE ML — exécutée de façon DÉTERMINISTE (pas laissée à
        # l'agent), pour garantir qu'elle a effectivement lieu et qu'elle
        # utilise bien le profil traduit dans le vocabulaire exact du modèle.
        ml_resultat = None
        if profile.get("notes") or profile.get("competences"):
            try:
                profil_traduit = traduire_profil_vers_vocabulaire_ml(profile, self.llm)
                tool_calls_history.append({
                    "tool_name": "traduire_profil_vers_vocabulaire_ml",
                    "tool_arguments": profile,
                })
                ml_resultat = analyser_profil_ml.invoke({"profil": profil_traduit})
                tool_calls_history.append({
                    "tool_name": "analyser_profil_ml",
                    "tool_arguments": profil_traduit,
                })
            except Exception as e:
                print(f"[ERREUR ML] Analyse du profil échouée : {e}")
                ml_resultat = {
                    "filiere_recommandee": "INCONNUE",
                    "score_adequation": 0.0,
                    "predict_proba": {},
                    "facteurs_cles": [f"Erreur lors de l'analyse : {e}"],
                    "status": "ERROR",
                }

        # 4. RECHERCHE RAG — également exécutée de façon déterministe, sur la
        # question brute de l'utilisateur, pour garantir un ancrage documentaire
        # systématique plutôt que dépendant du choix de l'agent.
        rag_resultats = []
        try:
            rag_resultats = rechercher_formation.invoke({"query": user_input})
            tool_calls_history.append({
                "tool_name": "rechercher_formation",
                "tool_arguments": {"query": user_input},
            })
        except Exception as e:
            print(f"[ERREUR RAG] Recherche documentaire échouée : {e}")

        # 5. Préparation du prompt système : le contexte ML/RAG est fourni
        # comme fait déjà établi, pas comme une action à décider par l'agent.
        profil_json = json.dumps(profile, ensure_ascii=False)
        ml_json = json.dumps(ml_resultat, ensure_ascii=False) if ml_resultat else "null"
        rag_json = json.dumps(rag_resultats, ensure_ascii=False)

        system_prompt = (
            "Tu es l'assistant académique ORIENT'IA de l'ISPM. Tu es autonome.\n"
            "Voici le profil de l'étudiant avec qui tu parles : " + profil_json + "\n\n"
            "RÉSULTAT DE L'ANALYSE ML (déjà calculé, NE PAS recalculer, NE PAS "
            "inventer d'autres chiffres) :\n" + ml_json + "\n\n"
            "RÉSULTATS DE LA RECHERCHE DOCUMENTAIRE RAG (déjà calculés, à citer "
            "tels quels dans la section Sources) :\n" + rag_json + "\n\n"
            "RÈGLE IMPÉRATIVE : Base ta synthèse et ta justification sur le résultat "
            "ML et les résultats RAG ci-dessus. Utilise tes autres outils "
            "(verifier_prerequis, comparer_parcours, rechercher_documentation_ispm) "
            "uniquement si des informations complémentaires sont nécessaires "
            "(ex: vérification de prérequis, comparaison explicite entre deux filières).\n"
            "Formate ta réponse finale selon ces 3 sections :\n"
            "**Recommandation / Synthèse**\n"
            "[Ta synthèse]\n\n"
            "**Justification & Outils**\n"
            "[Explications basées sur le score ML et les outils utilisés]\n\n"
            "**Sources & Citations**\n"
            "[Les documents cités]\n\n"
            f"{DISCLAIMER_ISPM}"
        )

        # 6. Création et exécution de l'agent LangGraph
        agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        response_state = agent_executor.invoke({
            "messages": [("user", user_input)]
        })

        # 7. Extraction de la traçabilité depuis l'historique des messages
        #    (s'ajoute aux appels déterministes déjà tracés ci-dessus)
        messages = response_state.get("messages", [])

        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls_history.append({
                        "tool_name": tool_call.get("name"),
                        "tool_arguments": tool_call.get("args")
                    })

        latency_ms = (time.time() - start_time) * 1000
        reponse_finale = _extraire_texte_reponse(messages[-1].content) if messages else ""

        return {
            "response": reponse_finale,
            "latency_ms": round(latency_ms, 2),
            "trace": {
                "trace_id": trace_id,
                "tool_calls": tool_calls_history,
                "ml_resultat": ml_resultat,
            }
        }


# =====================================================================
# ZONE DE TEST (MAIN)
# =====================================================================
if __name__ == "__main__":
    agent = OrientIAAgent()

    sample_profile = {
        "bac": "D",
        "notes": {"maths": 14.5, "svt": 15.0},
        "centres_interet": ["robotique", "musique"],
        "competences": ["analyse en laboratoire", "résolution de problèmes"]
    }

    sample_query = "Quelle filière est la plus adaptée à mon profil scientifique ?"

    result = agent.process_query(sample_query, sample_profile)

    print("🛠️ --- OUTILS UTILISÉS PAR L'IA ---")
    outils_utilises = result["trace"]["tool_calls"]

    if not outils_utilises:
        print("L'IA a répondu sans utiliser d'outils.")
    else:
        for i, call in enumerate(outils_utilises, 1):
            print(f"\n[{i}] Outil choisi : {call['tool_name']}")
            print(f"    ➡️ Arguments : {call['tool_arguments']}")

    print("\n" + "="*60 + "\n")
    print("💬 --- RÉPONSE FINALE GÉNÉRÉE ---")
    print(result["response"])
    print("\n" + "="*60)
    print(f"⏱️ Latence : {result['latency_ms']} ms")