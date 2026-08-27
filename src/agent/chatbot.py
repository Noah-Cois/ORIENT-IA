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
from typing import Dict, Any, Optional

# Import de LangGraph
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config import get_gemini_token
from src.agent.tools import (
    analyser_profil_ml,
    rechercher_formation,
    verifier_prerequis,
    comparer_parcours
)

DISCLAIMER_ISPM = (
    "*ORIENT’IA constitue un outil d’aide à l’orientation. "
    "Ses recommandations ne remplacent ni l’avis d’un conseiller pédagogique.*"
)

class OrientIAAgent:
    # CORRECTION ICI : On remet "gemini-3.5-flash" qui correspond à votre clé et version d'API
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
            # Nos outils
            self.tools = [analyser_profil_ml, rechercher_formation, verifier_prerequis, comparer_parcours]
            
        except ValueError as e:
            self.llm = None
            print(f"[ERREUR] Impossible d'initialiser Gemini : {e}")

    def _check_security(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self._injection_keywords + self._psycho_keywords)

    def process_query(self, user_input: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Traite la requête avec le moteur LangGraph."""
        start_time = time.time()
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"
        profile = user_profile or {}

        if self._check_security(user_input):
            return {
                "response": "Demande bloquée pour des raisons de sécurité ou de déontologie.",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "trace": {"tool_calls": []}
            }

        # 1. Préparation du prompt système avec le profil dynamique
        profil_json = json.dumps(profile, ensure_ascii=False)
        system_prompt = (
            "Tu es l'assistant académique ORIENT'IA de l'ISPM. Tu es autonome.\n"
            "Voici le profil de l'étudiant avec qui tu parles : " + profil_json + "\n\n"
            "RÈGLE IMPÉRATIVE : Utilise tes outils pour trouver l'information ou faire les calculs AVANT de répondre.\n"
            "Formate ta réponse finale selon ces 3 sections :\n"
            "**Recommandation / Synthèse**\n"
            "[Ta synthèse]\n\n"
            "**Justification & Outils**\n"
            "[Explications basées sur les outils]\n\n"
            "**Sources & Citations**\n"
            "[Les documents cités]\n\n"
            f"{DISCLAIMER_ISPM}"
        )

        # 2. Création de l'agent LangGraph
        agent_executor = create_agent(
            model=self.llm, 
            tools=self.tools, 
            system_prompt=system_prompt
        )

        # 3. Exécution de l'agent
        response_state = agent_executor.invoke({
            "messages": [("user", user_input)]
        })

        # 4. Extraction de la traçabilité depuis l'historique des messages
        tool_calls_history = []
        messages = response_state.get("messages", [])
        
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls_history.append({
                        "tool_name": tool_call.get("name"),
                        "tool_arguments": tool_call.get("args")
                    })

        latency_ms = (time.time() - start_time) * 1000
        reponse_finale = messages[-1].content

        return {
            "response": reponse_finale,
            "latency_ms": round(latency_ms, 2),
            "trace": {
                "trace_id": trace_id,
                "tool_calls": tool_calls_history
            }
        }


# =====================================================================
# ZONE DE TEST (MAIN) 
# =====================================================================
if __name__ == "__main__":
    agent = OrientIAAgent()
    
    sample_profile = {
        "bac": "C",
        "notes": {"maths": 15.0, "algo": 14.5},
        "centres_interet": ["développement web", "ia"]
    }
    
    sample_query = "Compare la filière GLSI et IA_DS, puis dis moi laquelle est la plus adaptée à mes notes."
    
    print("⏳ L'Agent LangGraph réfléchit... (patientez)\n")
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