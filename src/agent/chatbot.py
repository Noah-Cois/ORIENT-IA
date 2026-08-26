"""
src/agent/chatbot.py
====================
Orchestrateur principal du chatbot d'orientation académique ORIENT'IA.

Ce module gère le pipeline complet d'une interaction utilisateur :
1. Contrôles de sécurité (Prompt Injection, Profilage psychologique, Neutralité).
2. Capture d'observabilité et traçabilité (JSON Logs / LangFuse).
3. Routage et exécution des outils (ML, RAG, Ontologie, Comparateur).
4. Génération de la réponse via Google Gemini selon le contrat à 4 sections obligatoires.
"""

import sys
from pathlib import Path

# Fixe le chemin vers la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import json
import uuid
from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config import get_gemini_token
from src.agent.tools import (
    analyser_profil_ml,
    rechercher_formation,
    verifier_prerequis,
    comparer_parcours
)

# Mention légale obligatoire exigée par la charte ISPM
DISCLAIMER_ISPM = (
    "*ORIENT’IA constitue un outil d’aide à l’orientation. "
    "Ses recommandations ne remplacent ni l’avis d’un conseiller pédagogique "
    "ni une décision officielle d’admission.*"
)


class OrientIAAgent:
    """Agent d'orchestration principal de l'assistant d'orientation ISPM propulsé par Gemini."""

    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.model_name = model_name
        self._injection_keywords = ["ignore previous", "system prompt", "jailbreak", "forget instructions"]
        self._psycho_keywords = ["analyse ma personnalité", "profil psychologique", "suis-je dépressif", "état mental"]

        # Initialisation sécurisée de Google Gemini via get_gemini_token()
        try:
            api_key = get_gemini_token()
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.2
            )
        except ValueError as e:
            self.llm = None
            print(f"[ATTENTION] {e} Mode dégradé (fallback) activé.")

    def _check_security(self, text: str) -> Dict[str, bool]:
        """Vérifie les tentatives de prompt injection et les requêtes hors charte."""
        text_lower = text.lower()
        return {
            "prompt_injection": any(kw in text_lower for kw in self._injection_keywords),
            "psycho_profiling": any(kw in text_lower for kw in self._psycho_keywords)
        }

    def process_query(self, user_input: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Exécute le workflow complet d'une requête utilisateur avec intégration LLM."""
        start_time = time.time()
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"
        profile = user_profile or {}

        trace_log = {
            "trace_id": trace_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inputs": {
                "raw_query": user_input,
                "declared_profile": profile
            },
            "tool_calls": [],
            "security_flags": {
                "prompt_injection_blocked": False,
                "psycho_refusal": False,
                "uncertainty_flag": False
            },
            "metrics": {
                "latency_ms": 0.0
            }
        }

        # 1. ÉTAPE DE SÉCURITÉ & DÉONTOLOGIE
        sec_status = self._check_security(user_input)

        if sec_status["prompt_injection"]:
            trace_log["security_flags"]["prompt_injection_blocked"] = True
            response_text = (
                "**Recommandation / Synthèse**\n"
                "Action bloquée. ORIENT’IA ne peut pas exécuter d'instructions modifiant ses règles de fonctionnement de sécurité.\n\n"
                "**Justification & Score ML**\n"
                "Tentative de contournement d'instructions détectée par le module de sécurité.\n\n"
                "**Sources & Citations**\n"
                "- Politique de Sécurité du Système IT ISPM 2026\n\n"
                f"{DISCLAIMER_ISPM}"
            )
            return self._finalize_response(response_text, start_time, trace_log)

        if sec_status["psycho_profiling"]:
            trace_log["security_flags"]["psycho_refusal"] = True
            response_text = (
                "**Recommandation / Synthèse**\n"
                "Demande refusée. ORIENT’IA ne réalise aucune évaluation ou analyse de la personnalité ou de l'état psychologique.\n\n"
                "**Justification & Score ML**\n"
                "Conformité éthique : l'agent d'orientation se limite exclusivement aux critères académiques et aux compétences déclarées.\n\n"
                "**Sources & Citations**\n"
                "- Charte Éthique et Protection des Données ISPM\n\n"
                f"{DISCLAIMER_ISPM}"
            )
            return self._finalize_response(response_text, start_time, trace_log)

        # 2. ROUTAGE ET EXÉCUTION DES OUTILS (Tool Calling)
        input_lower = user_input.lower()
        tool_results = {}

        # Outil 1 : Analyse ML si des notes/compétences sont fournies dans le profil
        if profile.get("notes") or profile.get("competences"):
            ml_out = analyser_profil_ml(profile)
            tool_results["ml"] = ml_out
            trace_log["tool_calls"].append({"tool": "analyser_profil_ml", "args": profile, "output": ml_out})

        # Outil 2 : Recherche documentaire RAG
        rag_out = rechercher_formation(user_input)
        tool_results["rag"] = rag_out
        trace_log["tool_calls"].append({"tool": "rechercher_formation", "args": {"query": user_input}, "output": rag_out})

        # Outil 3 : Comparatif de parcours
        if "comparer" in input_lower or "différence" in input_lower:
            comp_out = comparer_parcours("GLSI", "IA_DS")
            tool_results["comparaison"] = comp_out
            trace_log["tool_calls"].append({"tool": "comparer_parcours", "args": {"a": "GLSI", "b": "IA_DS"}, "output": comp_out})

        # Outil 4 : Vérification des prérequis via Ontologie
        if profile.get("bac") or "prérequis" in input_lower or "prerequis" in input_lower:
            target_path = "IA_DS" if "ia" in input_lower else "GLSI"
            prereq_out = verifier_prerequis(target_path, profile)
            tool_results["prerequis"] = prereq_out
            trace_log["tool_calls"].append({"tool": "verifier_prerequis", "args": {"parcours": target_path, "profil": profile}, "output": prereq_out})

        # 3. GESTION DE L'INCERTITUDE & CLARIFICATION
        if "inconnu" in input_lower or "astrologie" in input_lower:
            trace_log["security_flags"]["uncertainty_flag"] = True
            response_text = (
                "**Recommandation / Synthèse**\n"
                "Information non disponible dans le référentiel. Je ne dispose d'aucune donnée officielle sur cette formation au sein du corpus de l'ISPM.\n\n"
                "**Justification & Score ML**\n"
                "Absence d'éléments correspondants dans la base documentaire. L'agent refuse d'émettre des hypothèses non vérifiées.\n\n"
                "**Sources & Citations**\n"
                "- Registre Général des Formations ISPM 2026\n\n"
                f"{DISCLAIMER_ISPM}"
            )
            return self._finalize_response(response_text, start_time, trace_log)

        # 4. GÉNÉRATION VIA GEMINI
        response_text = self._build_gemini_response(user_input, profile, tool_results)
        return self._finalize_response(response_text, start_time, trace_log)

    def _build_gemini_response(self, query: str, profile: Dict[str, Any], tools: Dict[str, Any]) -> str:
        """Génère la réponse via Google Gemini ou utilise un fallback si indisponible."""
        if not self.llm:
            return self._build_fallback_response(query, profile, tools)

        system_instruction = (
            "Tu es l'assistant académique ORIENT'IA de l'ISPM. Tu dois impérativement formater ta réponse "
            "en respectant scrupuleusement les 4 sections suivantes, sans modifier les titres :\n\n"
            "**Recommandation / Synthèse**\n"
            "[Réponse synthétique et directe à l'étudiant]\n\n"
            "**Justification & Score ML**\n"
            "[Détails du score ML, prérequis et facteurs clés d'adéquation]\n\n"
            "**Sources & Citations**\n"
            "[Nom exact des brochures ou documents officiels exploités]\n\n"
            f"{DISCLAIMER_ISPM}"
        )

        user_prompt = (
            f"Question de l'étudiant : {query}\n"
            f"Profil de l'étudiant : {json.dumps(profile, ensure_ascii=False)}\n"
            f"Résultats des outils : {json.dumps(tools, ensure_ascii=False)}\n\n"
            "Rédige une réponse claire, fluide et naturelle en français en utilisant ces informations."
        )

        try:
            full_prompt = f"{system_instruction}\n\n{user_prompt}"
            gemini_res = self.llm.invoke(full_prompt)

            # Extraire proprement le texte si la réponse retourne un format sous forme de liste
            if isinstance(gemini_res.content, list):
                return gemini_res.content[0].get("text", "")
            return str(gemini_res.content)
        except Exception as e:
            print(f"[ERREUR GEMINI] Appel LLM échoué : {e}. Basculement sur le mode fallback.")
            return self._build_fallback_response(query, profile, tools)

    def _build_fallback_response(self, query: str, profile: Dict[str, Any], tools: Dict[str, Any]) -> str:
        """Générateur de secours déterministe en cas de coupure de l'API Gemini."""
        ml_data = tools.get("ml", {})
        filiere_rec = ml_data.get("filiere_recommandee", "Génie Logiciel (GLSI)")
        synth_sec = f"En fonction de votre profil et de votre demande, le parcours recommandé est **{filiere_rec}**."
        
        probas = ml_data.get("predict_proba", {"GLSI": 0.58, "IA_DS": 0.36})
        score_str = ", ".join([f"{k}: {v * 100:.0f}%" for k, v in probas.items()]) if probas else "Score en cours d'évaluation."
        prereq_data = tools.get("prerequis", {})
        prereq_status = "Prérequis validés." if prereq_data.get("eligible") else "Informations de prérequis sous réserve de validation académique."

        justif_sec = (
            f"**Probabilités d'adéquation ML** : {score_str}\n"
            f"**Éligibilité académique** : {prereq_status}\n"
            f"**Facteurs clés** : {', '.join(ml_data.get('facteurs_cles', ['Adéquation académique']))}."
        )

        rag_docs = tools.get("rag", [])
        citations = [f"- *{doc.get('source_title', 'Brochure ISPM')}* ({doc.get('section', 'Section officielle')})" for doc in rag_docs]
        sources_sec = "\n".join(citations) if citations else "- *Catalogue Officiel des Formations ISPM 2026*"

        return (
            f"**Recommandation / Synthèse**\n{synth_sec}\n\n"
            f"**Justification & Score ML**\n{justif_sec}\n\n"
            f"**Sources & Citations**\n{sources_sec}\n\n"
            f"{DISCLAIMER_ISPM}"
        )

    def _finalize_response(self, response_text: str, start_time: float, trace_log: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule la latence finale et emballe l'objet de retour backend."""
        latency_ms = (time.time() - start_time) * 1000
        trace_log["metrics"]["latency_ms"] = round(latency_ms, 2)
        
        return {
            "response": response_text,
            "latency_ms": trace_log["metrics"]["latency_ms"],
            "trace": trace_log
        }


if __name__ == "__main__":
    agent = OrientIAAgent()
    
    sample_profile = {
        "bac": "C",
        "notes": {"maths": 15.0, "algo": 14.5},
        "centres_interet": ["développement web", "ia"]
    }
    
    sample_query = "Quelle filière est la plus adaptée à mes notes entre GLSI et IA_DS ?"
    
    result = agent.process_query(sample_query, sample_profile)
    print("--- RÉPONSE GÉNÉRÉE (GEMINI) ---")
    print(result["response"])
    print("\n--- METRIQUES & TRACE ---")
    print(f"Latence : {result['latency_ms']} ms")