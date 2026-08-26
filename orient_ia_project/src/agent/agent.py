"""Agent conversationnel ORIENT'IA — orchestration LLM + outils."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .tools import TOOL_DEFINITIONS, TOOL_DISPATCH
from ..observability.logger import log_tool_call


class OrientAgent:
    """Agent conversationnel qui orchestre RAG, ML, règles et ontologie."""

    def __init__(
        self,
        retriever: Any = None,
        predictor: Any = None,
        symbolic: Any = None,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        self.retriever = retriever
        self.predictor = predictor
        self.symbolic = symbolic
        self.model = model
        self.api_key = api_key
        self.client = None
        self.offline_mode = True
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                self.offline_mode = False
            except Exception:
                self.offline_mode = True
        self.conversation_history: list[dict] = []

    def _call_llm(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Appel LLM avec gestion des tool calls."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Exécute un outil et retourne le résultat JSON."""
        func = TOOL_DISPATCH.get(tool_name)
        if func is None:
            return json.dumps({"error": f"Outil inconnu: {tool_name}"}, ensure_ascii=False)

        # Injection des dépendances
        if tool_name in ("rechercher_formation", "verifier_prerequis", "comparer_parcours",
                         "rechercher_competences", "identifier_debouches"):
            tool_args["retriever"] = self.retriever
            tool_args["symbolic"] = self.symbolic
        elif tool_name in ("analyser_profil_ml", "calculer_score_adequation"):
            tool_args["predictor"] = self.predictor
        elif tool_name in ("verifier_adequation_profil", "get_resume_filiere", "lister_filieres"):
            tool_args["symbolic"] = self.symbolic

        try:
            result = func(**tool_args)
        except Exception as e:
            result = json.dumps({"error": str(e)}, ensure_ascii=False)

        return result

    def _build_rag_context(self, query: str) -> str:
        """Récupère les passages pertinents du corpus via le retriever."""
        if self.retriever is None:
            return ""
        results = self.retriever.retrieve(query, top_k=5)
        if not results:
            return ""
        lines = []
        for r in results:
            source = r.get("source", "inconnu")
            text = r.get("text", "")
            score = r.get("score", 0)
            lines.append(f"[Source: {source} | pertinence: {score:.2f}]\n{text}")
        return "\n\n".join(lines)

    def _build_ml_summary(self, profil: dict) -> str:
        """Lance le modèle ML et retourne un résumé texte."""
        if self.predictor is None:
            return ""
        result = self.predictor.predict(profil)
        if not result:
            return ""
        lines = ["Analyse ML du profil :"]
        for rec in result.get("recommandations", []):
            lines.append(f"- {rec['parcours']} (score: {rec['score']:.2f})")
        if result.get("facteurs"):
            lines.append("Facteurs influents :")
            for f in result["facteurs"]:
                lines.append(f"  - {f}")
        return "\n".join(lines)

    def _build_symbolic_summary(self, profil: dict) -> str:
        """Vérifie les prérequis de chaque filière via l'ontologie."""
        if self.symbolic is None:
            return ""
        lines = ["Vérification ontologique des prérequis :"]
        filieres = ["ISAIA", "IGGLIA", "IMAAA", "ISMP", "ISPEN", "IST", "BTS_Informatique", "BTS_Gestion"]
        for f in filieres:
            try:
                result = self.symbolic.verifier_prerequis(f, profil)
                status = "✅ OK" if result["satisfait"] else "❌ Non satisfait"
                lines.append(f"  - {f}: {status}")
            except Exception:
                pass
        return "\n".join(lines)

    def chat(self, query: str, profile: dict | None = None) -> dict:
        """Traite une question utilisateur avec tool calling multi-tours.

        Returns:
            dict avec keys: answer, tools_used, sources_cited, latency_ms
        """
        t0 = time.time()
        tools_used: list[str] = []
        sources_cited: list[str] = []
        refused = False
        errors: list[str] = []

        # 1. Construire le contexte RAG
        rag_context = self._build_rag_context(query)

        # 2. Construire le contexte ML si profil disponible
        ml_summary = ""
        if profile:
            ml_summary = self._build_ml_summary(profile)

        # 3. Construire le contexte symbolique si disponible
        symbolic_summary = ""
        if profile:
            symbolic_summary = self._build_symbolic_summary(profile)

        # 4. Mode hors ligne — réponse basée sur RAG + ML + ontologie
        if self.offline_mode:
            final_answer = self._build_offline_answer(query, profile, rag_context, ml_summary, symbolic_summary)
            latency_ms = (time.time() - t0) * 1000
            log_tool_call(
                query=query,
                tool_name="offline_synthesis",
                tool_input={"profile": profile} if profile else {},
                tool_output={"rag": len(rag_context), "ml": len(ml_summary), "symbolic": len(symbolic_summary)},
                latency_ms=latency_ms,
                profile=profile,
                retrieved_passages=rag_context[:500] if rag_context else None,
                ml_input=profile,
                ml_output=ml_summary if ml_summary else None,
                final_answer=final_answer,
                errors=["Mode hors ligne — clé API non configurée"],
                refused=refused,
            )
            return {
                "answer": final_answer,
                "tools_used": ["rechercher_formation", "analyser_profil_ml", "verifier_adequation_profil"] if profile else ["rechercher_formation"],
                "sources_cited": sources_cited,
                "latency_ms": latency_ms,
            }

        # 5. Mode en ligne — appel LLM avec tool calling
        user_prompt = build_user_prompt(query, profile or {}, rag_context, ml_summary)
        if symbolic_summary:
            user_prompt += f"\n\n## Vérification ontologique\n{symbolic_summary}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_prompt})

        for _ in range(5):
            response = self._call_llm(messages, tools=TOOL_DEFINITIONS)
            choice = response.choices[0]

            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                final_answer = choice.message.content or ""
                self.conversation_history.append({"role": "user", "content": user_prompt})
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]

                latency_ms = (time.time() - t0) * 1000
                log_tool_call(
                    query=query,
                    tool_name=", ".join(tools_used) if tools_used else None,
                    tool_input={"profile": profile} if profile else {},
                    tool_output={"tools_used": tools_used},
                    latency_ms=latency_ms,
                    profile=profile,
                    retrieved_passages=rag_context[:500] if rag_context else None,
                    ml_input=profile,
                    ml_output=ml_summary if ml_summary else None,
                    final_answer=final_answer,
                    errors=errors,
                    refused=refused,
                )
                return {
                    "answer": final_answer,
                    "tools_used": tools_used,
                    "sources_cited": sources_cited,
                    "latency_ms": latency_ms,
                }

            messages.append({"role": "assistant", "content": None, "tool_calls": choice.message.tool_calls})

            for tool_call in choice.message.tool_calls:
                fn = tool_call.function
                tool_name = fn.name
                try:
                    tool_args = json.loads(fn.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tools_used.append(tool_name)
                tool_result = self._execute_tool(tool_name, tool_args)

                try:
                    parsed = json.loads(tool_result)
                    if isinstance(parsed, dict) and "source" in parsed:
                        sources_cited.append(parsed["source"])
                except (json.JSONDecodeError, TypeError):
                    pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        latency_ms = (time.time() - t0) * 1000
        final_answer = "Je n'ai pas pu compléter l'analyse. Veuillez reformuler ou consulter un conseiller."
        refused = True
        errors.append("Boucle tool calling dépassée (5 itérations)")
        log_tool_call(
            query=query,
            tool_name=", ".join(tools_used),
            tool_input={},
            tool_output={},
            latency_ms=latency_ms,
            profile=profile,
            final_answer=final_answer,
            errors=errors,
            refused=refused,
        )
        return {
            "answer": final_answer,
            "tools_used": tools_used,
            "sources_cited": sources_cited,
            "latency_ms": latency_ms,
        }

    def _build_offline_answer(self, query: str, profile: dict | None, rag_context: str, ml_summary: str, symbolic_summary: str) -> str:
        """Construit une réponse en mode hors ligne (sans LLM)."""
        parts = []

        parts.append("**ORIENT-IA — Mode demonstration (sans LLM)**\n")
        parts.append("ORIENT-IA est un outil d'aide a l'orientation. Ses recommandations ne remplacent ni l'avis d'un conseiller pedagogique ni une decision officielle d'admission.\n")

        if rag_context:
            parts.append("**Sources documentaires trouvees :**\n")
            lines = rag_context.split("\n\n")[:3]
            for line in lines:
                parts.append(f"{line}\n")
            parts.append("")

        if ml_summary:
            parts.append(f"**Analyse ML :**\n{ml_summary}\n")

        if symbolic_summary:
            parts.append(f"**Verification ontologique :**\n{symbolic_summary}\n")

        query_lower = query.lower()
        if any(w in query_lower for w in ["isaia", "informatique", "ordi"]):
            parts.append("**Concernant l'ISAIA (Informatique Appliquee) :** Licence de 3 ans formant des informaticiens pour la sante. Pre-requis : Bac S avec 10/20 minimum en maths/physique.")
        elif any(w in query_lower for w in ["igglia", "geologie"]):
            parts.append("**Concernant l'IGGLIA (Geologie) :** Licence de 3 ans en geologie appliquee a la sante. Pre-requis : Bac S avec 10/20 en physico-chimie et SVT.")
        elif any(w in query_lower for w in ["imaaa", "math"]):
            parts.append("**Concernant l'IMAAA (Mathematiques) :** Licence de 3 ans en mathematiques appliquees. Pre-requis : Bac S Maths avec 12/20 minimum.")
        elif any(w in query_lower for w in ["ismp", "medecine", "preventive"]):
            parts.append("**Concernant l'ISMP (Medecine Preventive) :** Licence de 3 ans en medecine preventive. Pre-requis : Bac S/Science Exp avec 12/20.")
        elif any(w in query_lower for w in ["ispen", "nutrit", "aliment"]):
            parts.append("**Concernant l'ISPEN (Education Nutritive) :** Licence de 3 ans en nutrition. Pre-requis : Bac S/SVT avec 10/20.")
        elif any(w in query_lower for w in ["compar", "differ"]):
            parts.append("**Pour comparer les parcours**, veuillez poser une question precise (ex: 'Compare ISAIA et IGGLIA').")
        elif any(w in query_lower for w in ["pre-requis", "condition", "admission"]):
            parts.append("**Regles d'admission :** Bac ou equivalent, notes minimales par filiere, tests d'admission selon la formation.")
        else:
            parts.append("Posez une question sur les formations ISPM (ISAIA, IGGLIA, IMAAA, ISMP, ISPEN, IST, etc.) pour obtenir une reponse detaillee.")

        if profile:
            parts.append(f"\n**Profil analyse :** {', '.join(f'{k}={v}' for k, v in profile.items() if v)}")

        return "\n".join(parts)

    def reset(self):
        """Reinitialise l'historique conversation."""
        self.conversation_history = []
