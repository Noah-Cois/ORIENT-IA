"""Outils de l'agent conversationnel ORIENT'IA.

Chaque outil est une fonction Python callable par le LLM via function calling.
Minimum requis : 3 outils réels. On en implémente 9.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Outils RAG — recherche dans le corpus
# ---------------------------------------------------------------------------

def rechercher_formation(query: str, retriever: Any = None) -> str:
    """Recherche des formations correspondant à la requête dans le corpus ISPM.

    Args:
        query: Description du parcours ou mots-clés recherchés.
        retriever: Instance de Retriever (injection de dépendance).
    Returns:
        JSON string avec les formations trouvées.
    """
    if retriever is None:
        return json.dumps({"error": "Retriever non initialisé"}, ensure_ascii=False)
    results = retriever.retrieve(query, top_k=5)
    return json.dumps(results, ensure_ascii=False, default=str)


def verifier_prerequis(parcours: str, retriever: Any = None, symbolic: Any = None) -> str:
    """Vérifie les prérequis officiels d'un parcours donné.

    Utilise le moteur symbolique si disponible, sinon le RAG.

    Args:
        parcours: Nom ou identifiant du parcours (ex: 'ISAIA', 'IGGLIA').
        retriever: Instance de Retriever.
        symbolic: Instance de SymbolicEngine.
    Returns:
        JSON string avec les prérequis.
    """
    if symbolic is not None:
        try:
            prerequis = symbolic.get_prerequis_filiere(parcours)
            return json.dumps({
                "parcours": parcours,
                "source": "ontologie",
                "prerequis": prerequis
            }, ensure_ascii=False)
        except Exception:
            pass
    if retriever is None:
        return json.dumps({"error": "Retriever non initialisé"}, ensure_ascii=False)
    query = f"prérequis {parcours} conditions d'admission"
    results = retriever.retrieve(query, top_k=3)
    return json.dumps({"parcours": parcours, "source": "corpus", "prerequis": results}, ensure_ascii=False, default=str)


def comparer_parcours(a: str, b: str, retriever: Any = None, symbolic: Any = None) -> str:
    """Compare deux parcours sur la base du corpus documentaire.

    Args:
        a: Premier parcours.
        b: Deuxième parcours.
        retriever: Instance de Retriever.
        symbolic: Instance de SymbolicEngine.
    Returns:
        JSON string avec la comparaison.
    """
    result = {"parcours_a": a, "parcours_b": b}

    if symbolic is not None:
        try:
            result["resume_a"] = symbolic.get_resume_filiere(a)
            result["resume_b"] = symbolic.get_resume_filiere(b)
            result["source"] = "ontologie"
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            pass

    if retriever is None:
        return json.dumps({"error": "Retriever non initialisé"}, ensure_ascii=False)
    results_a = retriever.retrieve(a, top_k=3)
    results_b = retriever.retrieve(b, top_k=3)
    result["sources_a"] = results_a
    result["sources_b"] = results_b
    result["source"] = "corpus"
    return json.dumps(result, ensure_ascii=False, default=str)


def rechercher_competences(query: str, retriever: Any = None, symbolic: Any = None) -> str:
    """Recherche les compétences associées à un domaine ou une formation.

    Args:
        query: Domaine ou thème de compétences.
        retriever: Instance de Retriever.
        symbolic: Instance de SymbolicEngine.
    Returns:
        JSON string avec les compétences trouvées.
    """
    if symbolic is not None:
        try:
            filieres = symbolic.get_all_filieres()
            all_competences = {}
            for f_id in ["ISAIA", "IGGLIA", "IMAAA", "ISMP", "ISPEN", "IST", "ISPM_SantePublique", "BTS_Informatique", "BTS_Gestion"]:
                comps = symbolic.get_competences_filiere(f_id)
                if comps:
                    all_competences[f_id] = comps
            return json.dumps({
                "query": query,
                "source": "ontologie",
                "competences_par_filiere": all_competences
            }, ensure_ascii=False)
        except Exception:
            pass
    if retriever is None:
        return json.dumps({"error": "Retriever non initialisé"}, ensure_ascii=False)
    results = retriever.retrieve(f"compétences {query}", top_k=5)
    return json.dumps(results, ensure_ascii=False, default=str)


def identifier_debouches(parcours: str, retriever: Any = None, symbolic: Any = None) -> str:
    """Identifie les débouchés professionnels d'un parcours.

    Args:
        parcours: Nom du parcours.
        retriever: Instance de Retriever.
        symbolic: Instance de SymbolicEngine.
    Returns:
        JSON string avec les débouchés.
    """
    if symbolic is not None:
        try:
            metiers = symbolic.get_metiers_filiere(parcours)
            return json.dumps({
                "parcours": parcours,
                "source": "ontologie",
                "debouches": metiers
            }, ensure_ascii=False)
        except Exception:
            pass
    if retriever is None:
        return json.dumps({"error": "Retriever non initialisé"}, ensure_ascii=False)
    results = retriever.retrieve(f"débouchés métiers {parcours}", top_k=3)
    return json.dumps({"parcours": parcours, "source": "corpus", "debouches": results}, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Outils ML — analyse du profil
# ---------------------------------------------------------------------------

def analyser_profil_ml(profil: dict, predictor: Any = None) -> str:
    """Analyse le profil d'un candidat via le modèle ML et retourne des recommandations.

    Args:
        profil: Dictionnaire du profil (age, niveau, mention, interets, etc.)
        predictor: Instance de ModelPredictor.
    Returns:
        JSON string avec les recommandations ML.
    """
    if predictor is None:
        return json.dumps({"error": "Modèle ML non initialisé"}, ensure_ascii=False)
    result = predictor.predict(profil)
    return json.dumps(result, ensure_ascii=False, default=str)


def calculer_score_adequation(profil: dict, parcours: str, predictor: Any = None) -> str:
    """Calcule un score d'adéquation entre un profil et un parcours spécifique.

    Args:
        profil: Dictionnaire du profil.
        parcours: Nom du parcours cible.
        predictor: Instance de ModelPredictor.
    Returns:
        JSON string avec le score et les facteurs.
    """
    if predictor is None:
        return json.dumps({"error": "Modèle ML non initialisé"}, ensure_ascii=False)
    result = predictor.predict(profil, target_parcours=parcours)
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Outils symboliques — logique déterministe
# ---------------------------------------------------------------------------

def verifier_adequation_profil(parcours: str, profil: dict, symbolic: Any = None) -> str:
    """Vérifie si un profil satisfait les prérequis d'un parcours via l'ontologie.

    Outil déterministe : pas de ML, pas de LLM, juste la logique pure.

    Args:
        parcours: Identifiant du parcours (ex: 'ISAIA').
        profil: Dict avec 'mention' et 'moyenne'.
        symbolic: Instance de SymbolicEngine.
    Returns:
        JSON string avec le résultat de la vérification.
    """
    if symbolic is None:
        return json.dumps({"error": "Moteur symbolique non initialisé"}, ensure_ascii=False)
    try:
        result = symbolic.verifier_prerequis(parcours, profil)
        return json.dumps({
            "parcours": parcours,
            "source": "ontologie",
            "satisfait": result["satisfait"],
            "details": result["details"]
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_resume_filiere(parcours: str, symbolic: Any = None) -> str:
    """Retourne un résumé complet d'une filière depuis l'ontologie.

    Args:
        parcours: Identifiant du parcours (ex: 'ISAIA').
        symbolic: Instance de SymbolicEngine.
    Returns:
        JSON string avec le résumé (matières, compétences, prérequis, métiers).
    """
    if symbolic is None:
        return json.dumps({"error": "Moteur symbolique non initialisé"}, ensure_ascii=False)
    try:
        resume = symbolic.get_resume_filiere(parcours)
        return json.dumps(resume, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def lister_filieres(symbolic: Any = None) -> str:
    """Liste toutes les filières disponibles dans l'ontologie.

    Args:
        symbolic: Instance de SymbolicEngine.
    Returns:
        JSON string avec la liste des filières.
    """
    if symbolic is None:
        return json.dumps({"error": "Moteur symbolique non initialisé"}, ensure_ascii=False)
    try:
        filieres = symbolic.get_all_filieres()
        return json.dumps({"filieres": filieres, "source": "ontologie"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Outils explicatifs
# ---------------------------------------------------------------------------

def expliquer_recommandation(
    parcours: str,
    profil: dict,
    sources_ml: str = "",
    sources_rag: str = "",
    regles: str = "",
) -> str:
    """Génère une explication structurée de la recommandation avec traçabilité des sources.

    Args:
        parcours: Parcours recommandé.
        profil: Profil du candidat.
        sources_ml: Résultat brut du modèle ML.
        sources_rag: Passages RAG utilisés.
        regles: Règles pédagogiques appliquées.
    Returns:
        JSON string structurant l'explication par source.
    """
    return json.dumps(
        {
            "parcours": parcours,
            "profil_resume": {k: v for k, v in profil.items() if v},
            "facteurs_ml": sources_ml or "Non utilisé",
            "sources_documentaires": sources_rag or "Non consultées",
            "regles_pedagogiques": regles or "Non vérifiées",
            "note": "Cette explication est une synthèse. Consultez un conseiller pédagogique pour une décision officielle.",
        },
        ensure_ascii=False,
        default=str,
    )


# ---------------------------------------------------------------------------
# Registre des outils (pour le function calling)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "rechercher_formation",
            "description": "Recherche des formations ISPM correspondant à une requête dans le corpus documentaire.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Mots-clés ou description du parcours recherché"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verifier_prerequis",
            "description": "Vérifie les prérequis et conditions d'admission officiels d'un parcours via l'ontologie.",
            "parameters": {
                "type": "object",
                "properties": {"parcours": {"type": "string", "description": "Nom du parcours (ex: ISAIA, IGGLIA)"}},
                "required": ["parcours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comparer_parcours",
            "description": "Compare deux parcours académiques (matières, compétences, prérequis, métiers).",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "Premier parcours"},
                    "b": {"type": "string", "description": "Deuxième parcours"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rechercher_competences",
            "description": "Recherche les compétences associées à un domaine ou une formation.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Domaine ou thème"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identifier_debouches",
            "description": "Identifie les débouchés professionnels d'un parcours via l'ontologie.",
            "parameters": {
                "type": "object",
                "properties": {"parcours": {"type": "string", "description": "Nom du parcours"}},
                "required": ["parcours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyser_profil_ml",
            "description": "Analyse le profil d'un candidat via le modèle ML pour recommander des parcours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profil": {
                        "type": "object",
                        "description": "Profil du candidat (age, niveau, mention, interets, etc.)",
                    }
                },
                "required": ["profil"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculer_score_adequation",
            "description": "Calcule un score d'adéquation entre un profil et un parcours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profil": {"type": "object", "description": "Profil du candidat"},
                    "parcours": {"type": "string", "description": "Parcours cible"},
                },
                "required": ["profil", "parcours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verifier_adequation_profil",
            "description": "Vérifie déterministiquement si un profil satisfait les prérequis d'un parcours (ontologie OWL).",
            "parameters": {
                "type": "object",
                "properties": {
                    "parcours": {"type": "string", "description": "Identifiant du parcours (ex: ISAIA)"},
                    "profil": {"type": "object", "description": "Profil avec mention et moyenne"},
                },
                "required": ["parcours", "profil"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_resume_filiere",
            "description": "Retourne un résumé complet d'une filière depuis l'ontologie (matières, compétences, prérequis, métiers).",
            "parameters": {
                "type": "object",
                "properties": {
                    "parcours": {"type": "string", "description": "Identifiant du parcours (ex: ISAIA)"},
                },
                "required": ["parcours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lister_filieres",
            "description": "Liste toutes les filières disponibles dans l'ontologie ISPM.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expliquer_recommandation",
            "description": "Génère une explication structurée de la recommandation avec traçabilité des sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parcours": {"type": "string"},
                    "profil": {"type": "object"},
                    "sources_ml": {"type": "string"},
                    "sources_rag": {"type": "string"},
                    "regles": {"type": "string"},
                },
                "required": ["parcours", "profil"],
            },
        },
    },
]

# Mapping nom → fonction pour l'dispatch
TOOL_DISPATCH = {
    "rechercher_formation": rechercher_formation,
    "verifier_prerequis": verifier_prerequis,
    "comparer_parcours": comparer_parcours,
    "rechercher_competences": rechercher_competences,
    "identifier_debouches": identifier_debouches,
    "analyser_profil_ml": analyser_profil_ml,
    "calculer_score_adequation": calculer_score_adequation,
    "verifier_adequation_profil": verifier_adequation_profil,
    "get_resume_filiere": get_resume_filiere,
    "lister_filieres": lister_filieres,
    "expliquer_recommandation": expliquer_recommandation,
}
