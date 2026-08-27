"""
src/agent/tools.py
===================
Module de définition des contrats et stubs d'outils pour l'assistant ORIENT'IA.

Ce fichier définit l'interface formelle (Tool Calling) utilisée par l'agent LLM.
Tant que la clé et la structure des dictionnaires retournés restent inchangées,
les modules ML, RAG et IA Symbolique peuvent remplacer ces stubs par leur logique finale.
"""
from langchain_core.tools import tool
from typing import Dict, List, Any, Optional

# ==============================================================================
# 1. OUTIL MACHINE LEARNING : ANALYSER PROFIL ML
# ==============================================================================
@tool
def analyser_profil_ml(profil: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consomme les caractéristiques déclarées de l'étudiant et retourne les prédictions
    du modèle Machine Learning (probabilités d'adéquation et recommandations).

    Args:
        profil (Dict[str, Any]): Dictionnaire du profil candidat.
            Exemple:
            {
                "bac": "C",
                "notes": {"maths": 15.0, "physique": 14.0, "algo": 16.0},
                "centres_interet": ["développement web", "intelligence artificielle"],
                "competences": ["python", "sql"]
            }

    Returns:
        Dict[str, Any]: Contrat d'adéquation ML normalisé :
            - filiere_recommandee (str): Nom/Code de la filière principale (ex: "GLSI").
            - score_adequation (float): Score de confiance global entre 0.0 et 1.0.
            - predict_proba (Dict[str, float]): Probabilités associées à chaque filière.
            - facteurs_cles (List[str]): Facteurs prédictifs explicatifs (Explicabilité / XAI).
            - status (str): "SUCCESS", "INCOMPLETE_PROFILE", ou "ERROR".
    """
    if not profil or not isinstance(profil, dict):
        return {
            "filiere_recommandee": "INCONNUE",
            "score_adequation": 0.0,
            "predict_proba": {},
            "facteurs_cles": ["Profil non renseigné ou invalide"],
            "status": "INCOMPLETE_PROFILE"
        }

    # --------------------------------------------------------------------------
    # TODO: Remplacer par l'inférence réelle (ex: joblib.load("models/model.joblib"))
    # --------------------------------------------------------------------------
    notes = profil.get("notes", {})
    math_note = notes.get("maths", 10.0)
    algo_note = notes.get("algo", 10.0)

    if math_note >= 14.0 or algo_note >= 14.0:
        filiere_rec = "Génie Logiciel & Systèmes d'Information (GLSI)"
        probas = {"GLSI": 0.58, "IA_DS": 0.36, "RSI": 0.06}
        score = 0.58
        facteurs = [
            "Solides compétences logiques et mathématiques",
            "Appétence déclarée pour la programmation"
        ]
    else:
        filiere_rec = "Réseaux & Systèmes Informatiques (RSI)"
        probas = {"RSI": 0.45, "GLSI": 0.35, "IA_DS": 0.20}
        score = 0.45
        facteurs = [
            "Profil équilibré",
            "Adéquation avec l'administration systèmes & réseaux"
        ]

    return {
        "filiere_recommandee": filiere_rec,
        "score_adequation": float(score),
        "predict_proba": probas,
        "facteurs_cles": facteurs,
        "status": "SUCCESS"
    }


# ==============================================================================
# 2. OUTIL RAG / RECHERCHE DOCUMENTAIRE : RECHERCHER FORMATION
# ==============================================================================
@tool
def rechercher_formation(query: str, filiere: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Interroge la base de connaissances documentaire (index vectoriel/lexical ISPM)
    pour extraire des passages officiels vérifiés.

    Args:
        query (str): Mots-clés ou question de l'utilisateur.
        filiere (Optional[str]): Filtre optionnel par filière ("GLSI", "IA_DS", "RSI").

    Returns:
        List[Dict[str, Any]]: Liste d'extraits documentaires avec métadonnées :
            [
                {
                    "document_id": str,
                    "source_title": str,
                    "section": str,
                    "content": str,
                    "relevance_score": float
                },
                ...
            ]
    """
    if not query or not query.strip():
        return []

    # --------------------------------------------------------------------------
    # TODO: Remplacer par l'interrogation du VectorStore (ex: ChromaDB / FAISS)
    # --------------------------------------------------------------------------
    query_lower = query.lower()

    if "math" in query_lower or "prerequis" in query_lower or "bac" in query_lower:
        return [
            {
                "document_id": "ISPM_CATALOG_2026_SEC3",
                "source_title": "Conditions Générales d'Admission ISPM 2026",
                "section": "Prérequis Académiques par Série de Bac",
                "content": (
                    "L'admission en parcours scientifique exige un Bac C, D ou S. "
                    "Une moyenne supérieure ou égale à 13/20 en mathématiques est "
                    "fortement recommandée pour la filière IA & Data Science."
                ),
                "relevance_score": 0.94
            },
            {
                "document_id": "ISPM_MAQUETTE_GLSI_2026",
                "source_title": "Maquette Pédagogique GLSI",
                "section": "Modules de Spécialité L3 / M1",
                "content": (
                    "La filière GLSI couvre l'Algorithmique avancée, le Génie Logiciel, "
                    "l'Architecture Web/Mobile et la gestion de bases de données NoSQL."
                ),
                "relevance_score": 0.88
            }
        ]

    return [
        {
            "document_id": "ISPM_GENERAL_GUIDE",
            "source_title": "Guide de l'Étudiant ISPM",
            "section": "Présentation Générale des Cursus",
            "content": (
                "L'ISPM propose des formations académiques en Génie Logiciel (GLSI), "
                "Intelligence Artificielle & Data Science (IA_DS) et Réseaux (RSI)."
            ),
            "relevance_score": 0.75
        }
    ]


# ==============================================================================
# 3. OUTIL IA SYMBOLIQUE / ONTOLOGIE : VÉRIFIER PRÉREQUIS
# ==============================================================================

@tool
def verifier_prerequis(parcours: str, profil: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vérifie formellement l'éligibilité d'un candidat selon les règles strictes
    et l'ontologie académique de l’établissement.

    Args:
        parcours (str): Identifiant du parcours ciblé (ex: "GLSI", "IA_DS", "RSI").
        profil (Dict[str, Any]): Dictionnaire du profil candidat (Bac, notes, etc.).

    Returns:
        Dict[str, Any]: Rapport formel d'éligibilité :
            - parcours (str): Code de la filière contrôlée.
            - eligible (bool): True si toutes les conditions formelles sont remplies.
            - prerequis_valides (List[str]): Conditions confirmées.
            - prerequis_manquants (List[str]): Éléments bloquants ou critères non satisfaits.
            - conditions_particulieres (List[str]): Avertissements ou clauses.
            - statut_regle (str): "VALIDE", "REJETE", ou "INFORMATION_INCOMPLETE".
    """
    bac = profil.get("bac", "").upper() if profil else ""
    notes = profil.get("notes", {}) if profil else {}
    math_note = notes.get("maths", 0.0)

    # --------------------------------------------------------------------------
    # TODO: Remplacer par les requêtes Ontologie / Graphe de connaissances (SPARQL)
    # --------------------------------------------------------------------------
    if not bac:
        return {
            "parcours": parcours,
            "eligible": False,
            "prerequis_valides": [],
            "prerequis_manquants": ["Série du Baccalauréat non renseignée"],
            "conditions_particulieres": ["Préciser la série du Bac (C, D, S, etc.)"],
            "statut_regle": "INFORMATION_INCOMPLETE"
        }

    valides = []
    manquants = []

    if bac in ["C", "S", "D"]:
        valides.append(f"Baccalauréat Scientifique ({bac}) conforme")
    else:
        manquants.append(f"Baccalauréat {bac} non prioritaire pour les cursus scientifiques")

    if parcours.upper() == "IA_DS" and math_note < 13.0:
        manquants.append("Moyenne en Mathématiques < 13/20 exigée pour IA & Data Science")
    elif math_note >= 10.0:
        valides.append("Niveau de Mathématiques suffisant (>= 10/20)")

    is_eligible = len(manquants) == 0

    return {
        "parcours": parcours,
        "eligible": is_eligible,
        "prerequis_valides": valides,
        "prerequis_manquants": manquants,
        "conditions_particulieres": [
            "Éligibilité indicative sous réserve de validation par la commission d'admission"
        ],
        "statut_regle": "VALIDE" if is_eligible else "REJETE"
    }


# ==============================================================================
# 4. OUTIL COMPARATIF : COMPARER PARCOURS
# ==============================================================================
@tool
def comparer_parcours(a: str, b: str) -> Dict[str, Any]:
    """
    Effectue une comparaison neutre et structurée entre deux filières académiques.

    Args:
        a (str): Identifiant de la première filière (ex: "GLSI").
        b (str): Identifiant de la seconde filière (ex: "IA_DS").

    Returns:
        Dict[str, Any]: Tableau comparatif structuré :
            - parcours_a (str): Nom du parcours A.
            - parcours_b (str): Nom du parcours B.
            - tronc_commun (List[str]): Modules partagés.
            - differences_cles (Dict[str, Dict[str, str]]): Comparaison axe par axe.
            - debouches_a (List[str]): Métiers visés par A.
            - debouches_b (List[str]): Métiers visés par B.
    """
    # --------------------------------------------------------------------------
    # TODO: Remplacer par la synthèse structurée depuis le référentiel d'études
    # --------------------------------------------------------------------------
    return {
        "parcours_a": a,
        "parcours_b": b,
        "tronc_commun": [
            "Bases de l'Algorithmique & Débogage",
            "Fondamentaux des Réseaux Informatiques",
            "Gestion de Projet & Anglais Technique"
        ],
        "differences_cles": {
            "Orientation principale": {
                a: "Conception logicielle, Architecture applicative, Web/Mobile & Qualité du code",
                b: "Ingénierie des données, Modélisation Machine/Deep Learning & Analytics"
            },
            "Poids des Mathématiques": {
                a: "Mathématiques appliquées (Algèbre relationnelle, Théorie des graphes)",
                b: "Mathématiques avancées (Probabilités, Optimisation, Calcul matriciel)"
            }
        },
        "debouches_a": [
            "Ingénieur Étude et Développement",
            "Architecte Logiciel / Tech Lead",
            "Développeur Fullstack / DevOps"
        ],
        "debouches_b": [
            "Data Scientist / Data Engineer",
            "Ingénieur en Intelligence Artificielle",
            "Consultant Business Intelligence & Analytics"
        ]
    }