"""
src/agent/tools.py
===================
Module de définition des contrats et stubs d'outils pour l'assistant ORIENT'IA.

Ce fichier définit l'interface formelle (Tool Calling) utilisée par l'agent LLM.
Tant que la clé et la structure des dictionnaires retournés restent inchangées,
les modules ML, RAG et IA Symbolique peuvent remplacer ces stubs par leur logique finale.
"""

from typing import Dict, List, Any, Optional
from langchain_core.tools import tool

import sys
from pathlib import Path

# Permet d'importer src.ml.predict quel que soit le point d'entrée du script
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ml.predict import predire_orientation_top3

import json
import pandas as pd


def _charger_vocabulaire_ml() -> Dict[str, List[str]]:
    """
    Charge les valeurs CATÉGORIELLES réellement vues à l'entraînement du modèle
    (data/synthetic/profils_etudiants_synthetiques.csv), pour guider la traduction
    LLM (Option B) : le modèle ne comprend QUE des combinaisons exactes déjà vues.
    """
    csv_path = ROOT_DIR / "data" / "synthetic" / "profils_etudiants_synthetiques.csv"
    df = pd.read_csv(csv_path)
    colonnes = ["serie", "matieres_fortes", "matieres_faibles", "centres_interet", "competences"]
    return {col: sorted(df[col].dropna().unique().tolist()) for col in colonnes if col in df.columns}

@tool
def traduire_profil_vers_vocabulaire_ml(profil_libre: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    Traduit un profil exprimé en langage libre (venant du chatbot/utilisateur)
    vers les valeurs EXACTES du vocabulaire appris par le modèle ML (Option B).

    Le modèle ML utilise un OneHotEncoder qui ne reconnaît QUE des combinaisons
    identiques à celles vues pendant l'entraînement (ex: "Robotique; Musique").
    Une valeur inventée par l'utilisateur (ex: "je code et j'aime la robotique")
    doit donc être ramenée à la valeur la plus proche de la liste autorisée
    AVANT d'appeler analyser_profil_ml, sinon la confiance de prédiction
    s'effondre (~14% observé vs ~87% avec une valeur exacte).

    Args:
        profil_libre: profil brut tel qu'exprimé par l'utilisateur/chatbot.
        llm: instance LLM déjà initialisée (ex: self.llm de OrientIAAgent).

    Returns:
        Dict prêt à être passé à analyser_profil_ml, avec des valeurs
        catégorielles garanties dans le vocabulaire connu du modèle.
    """
    vocab = _charger_vocabulaire_ml()

    notes = profil_libre.get("notes", {}) or {}
    moyenne_generale = profil_libre.get("moyenne_generale")
    if moyenne_generale is None:
        moyenne_generale = sum(notes.values()) / len(notes) if notes else 10.0

    prompt = f"""Tu dois traduire le profil libre d'un étudiant vers des valeurs EXACTES
issues des listes autorisées ci-dessous. Choisis dans chaque liste la valeur la plus
proche du profil. Ne modifie JAMAIS l'orthographe : copie-colle exactement la chaîne choisie.

SÉRIES AUTORISÉES : {vocab.get('serie', [])}

COMBINAISONS "MATIÈRES FORTES" AUTORISÉES (choisis la plus proche) :
{vocab.get('matieres_fortes', [])}

COMBINAISONS "MATIÈRES FAIBLES" AUTORISÉES (choisis la plus proche) :
{vocab.get('matieres_faibles', [])}

COMBINAISONS "CENTRES D'INTÉRÊT" AUTORISÉES (choisis la plus proche) :
{vocab.get('centres_interet', [])}

COMBINAISONS "COMPÉTENCES" AUTORISÉES (choisis la plus proche) :
{vocab.get('competences', [])}

PROFIL LIBRE DE L'ÉTUDIANT :
{json.dumps(profil_libre, ensure_ascii=False)}

Réponds UNIQUEMENT avec un objet JSON strict, sans texte autour, sans balises markdown,
au format exact suivant :
{{"serie": "...", "moyenne_generale": <nombre>, "matieres_fortes": "...", "matieres_faibles": "...", "centres_interet": "...", "competences": "..."}}
Chaque valeur catégorielle DOIT être copiée EXACTEMENT depuis les listes fournies ci-dessus."""

    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = content[0].get("text", "")
        content_clean = content.strip()
        if content_clean.startswith("```"):
            content_clean = content_clean.strip("`")
            content_clean = content_clean.replace("json", "", 1).strip()
        traduit = json.loads(content_clean)
        traduit.setdefault("moyenne_generale", moyenne_generale)
        return traduit
    except Exception as e:
        # Fallback : on retombe sur un mapping brut (non garanti dans le vocabulaire)
        # plutôt que de faire planter tout le pipeline.
        print(f"[ATTENTION] Échec de la traduction LLM du profil ({e}). Fallback brut utilisé.")
        return {
            "serie": profil_libre.get("serie") or profil_libre.get("bac", "D"),
            "moyenne_generale": moyenne_generale,
            "matieres_fortes": str(profil_libre.get("matieres_fortes", "")),
            "matieres_faibles": str(profil_libre.get("matieres_faibles", "")),
            "centres_interet": str(profil_libre.get("centres_interet", "")),
            "competences": str(profil_libre.get("competences", "")),
        }


# ==============================================================================
# 1. OUTIL MACHINE LEARNING : ANALYSER PROFIL ML
# ==============================================================================
@tool
def _mapper_profil_vers_input_ml(profil: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traduit le profil "libre" envoyé par le chatbot (bac, notes, listes libres)
    vers le format strict attendu par le modèle entraîné (serie, moyenne_generale,
    matieres_fortes, matieres_faibles, centres_interet, competences en strings
    séparées par '; ').

    ⚠️ LIMITE CONNUE : le modèle a appris des combinaisons EXACTES vues dans le
    CSV d'entraînement (ex: "Robotique; Musique"). Une valeur libre qui ne
    correspond à aucune combinaison connue sera traitée comme "inconnue" par
    le OneHotEncoder, et la confiance de la prédiction chutera fortement
    (voir tests : ~14% avec valeurs inventées vs ~87% avec valeurs exactes).
    Ce mapping fait de son mieux mais ne résout pas ce problème de fond —
    à traiter via une liste de choix contrainte côté frontend, ou via une
    étape de traduction par le LLM vers le vocabulaire exact du modèle.
    """
    notes = profil.get("notes", {}) or {}
    moyenne_generale = profil.get("moyenne_generale")
    if moyenne_generale is None:
        moyenne_generale = sum(notes.values()) / len(notes) if notes else 10.0

    def _to_str(valeur, sep="; "):
        if isinstance(valeur, list):
            return sep.join(str(v) for v in valeur)
        return str(valeur) if valeur is not None else ""

    return {
        "serie": profil.get("serie") or profil.get("bac", "D"),
        "moyenne_generale": float(moyenne_generale),
        "matieres_fortes": _to_str(profil.get("matieres_fortes", "")),
        "matieres_faibles": _to_str(profil.get("matieres_faibles", "")),
        "centres_interet": _to_str(profil.get("centres_interet", "")),
        "competences": _to_str(profil.get("competences", "")),
    }


@tool
def analyser_profil_ml(profil: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consomme les caractéristiques déclarées de l'étudiant et retourne les prédictions
    du modèle Machine Learning réel (RandomForest entraîné sur les profils ISPM),
    au format contrat attendu par l'agent.

    Args:
        profil (Dict[str, Any]): Dictionnaire du profil candidat.
            Exemple:
            {
                "serie": "D",
                "moyenne_generale": 14.5,
                "matieres_fortes": "Sciences Physiques et Chimiques; SVT / Biologie-Géologie",
                "matieres_faibles": "Histoire-Géographie; Anglais",
                "centres_interet": "Robotique; Musique",
                "competences": "Analyse en laboratoire; Pharmacologie; Résolution de problèmes"
            }

    Returns:
        Dict[str, Any]: Contrat d'adéquation ML normalisé :
            - filiere_recommandee (str): Code de la filière principale (ex: "PIP", "ESIIA"...).
            - score_adequation (float): Score de confiance du top 1, entre 0.0 et 1.0.
            - predict_proba (Dict[str, float]): Probabilités des 3 meilleures filières.
            - facteurs_cles (List[str]): Facteurs explicatifs (simplifiés pour l'instant).
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

    try:
        input_ml = _mapper_profil_vers_input_ml(profil)
        top3 = predire_orientation_top3(input_ml)

        if not top3:
            return {
                "filiere_recommandee": "INCONNUE",
                "score_adequation": 0.0,
                "predict_proba": {},
                "facteurs_cles": ["Le modèle n'a retourné aucune prédiction"],
                "status": "ERROR"
            }

        filiere_rec = top3[0]["filiere"]
        score = top3[0]["confiance"] / 100.0
        predict_proba = {item["filiere"]: round(item["confiance"] / 100.0, 4) for item in top3}

        facteurs = [
            f"Série renseignée : {input_ml['serie']}",
            f"Moyenne générale : {input_ml['moyenne_generale']}",
        ]
        if input_ml["matieres_fortes"]:
            facteurs.append(f"Matières fortes : {input_ml['matieres_fortes']}")

        return {
            "filiere_recommandee": filiere_rec,
            "score_adequation": round(float(score), 4),
            "predict_proba": predict_proba,
            "facteurs_cles": facteurs,
            "status": "SUCCESS"
        }

    except FileNotFoundError as e:
        return {
            "filiere_recommandee": "INCONNUE",
            "score_adequation": 0.0,
            "predict_proba": {},
            "facteurs_cles": [f"Modèle introuvable : {e}"],
            "status": "ERROR"
        }
    except Exception as e:
        return {
            "filiere_recommandee": "INCONNUE",
            "score_adequation": 0.0,
            "predict_proba": {},
            "facteurs_cles": [f"Erreur d'inférence ML : {e}"],
            "status": "ERROR"
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