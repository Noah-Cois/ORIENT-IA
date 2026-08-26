"""Moteur de requêtes et logique symbolique — ORIENT'IA.

Charge l'ontologie ISPM, exécute des requêtes SPARQL,
et fournit des réponses déterministes à l'agent conversationnel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rdflib import Graph, Namespace, RDF, RDFS

ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ontology" / "ispm_ontology.ttl"
ISPM = Namespace("http://www.ispm.mg/ontologies/orientia#")


class SymbolicEngine:
    """Moteur logique basé sur l'ontologie OWL de l'ISPM."""

    def __init__(self, ontology_path: Path | str | None = None):
        self.path = Path(ontology_path) if ontology_path else ONTOLOGY_PATH
        self.graph = Graph()
        self._load()

    def _load(self) -> None:
        """Charge l'ontologie Turtle dans le graphe RDF."""
        if not self.path.exists():
            raise FileNotFoundError(f"Ontologie non trouvée : {self.path}")
        self.graph.parse(str(self.path), format="turtle")

    def _query(self, sparql: str) -> list[dict]:
        """Exécute une requête SPARQL et retourne les résultats."""
        results = []
        for row in self.graph.query(sparql):
            results.append({str(var): str(row[var]) for var in self.graph.query(sparql).vars})
        return results

    # ----------------------------------------------------------------
    # Requêtes prédéfinies pour l'agent
    # ----------------------------------------------------------------

    def get_parcours_filiere(self, filiere: str) -> dict | None:
        """Retourne le parcours d'une filière donnée."""
        sparql = f"""
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?parcoursNom WHERE {{
            ispm:{filiere} ispm:appartientAParcours ?p .
            ?p ispm:nom ?parcoursNom .
        }}
        """
        results = self._query(sparql)
        return results[0] if results else None

    def get_matieres_filiere(self, filiere: str) -> list[str]:
        """Retourne les matières enseignées dans une filière."""
        sparql = f"""
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?nom WHERE {{
            ispm:{filiere} ispm:enseigneMatiere ?m .
            ?m ispm:nom ?nom .
        }}
        """
        return [r["nom"] for r in self._query(sparql)]

    def get_competences_filiere(self, filiere: str) -> list[str]:
        """Retourne les compétences développées par une filière."""
        sparql = f"""
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?nom WHERE {{
            ispm:{filiere} ispm:developpeCompetence ?c .
            ?c ispm:nom ?nom .
        }}
        """
        return [r["nom"] for r in self._query(sparql)]

    def get_prerequis_filiere(self, filiere: str) -> list[dict]:
        """Retourne les prérequis d'une filière avec moyenne minimale."""
        sparql = f"""
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?nom ?moyenne WHERE {{
            ispm:{filiere} ispm:exigePrerequis ?p .
            ?p ispm:nom ?nom .
            OPTIONAL {{ ?p ispm:moyenneMinimale ?moyenne }}
        }}
        """
        return self._query(sparql)

    def get_metiers_filiere(self, filiere: str) -> list[str]:
        """Retourne les métiers préparés par une filière."""
        sparql = f"""
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?nom WHERE {{
            ispm:{filiere} ispm:prepareAMetier ?met .
            ?met ispm:nom ?nom .
        }}
        """
        return [r["nom"] for r in self._query(sparql)]

    def get_filieres_par_parcours(self, parcours: str) -> list[str]:
        """Retourne les filières d'un parcours donné."""
        sparql = f"""
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?nom WHERE {{
            ?f ispm:appartientAParcours ispm:{parcours} .
            ?f ispm:nom ?nom .
        }}
        """
        return [r["nom"] for r in self._query(sparql)]

    def get_filieres_par_competence(self, competence: str) -> list[str]:
        """Retourne les filières qui développent une compétence donnée."""
        sparql = f"""
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?nom WHERE {{
            ?f ispm:developpeCompetence ispm:{competence} .
            ?f ispm:nom ?nom .
        }}
        """
        return [r["nom"] for r in self._query(sparql)]

    def get_filieres_par_metier(self, metier: str) -> list[str]:
        """Retourne les filières qui préparent à un métier donné."""
        sparql = f"""
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?nom WHERE {{
            ?f ispm:prepareAMetier ispm:{metier} .
            ?f ispm:nom ?nom .
        }}
        """
        return [r["nom"] for r in self._query(sparql)]

    def verifier_prerequis(self, filiere: str, profil: dict) -> dict:
        """Vérifie si un profil satisfait les prérequis d'une filière.

        Args:
            filiere: Identifiant de la filière (ex: 'ISAIA').
            profil: Dict avec au minimum 'mention', 'moyenne'.

        Returns:
            dict avec 'satisfait' (bool), 'details' (liste).
        """
        prerequis = self.get_prerequis_filiere(filiere)
        details = []
        satisfait = True

        for pre in prerequis:
            nom = pre.get("nom", "")
            moyenne = pre.get("moyenne")

            if "Baccalaureat" in nom:
                mention = profil.get("mention", "")
                if "S" in nom and "S" not in mention and "Science" not in mention:
                    satisfait = False
                    details.append(f"❌ {nom} — mention actuelle : {mention}")
                else:
                    details.append(f"✅ {nom}")

            elif moyenne:
                moyenne_profil = profil.get("moyenne", 0)
                if float(moyenne_profil) < float(moyenne):
                    satisfait = False
                    details.append(f"❌ {nom} — moyenne {moyenne_profil} < {moyenne}")
                else:
                    details.append(f"✅ {nom} — moyenne {moyenne_profil} ≥ {moyenne}")
            else:
                details.append(f"ℹ️ {nom}")

        return {"satisfait": satisfait, "details": details}

    def get_all_filieres(self) -> list[str]:
        """Retourne toutes les filières disponibles."""
        sparql = """
        PREFIX ispm: <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?nom WHERE {
            ?f a ispm:Filiere .
            ?f ispm:nom ?nom .
        }
        """
        return [r["nom"] for r in self._query(sparql)]

    def get_resume_filiere(self, filiere: str) -> dict:
        """Retourne un résumé complet d'une filière."""
        return {
            "nom": filiere,
            "parcours": self.get_parcours_filiere(filiere),
            "matieres": self.get_matieres_filiere(filiere),
            "competences": self.get_competences_filiere(filiere),
            "prerequis": self.get_prerequis_filiere(filiere),
            "metiers": self.get_metiers_filiere(filiere),
        }


if __name__ == "__main__":
    engine = SymbolicEngine()
    print("Filières disponibles :", engine.get_all_filieres())
    print("\n--- ISAIA ---")
    print(engine.get_resume_filiere("ISAIA"))
    print("\n--- Vérification prérequis ---")
    profil = {"mention": "Sciences Experimentales", "moyenne": 14.0}
    print(engine.verifier_prerequis("ISAIA", profil))
