import os
from rdflib import Graph, Namespace

class ISPMOntologyEngine:
    """Moteur de raisonnement symbolique basé sur l'ontologie ISPM."""

    def __init__(self, ontology_path="data/ontology/ispm_ontology.ttl"):
        self.ontology_path = ontology_path
        self.g = Graph()
        self.ns = Namespace("http://www.ispm.mg/ontologies/orientia#")
        self._load_ontology()

    def _load_ontology(self):
        if os.path.exists(self.ontology_path):
            self.g.parse(self.ontology_path, format="ttl")
        else:
            raise FileNotFoundError(f"L'ontologie est introuvable à l'emplacement : {self.ontology_path}")

    def get_filiere_info(self, code_filiere: str) -> dict:
        """Récupère le parcours, les prérequis et les métiers associés à une filière."""
        code_filiere_upper = code_filiere.upper()
        query = f"""
        PREFIX : <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?parcours ?prerequis ?metier WHERE {{
            :{code_filiere_upper} :appartientAParcours ?parcours .
            OPTIONAL {{ :{code_filiere_upper} :exigePrerequis ?prerequis . }}
            OPTIONAL {{ :{code_filiere_upper} :prepareAMetier ?metier . }}
        }}
        """
        results = self.g.query(query)

        parcours = None
        prerequis_set = set()
        metiers_set = set()

        for row in results:
            if row.parcours:
                parcours = str(row.parcours).split("#")[-1]
            if row.prerequis:
                prerequis_set.add(str(row.prerequis).split("#")[-1])
            if row.metier:
                metiers_set.add(str(row.metier).split("#")[-1])

        return {
            "filiere": code_filiere_upper,
            "parcours": parcours,
            "prerequis": list(prerequis_set),
            "metiers_cibles": list(metiers_set)
        }

    def list_filieres_by_parcours(self, nom_parcours: str) -> list:
        """Retourne toutes les filières rattachées à un parcours (ex: 'Parcours_InfoTelecom')."""
        query = f"""
        PREFIX : <http://www.ispm.mg/ontologies/orientia#>
        SELECT ?filiere WHERE {{
            ?filiere :appartientAParcours :{nom_parcours} .
        }}
        """
        results = self.g.query(query)
        return [str(row.filiere).split("#")[-1] for row in results]

if __name__ == "__main__":
    # Test d'exécution rapide en local
    engine = ISPMOntologyEngine()
    print("--- Test ISAIA ---")
    print(engine.get_filiere_info("ISAIA"))
    print("\n--- Filières InfoTelecom ---")
    print(engine.list_filieres_by_parcours("Parcours_InfoTelecom"))
