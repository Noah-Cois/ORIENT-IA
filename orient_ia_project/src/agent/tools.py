import os
import sys

# On s'assure que Python trouve le dossier 'ml' pour importer predict.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))
from predict import recommander_parcours

# Exemple d'intégration si vous utilisez LangChain plus tard (très classique pour les agents)
# Pour l'instant, c'est un outil standard que tu peux adapter selon votre framework
def outil_analyse_profil_ml(serie_bac: str, note_maths: float, note_physique: float, niveau_prog: int, appetence_comm: int) -> str:
    """
    Outil permettant à l'Agent d'interroger le modèle Machine Learning 
    pour obtenir une recommandation de parcours ISPM basée sur le profil du candidat.
    """
    try:
        # Appel direct à la fonction de prédiction que tu as créée
        recommandation = recommander_parcours(
            serie_bac=serie_bac,
            note_maths=note_maths,
            note_physique=note_physique,
            niveau_prog=niveau_prog,
            appetence_comm=appetence_comm
        )
        
        # Le LLM s'attend généralement à une réponse textuelle claire
        return f"Le modèle Machine Learning recommande le parcours : {recommandation}."
        
    except Exception as e:
        return f"Erreur lors de l'appel au modèle ML : {str(e)}"

# Test rapide pour vérifier que la liaison fonctionne
if __name__ == "__main__":
    print("Test de l'outil pour l'Agent :")
    resultat = outil_analyse_profil_ml(serie_bac="S", note_maths=14.5, note_physique=13.0, niveau_prog=4, appetence_comm=3)
    print(resultat)