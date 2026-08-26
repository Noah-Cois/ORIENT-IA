"""Tests unitaires — Logique symbolique ORIENT'IA.

Vérifie que les requêtes SPARQL renvoient les bons résultats
pour chaque filière de l'ISPM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.symbolic.logic import SymbolicEngine


def test_engine_loads():
    """L'ontologie se charge correctement."""
    engine = SymbolicEngine()
    assert len(engine.graph) > 0, "Le graphe RDF est vide"


def test_all_filieres():
    """Toutes les filières sont présentes."""
    engine = SymbolicEngine()
    filieres = engine.get_all_filieres()
    assert len(filieres) == 9, f"Attendu 9 filières, trouvé {len(filieres)}"
    # Les noms complets contiennent les identifiants
    for name in ["ISAIA", "IGGLIA", "IMAAA", "ISMP", "ISPEN", "IST",
                 "ISPM", "BTS Informatique", "BTS Gestion"]:
        assert any(name in f for f in filieres), f"Filière manquante : {name}"


def test_isaia_parcours():
    """ISAIA appartient au parcours InfoTelecom."""
    engine = SymbolicEngine()
    result = engine.get_parcours_filiere("ISAIA")
    assert result is not None
    assert "Telecommunication" in result["parcoursNom"]


def test_isaia_matieres():
    """ISAIA enseigne au moins 3 matières."""
    engine = SymbolicEngine()
    matieres = engine.get_matieres_filiere("ISAIA")
    assert len(matieres) >= 3, f"Attendu ≥3 matières, trouvé {len(matieres)}"
    assert "Algorithmique et Programmation" in matieres


def test_isaia_competences():
    """ISAIA développe des compétences ML et DataScience."""
    engine = SymbolicEngine()
    comps = engine.get_competences_filiere("ISAIA")
    assert "Machine Learning" in comps
    assert "Data Science" in comps


def test_isaia_prerequis():
    """ISAIA exige un Bac S et 10/20 minimum."""
    engine = SymbolicEngine()
    pres = engine.get_prerequis_filiere("ISAIA")
    noms = [p["nom"] for p in pres]
    assert any("Baccalaureat" in n for n in noms), "Prérequis Bac manquant"
    assert any("Mathematiques" in n for n in noms), "Prérequis Maths manquant"


def test_isaia_metiers():
    """ISAIA prépare à DataScientist et DevFullStack."""
    engine = SymbolicEngine()
    metiers = engine.get_metiers_filiere("ISAIA")
    assert "Data Scientist" in metiers
    assert "Developpeur Full Stack" in metiers


def test_igglia_parcours():
    """IGGLIA appartient au parcours Sciences."""
    engine = SymbolicEngine()
    result = engine.get_parcours_filiere("IGGLIA")
    assert result is not None
    assert "Science" in result["parcoursNom"]


def test_master_prerequis_bac3():
    """Le Master Santé exige un Bac+3 minimum."""
    engine = SymbolicEngine()
    pres = engine.get_prerequis_filiere("ISPM_SantePublique")
    noms = [p["nom"] for p in pres]
    assert any("Bac+3" in n for n in noms), "Prérequis Bac+3 manquant pour le Master"


def test_filieres_parcours():
    """Au moins 2 filières dans le parcours InfoTelecom."""
    engine = SymbolicEngine()
    filieres = engine.get_filieres_par_parcours("Parcours_InfoTelecom")
    assert len(filieres) >= 2, f"Attendu ≥2 filières, trouvé {len(filieres)}"


def test_filieres_par_competence():
    """Au moins 2 filières développent Machine Learning."""
    engine = SymbolicEngine()
    filieres = engine.get_filieres_par_competence("Comp_ML")
    assert len(filieres) >= 2, f"Attendu ≥2 filières, trouvé {len(filieres)}"


def test_filieres_par_metier():
    """Au moins 1 filière prépare au métier de DataScientist."""
    engine = SymbolicEngine()
    filieres = engine.get_filieres_par_metier("Met_DataScientist")
    assert len(filieres) >= 1


def test_verifier_prerequis_ok():
    """Profil satisfaisant les prérequis d'ISAIA."""
    engine = SymbolicEngine()
    profil = {"mention": "Sciences Experimentales", "moyenne": 14.0}
    result = engine.verifier_prerequis("ISAIA", profil)
    assert result["satisfait"] is True


def test_verifier_prerequis_fail_moyenne():
    """Profil avec moyenne insuffisante pour ISAIA."""
    engine = SymbolicEngine()
    profil = {"mention": "Sciences Experimentales", "moyenne": 8.0}
    result = engine.verifier_prerequis("ISAIA", profil)
    assert result["satisfait"] is False


def test_verifier_prerequis_fail_mention():
    """Profil avec mauvaise mention pour IGGLIA."""
    engine = SymbolicEngine()
    profil = {"mention": "Humanites Generales", "moyenne": 14.0}
    result = engine.verifier_prerequis("IGGLIA", profil)
    assert result["satisfait"] is False


def test_resume_filiere():
    """Le résumé d'ISAIA contient toutes les clés attendues."""
    engine = SymbolicEngine()
    resume = engine.get_resume_filiere("ISAIA")
    assert "nom" in resume
    assert "parcours" in resume
    assert "matieres" in resume
    assert "competences" in resume
    assert "prerequis" in resume
    assert "metiers" in resume


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  OK {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR {test.__name__}: {e}")
            failed += 1
    print(f"\nRésultat : {passed} passés, {failed} échoués sur {passed + failed}")
