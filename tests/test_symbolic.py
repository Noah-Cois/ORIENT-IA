import pytest
from src.symbolic.logic import ISPMOntologyEngine

@pytest.fixture
def engine():
    return ISPMOntologyEngine(ontology_path="data/ontology/ispm_ontology.ttl")

def test_isaia_details(engine):
    info = engine.get_filiere_info("ISAIA")
    assert info["filiere"] == "ISAIA"
    assert info["parcours"] == "Parcours_InfoTelecom"
    assert "Prerequis_MathsAvance" in info["prerequis"]

def test_list_filieres_info(engine):
    filieres = engine.list_filieres_by_parcours("Parcours_InfoTelecom")
    assert "ISAIA" in filieres
    assert "IGGLIA" in filieres
    assert len(filieres) >= 4
