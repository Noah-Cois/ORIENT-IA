from rdflib import Graph, Namespace, RDF, RDFS, OWL

g = Graph()
g.parse("D:/Mini projet/Projet Clinique/ORIENT-IA/data/ontology/ispm_ontology.ttl", format="turtle")
print(f"Triples charges : {len(g)}")

ISPM = Namespace("http://www.ispm.mg/ontologies/orientia#")

# Filières
fcs = list(g.subjects(predicate=RDF.type, object=ISPM.Filiere))
print(f"\nFileres : {len(fcs)}")
for f in fcs:
    nom = list(g.objects(f, ISPM.nom))
    name = str(nom[0]) if nom else f.split("#")[-1]
    mat = len(list(g.objects(f, ISPM.enseigneMatiere)))
    comp = len(list(g.objects(f, ISPM.developpeCompetence)))
    pre = len(list(g.objects(f, ISPM.exigePrerequis)))
    met = len(list(g.objects(f, ISPM.prepareAMetier)))
    print(f"  {name}: {mat} mat, {comp} comp, {pre} pre, {met} met")

# Object Properties
ops = list(g.subjects(predicate=RDF.type, object=OWL.ObjectProperty))
print(f"\nObject Properties : {len(ops)}")
for op in ops:
    print(f"  - {op.split('#')[-1]}")

# Datatype Properties
dps = list(g.subjects(predicate=RDF.type, object=OWL.DatatypeProperty))
print(f"\nDatatype Properties : {len(dps)}")

# Individus par classe
for cls_name in ["Parcours", "Matiere", "Competence", "Prerequis", "Metier"]:
    cls = ISPM[cls_name]
    inds = list(g.subjects(predicate=RDF.type, object=cls))
    print(f"\n{cls_name} : {len(inds)} individus")
    for ind in inds:
        nom = list(g.objects(ind, ISPM.nom))
        name = str(nom[0]) if nom else ind.split("#")[-1]
        print(f"  - {name}")

print("\nOntologie valide !")
