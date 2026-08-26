@echo off
echo Creation de l'architecture du projet orient_ia_project...

:: Creation des dossiers
mkdir "orient_ia_project\data\raw" 2>nul
mkdir "orient_ia_project\data\processed" 2>nul
mkdir "orient_ia_project\data\corpus" 2>nul
mkdir "orient_ia_project\data\ontology" 2>nul
mkdir "orient_ia_project\notebooks" 2>nul
mkdir "orient_ia_project\src\frontend" 2>nul
mkdir "orient_ia_project\src\agent" 2>nul
mkdir "orient_ia_project\src\ml" 2>nul
mkdir "orient_ia_project\src\rag" 2>nul
mkdir "orient_ia_project\src\symbolic" 2>nul
mkdir "orient_ia_project\src\db" 2>nul
mkdir "orient_ia_project\tests" 2>nul
mkdir "orient_ia_project\traces" 2>nul

:: Creation des fichiers
type nul > "orient_ia_project\data\ontology\ispm_ontology.owl"
type nul > "orient_ia_project\notebooks\ml_exploration.ipynb"
type nul > "orient_ia_project\src\frontend\app.py"
type nul > "orient_ia_project\src\agent\chatbot.py"
type nul > "orient_ia_project\src\agent\tools.py"
type nul > "orient_ia_project\src\ml\train.py"
type nul > "orient_ia_project\src\ml\predict.py"
type nul > "orient_ia_project\src\rag\ingest.py"
type nul > "orient_ia_project\src\rag\search.py"
type nul > "orient_ia_project\src\symbolic\logic.py"
type nul > "orient_ia_project\src\db\schema.sql"
type nul > "orient_ia_project\src\db\database.py"
type nul > "orient_ia_project\tests\test_evaluation.py"
type nul > "orient_ia_project\traces\execution_logs.json"
type nul > "orient_ia_project\requirements.txt"
type nul > "orient_ia_project\README.md"

echo Architecture creee avec succes !
pause