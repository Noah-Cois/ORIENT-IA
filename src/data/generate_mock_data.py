import os
import pandas as pd
import numpy as np

def generate_ispm_dataset(n_samples=1500):
    np.random.seed(42)
    
    # 1. Liste de toutes les séries du Bac à Madagascar
    bacs = ['Bac C', 'Bac D', 'Bac A1', 'Bac A2', 'Bac S', 'Bac OSE', 'Bac Technique', 'Bac L']
    probs_bac = [0.20, 0.20, 0.08, 0.12, 0.15, 0.10, 0.10, 0.05]
    
    data = []
    
    for _ in range(n_samples):
        bac = np.random.choice(bacs, p=probs_bac)
        
        # 2. Simulation des notes selon le Bac (Loi normale ajustée)
        if bac in ['Bac C', 'Bac S']:
            maths = np.clip(np.random.normal(14.5, 2.5), 8, 20)
            physique = np.clip(np.random.normal(14.0, 2.5), 8, 20)
            francais = np.clip(np.random.normal(11.0, 2.0), 6, 17)
            malagasy = np.clip(np.random.normal(11.5, 2.0), 6, 17)
        elif bac == 'Bac D':
            maths = np.clip(np.random.normal(12.5, 2.5), 7, 20)
            physique = np.clip(np.random.normal(12.5, 2.5), 7, 20)
            francais = np.clip(np.random.normal(11.5, 2.0), 6, 17)
            malagasy = np.clip(np.random.normal(12.0, 2.0), 7, 18)
        elif bac == 'Bac A2':
            maths = np.clip(np.random.normal(10.5, 2.2), 5, 16)
            physique = np.clip(np.random.normal(9.5, 2.0), 5, 15)
            francais = np.clip(np.random.normal(13.5, 2.0), 8, 19)
            malagasy = np.clip(np.random.normal(14.0, 2.0), 8, 19)
        elif bac == 'Bac A1':
            maths = np.clip(np.random.normal(8.5, 2.0), 4, 14)
            physique = np.clip(np.random.normal(8.0, 2.0), 4, 13)
            francais = np.clip(np.random.normal(14.0, 2.0), 9, 20)
            malagasy = np.clip(np.random.normal(14.5, 2.0), 9, 20)
        elif bac == 'Bac Technique':
            maths = np.clip(np.random.normal(11.5, 2.5), 6, 18)
            physique = np.clip(np.random.normal(14.0, 2.0), 8, 20)
            francais = np.clip(np.random.normal(10.0, 2.0), 5, 16)
            malagasy = np.clip(np.random.normal(11.0, 2.0), 6, 16)
        elif bac == 'Bac OSE':
            maths = np.clip(np.random.normal(10.5, 2.0), 5, 16)
            physique = np.clip(np.random.normal(9.0, 2.0), 4, 14)
            francais = np.clip(np.random.normal(12.0, 2.0), 7, 18)
            malagasy = np.clip(np.random.normal(12.5, 2.0), 7, 18)
        else: # Bac L (Littéraire général)
            maths = np.clip(np.random.normal(8.0, 2.0), 4, 13)
            physique = np.clip(np.random.normal(7.5, 2.0), 4, 13)
            francais = np.clip(np.random.normal(13.5, 2.0), 8, 19)
            malagasy = np.clip(np.random.normal(14.0, 2.0), 8, 19)
            
        # 3. Compétences et intérêts (Auto-évaluation de 1 à 5)
        prog = np.random.randint(1, 6)
        elec = np.random.randint(1, 6)
        design = np.random.randint(1, 6)
        gestion = np.random.randint(1, 6)
        
        # 4. Calcul du score scientifique pondéré selon les coefficients du Bac
        if bac in ['Bac C', 'Bac S']:
            score_sci = ((maths * 5) + (physique * 5)) / 10
        elif bac == 'Bac D':
            score_sci = ((maths * 4) + (physique * 4)) / 8
        elif bac == 'Bac A2':
            score_sci = ((maths * 3) + (physique * 2)) / 5
        elif bac == 'Bac Technique':
            score_sci = ((maths * 3) + (physique * 5)) / 8
        else:  # A1, OSE, L
            score_sci = ((maths * 1) + (physique * 1)) / 2

        # 5. Attribution logique de la mention ISPM optimale
        if elec >= 4 and physique >= 12 and bac in ['Bac C', 'Bac S', 'Bac Technique', 'Bac D']:
            parcours = 'ESIIA'
        elif score_sci >= 14 and prog >= 3:
            parcours = 'ISAIA' if maths >= 15 else 'IGGLIA'
        elif design >= 4 or bac in ['Bac A1', 'Bac A2', 'Bac L']:
            parcours = 'IMTICIA'
        elif bac == 'Bac Technique' and physique >= 13:
            parcours = 'EMII'
        elif bac in ['Bac OSE', 'Bac A2'] or gestion >= 4:
            parcours = 'CAA_EMP'
        else:
            parcours = 'IGGLIA'
            
        # 6. Bruit aléatoire (8% de choix atypiques)
        if np.random.rand() < 0.08:
            parcours = np.random.choice(['IGGLIA', 'ESIIA', 'IMTICIA', 'ISAIA', 'EMII', 'CAA_EMP'])
            
        data.append([
            bac, round(float(maths), 1), round(float(physique), 1),
            round(float(francais), 1), round(float(malagasy), 1),
            round(float(score_sci), 1), prog, elec, design, gestion, parcours
        ])
        
    columns = [
        'serie_bac', 'note_maths', 'note_physique', 'note_francais', 
        'note_malagasy', 'score_sci_pondere', 'niveau_prog', 
        'interet_elec', 'appetence_design', 'interet_gestion', 'parcours_cible'
    ]
    
    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    df = generate_ispm_dataset(1500)
    df.to_csv(os.path.join(output_dir, 'mock_profiles.csv'), index=False)
    print("Dataset ML mis à jour avec toutes les séries et coefficients dans data/processed/mock_profiles.csv (1500 lignes).")