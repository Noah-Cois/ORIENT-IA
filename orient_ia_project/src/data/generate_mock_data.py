import pandas as pd
import numpy as np
import os

def generate_ispm_dataset(n_samples=1200):
    np.random.seed(42)
    
    bacs = ['Bac C', 'Bac D', 'Bac S', 'Bac Technique', 'Bac OSE', 'Bac L']
    probs_bac = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
    
    data = []
    
    for _ in range(n_samples):
        bac = np.random.choice(bacs, p=probs_bac)
        
        # Simulation des notes selon le Bac avec np.clip()
        if bac == 'Bac C':
            maths = np.clip(np.random.normal(14.5, 2.2), 8, 20)
            physique = np.clip(np.random.normal(14.0, 2.2), 8, 20)
        elif bac in ['Bac D', 'Bac S']:
            maths = np.clip(np.random.normal(12.5, 2.5), 7, 20)
            physique = np.clip(np.random.normal(12.5, 2.5), 7, 20)
        elif bac == 'Bac Technique':
            maths = np.clip(np.random.normal(11.5, 2.5), 6, 18)
            physique = np.clip(np.random.normal(14.0, 2.0), 8, 20)
        elif bac == 'Bac OSE':
            maths = np.clip(np.random.normal(10.5, 2.0), 5, 16)
            physique = np.clip(np.random.normal(9.0, 2.0), 4, 14)
        else: # Bac L
            maths = np.clip(np.random.normal(8.5, 2.0), 4, 14)
            physique = np.clip(np.random.normal(7.5, 2.0), 4, 13)
            
        prog = np.random.randint(1, 6)
        elec = np.random.randint(1, 6)
        design = np.random.randint(1, 6)
        gestion = np.random.randint(1, 6)
        
        # Attribution de la mention ISPM
        if elec >= 4 and physique >= 12:
            parcours = 'ESIIA'
        elif maths >= 13 and prog >= 3:
            parcours = 'ISAIA' if maths >= 15 else 'IGGLIA'
        elif design >= 4 or bac == 'Bac L':
            parcours = 'IMTICIA'
        elif bac == 'Bac Technique' and physique >= 13:
            parcours = 'EMII'
        elif bac == 'Bac OSE' or gestion >= 4:
            parcours = 'CAA_EMP'
        else:
            parcours = 'IGGLIA'
            
        # 8% de bruit aléatoire
        if np.random.rand() < 0.08:
            parcours = np.random.choice(['IGGLIA', 'ESIIA', 'IMTICIA', 'ISAIA', 'EMII', 'CAA_EMP'])
            
        data.append([
            bac, round(float(maths), 1), round(float(physique), 1), 
            prog, elec, design, gestion, parcours
        ])
        
    columns = [
        'serie_bac', 'note_maths', 'note_physique', 
        'niveau_prog', 'interet_elec', 'appetence_design', 
        'interet_gestion', 'parcours_cible'
    ]
    
    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    df = generate_ispm_dataset()
    df.to_csv('data/processed/mock_profiles.csv', index=False)
    print("Dataset mis à jour avec succès dans data/processed/mock_profiles.csv ! (1200 lignes)")