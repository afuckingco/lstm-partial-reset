import numpy as np
import pandas as pd

def generate_data_sudden(seed, n_subjects=17, n_days=120, drift_day=80):
    np.random.seed(seed)
    records = []
    for subj in range(n_subjects):
        inertia = np.random.uniform(0.2, 0.9)
        stress = np.random.uniform(3, 7)
        for day in range(n_days):
            drift = 2.0 if day >= drift_day else 0.0
            change = (drift + np.random.normal(0,0.1) - stress) * (1 - inertia)
            stress += change
            stress = np.clip(stress, 1, 10)
            mood = 10 - stress + np.random.normal(0,0.3)
            if day >= drift_day:
                mood += 0.5
            mood = np.clip(mood, 1, 10)
            sleep = 7 - 0.1*(stress-5) + np.random.normal(0,0.3)
            sleep = np.clip(sleep, 4, 10)
            social = 6 - 0.2*(stress-5) + np.random.normal(0,0.4)
            social = np.clip(social, 1, 10)
            records.append([subj, day, stress, mood, sleep, social, inertia])
    return pd.DataFrame(records, columns=['subject','day','stress','mood','sleep','social','inertia'])
