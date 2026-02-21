"""
COMPAS recidivism dataset loader.

Following ProPublica's preprocessing: filter to valid screening arrest dates
and valid recidivism flags. Protected attribute: race (k=6).
Binary label: two_year_recid.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/"
    "compas-analysis/master/compas-scores-two-years.csv"
)


def load_compas(data_path=None):
    """Load and preprocess the COMPAS recidivism dataset.

    Parameters
    ----------
    data_path : str or None
        Path to the CSV file. If None, attempts to download.

    Returns
    -------
    X : ndarray, shape (n, d)
        Standardized feature matrix (d=7).
    Y : ndarray, shape (n,)
        Binary labels {0, 1} (two_year_recid).
    A : ndarray, shape (n,)
        Group membership {0, ..., k-1}.
    group_names : list of str
        Names corresponding to group indices.
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        cache_dir = './data/compas'
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'compas-scores-two-years.csv')
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
        else:
            df = pd.read_csv(COMPAS_URL)
            df.to_csv(cache_path, index=False)

    # ProPublica filtering
    df = df[
        (df['days_b_screening_arrest'] >= -30) &
        (df['days_b_screening_arrest'] <= 30) &
        (df['is_recid'] != -1)
    ].copy()

    # Drop rows with missing values in key columns
    feature_cols = [
        'age', 'priors_count', 'c_jail_in', 'c_jail_out',
        'c_charge_degree', 'sex', 'age_cat', 'score_text'
    ]
    df = df.dropna(subset=feature_cols + ['two_year_recid', 'race'])

    # Protected attribute: race
    group_names = [
        'Caucasian', 'African-American', 'Hispanic',
        'Asian', 'Native American', 'Other'
    ]
    race_encoder = {name: i for i, name in enumerate(group_names)}
    df = df[df['race'].isin(group_names)]
    A = df['race'].map(race_encoder).values

    # Labels
    Y = df['two_year_recid'].values.astype(int)

    # Features (d=7)
    # 1. age (continuous)
    age = df['age'].values.astype(float)

    # 2. priors_count (integer)
    priors = df['priors_count'].values.astype(float)

    # 3. days_in_jail (continuous)
    jail_in = pd.to_datetime(df['c_jail_in'], errors='coerce')
    jail_out = pd.to_datetime(df['c_jail_out'], errors='coerce')
    days_in_jail = (jail_out - jail_in).dt.total_seconds() / 86400.0
    days_in_jail = days_in_jail.fillna(0).values

    # 4. c_charge_degree (binary: F=1, M=0)
    charge_degree = (df['c_charge_degree'] == 'F').astype(float).values

    # 5. sex (binary: Male=1, Female=0)
    sex = (df['sex'] == 'Male').astype(float).values

    # 6. age_cat (ordinal: <25=0, 25-45=1, >45=2)
    age_cat_map = {'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2}
    age_cat = df['age_cat'].map(age_cat_map).fillna(1).values.astype(float)

    # 7. score_text (ordinal: Low=0, Medium=1, High=2)
    score_map = {'Low': 0, 'Medium': 1, 'High': 2}
    score = df['score_text'].map(score_map).fillna(0).values.astype(float)

    X_raw = np.column_stack([
        age, priors, days_in_jail, charge_degree, sex, age_cat, score
    ])

    # Standardize continuous features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    return X, Y, A, group_names
