"""
UCI Adult (Census Income) dataset loader.

Protected attribute: race (k=5).
Binary label: income > 50K.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


ADULT_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
ADULT_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)

COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]


def load_adult(data_dir=None):
    """Load and preprocess the UCI Adult dataset.

    Parameters
    ----------
    data_dir : str or None
        Directory containing adult.data and adult.test.

    Returns
    -------
    X : ndarray, shape (n, d)
        Feature matrix (d=14 raw, higher after one-hot encoding).
    Y : ndarray, shape (n,)
        Binary labels {0, 1}.
    A : ndarray, shape (n,)
        Group membership {0, ..., k-1}.
    group_names : list of str
        Names corresponding to group indices.
    """
    cache_dir = data_dir or './data/adult'
    os.makedirs(cache_dir, exist_ok=True)

    # Load or download data
    train_path = os.path.join(cache_dir, 'adult.data')
    test_path = os.path.join(cache_dir, 'adult.test')

    if not os.path.exists(train_path):
        df_train = pd.read_csv(ADULT_TRAIN_URL, names=COLUMN_NAMES,
                               sep=r',\s*', engine='python', na_values='?')
        df_train.to_csv(train_path, index=False)
    else:
        df_train = pd.read_csv(train_path)

    if not os.path.exists(test_path):
        df_test = pd.read_csv(ADULT_TEST_URL, names=COLUMN_NAMES,
                              sep=r',\s*', engine='python', na_values='?',
                              skiprows=1)
        df_test.to_csv(test_path, index=False)
    else:
        df_test = pd.read_csv(test_path)

    # Combine train + test
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop missing values
    df = df.dropna()

    # Protected attribute: race (k=5)
    group_names = [
        'White', 'Black', 'Asian-Pac-Islander',
        'Amer-Indian-Eskimo', 'Other'
    ]
    race_encoder = {name: i for i, name in enumerate(group_names)}
    df = df[df['race'].isin(group_names)]
    A = df['race'].map(race_encoder).values

    # Binary label: income > 50K
    Y = df['income'].str.contains('>50K').astype(int).values

    # Features
    continuous_cols = [
        'age', 'education-num', 'capital-gain', 'capital-loss',
        'hours-per-week'
    ]
    categorical_cols = [
        'workclass', 'marital-status', 'occupation', 'relationship'
    ]
    # Native country: binary US/Other
    df['native-country-binary'] = (
        df['native-country'] == 'United-States'
    ).astype(float)

    # Continuous features
    X_cont = df[continuous_cols].values.astype(float)

    # Categorical features via one-hot encoding
    cat_data = df[categorical_cols].values
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(cat_data)

    # Native country (binary)
    X_country = df['native-country-binary'].values.reshape(-1, 1)

    X_raw = np.hstack([X_cont, X_cat, X_country])

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    return X, Y, A, group_names
