"""
Folktables (ACSIncome) dataset loader.

Uses ACSIncome task from the 2018 American Community Survey.
Protected attribute: RAC1P (race), k=9 groups.
Binary label: income > 50K.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_folktables(data_dir=None, year=2018, states=None):
    """Load and preprocess the Folktables ACSIncome dataset.

    Parameters
    ----------
    data_dir : str or None
        Directory to cache downloaded data.
    year : int
        Survey year (default 2018).
    states : list or None
        List of state codes; None for all states.

    Returns
    -------
    X : ndarray, shape (n, d)
        Standardized feature matrix.
    Y : ndarray, shape (n,)
        Binary labels {0, 1}.
    A : ndarray, shape (n,)
        Group membership {0, ..., k-1}.
    group_names : list of str
        Names corresponding to group indices.
    """
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError:
        raise ImportError(
            "Please install folktables: pip install folktables==0.0.12"
        )

    # Download data
    if states is None:
        states = ['CA']  # Default to California for manageable size
    data_source = ACSDataSource(
        survey_year=str(year), horizon='1-Year', survey='person',
        root_dir=data_dir or './data/folktables'
    )
    acs_data = data_source.get_data(states=states, download=True)

    # Extract features and labels using ACSIncome task
    X_df, Y_arr, _ = ACSIncome.df_to_numpy(acs_data)

    # Protected attribute: RAC1P (race)
    race_col = acs_data['RAC1P'].values
    group_names = [
        'White', 'Black', 'American Indian', 'Alaska Native',
        'Asian', 'Hawaiian/PI', 'Other', 'Two Races', 'Three+ Races'
    ]
    # RAC1P is 1-indexed in ACS
    A = (race_col - 1).astype(int)

    # Remove rows with missing values
    valid_mask = ~(np.isnan(X_df).any(axis=1) | np.isnan(Y_arr))
    X_df = X_df[valid_mask]
    Y_arr = Y_arr[valid_mask]
    A = A[valid_mask]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    Y = Y_arr.astype(int)

    return X, Y, A, group_names


def load_folktables_all_states(data_dir=None, year=2018):
    """Load Folktables for all US states (full 195K dataset)."""
    all_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'PR'
    ]
    return load_folktables(data_dir=data_dir, year=year, states=all_states)
