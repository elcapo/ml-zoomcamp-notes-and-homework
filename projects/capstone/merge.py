"""
Data Merging Script for ECV 2024
Combines household and person data using hybrid aggregation strategy
"""

import pandas as pd
import numpy as np


def load_data(data_dir='data'):
    """Load household and person datasets"""

    households = pd.read_csv(
        f'{data_dir}/ECV_Th_2024/CSV/ECV_Th_2024.tab',
        sep='\t',
        encoding='latin-1',
        low_memory=False
    )

    persons = pd.read_csv(
        f'{data_dir}/ECV_Tp_2024/CSV/ECV_Tp_2024.tab',
        sep='\t',
        encoding='latin-1',
        low_memory=False
    )

    print(f"✓ Loaded {len(households)} households")
    print(f"✓ Loaded {len(persons)} persons")

    return households, persons


def preprocess_persons(persons):
    """Clean and prepare person data"""

    persons['household_id'] = (persons['PB030'] / 100).astype(int)
    persons['age'] = persons['PB110'] - persons['PB140']
    persons = persons.replace([-1, -2, -3, -4, -5, -6], np.nan)

    return persons


def merge_data(households, persons):
    """
    Merge household and person data using a hybrid aggregation strategy.

    In other words, it combines:
    - Household head characteristics
    - Aggregated household composition statistics
    - Derived features
    """

    persons_clean = preprocess_persons(persons.copy())

    # Household head
    household_heads = persons_clean.sort_values('PB030').groupby('household_id').first()
    head_features = household_heads[[
        'age', 'PB150', 'PB190', 'PE021', 'PL051A', 'PL060',
        'PY010N', 'PH010'
    ]]
    head_features.columns = ['head_' + col for col in head_features.columns]

    # Household composition
    composition = persons_clean.groupby('household_id').agg({
        'PB030': 'count',                       # Household size
        'age': ['mean', 'std'],                 # Age statistics
        'PB150': lambda x: (x == 1).sum(),      # Number of males
        'PE021': 'max',                         # Maximum education
        'PL051A': lambda x: (x == 1).sum(),     # Number employed
        'PY010N': 'sum',                        # Total employee income
        'PY090N': 'sum',                        # Total pension
        'PH010': 'mean',                        # Average health
    })

    composition.columns = [
        'household_size', 'mean_age', 'std_age', 'num_males',
        'max_education', 'num_employed', 'total_employee_income',
        'total_pension', 'avg_health'
    ]

    # Derived features
    composition['employment_rate'] = composition['num_employed'] / composition['household_size']

    # Age-based composition
    children = persons_clean[persons_clean['age'] < 18].groupby('household_id').size()
    elderly = persons_clean[persons_clean['age'] >= 65].groupby('household_id').size()

    composition['num_children'] = children.reindex(composition.index, fill_value=0)
    composition['num_elderly'] = elderly.reindex(composition.index, fill_value=0)
    composition['num_working_age'] = (
        composition['household_size'] -
        composition['num_children'] -
        composition['num_elderly']
    )
    composition['dependency_ratio'] = (
        (composition['num_children'] + composition['num_elderly']) /
        composition['num_working_age'].clip(lower=1)
    )

    # Income per capita
    composition['income_per_capita'] = (
        composition['total_employee_income'] / composition['household_size']
    )

    # Combile households and people
    person_features = pd.concat([head_features, composition], axis=1)

    df = households.merge(
        person_features,
        left_on='HB030',
        right_index=True,
        how='left'
    )

    # Select features
    household_features = ['HY020', 'HY022', 'HY023', 'HB070', 'HB100']
    available_features = [f for f in household_features if f in df.columns]

    X = df[available_features + person_features.columns.tolist()]
    y = df['vhPobreza']

    print(f"✓ Dataset shape: {X.shape}")
    print(f"✓ Features: {len(X.columns)}")
    print(f"✓ Samples: {len(X)}")
    print()

    print("Target distribution:")
    print(y.value_counts())
    print()

    print("Sample features:")
    head_cols = [c for c in X.columns if c.startswith('head_')][:5]
    comp_cols = [c for c in X.columns if not c.startswith('head_') and not c.startswith('HY') and not c.startswith('HB')][:5]
    print(f"* Head: {head_cols}")
    print(f"* Composition: {comp_cols}")

    return X, y, df


def main():
    """Main execution"""

    print("# Data Load\n")
    households, persons = load_data()
    print()

    print("# Data Merge\n")
    X, y, df = merge_data(households, persons)
    print()

    print("# Data Export\n")
    merged = pd.concat([X, y], axis=1)
    merged.to_csv('data/merged.csv', index=False)
    print(f"✓ Saved to 'data/merged.csv' ({len(merged)} rows, {len(merged.columns)} columns)")
    print()


if __name__ == '__main__':
    main()
