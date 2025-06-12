import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_model_features(df):
    model_df = df.copy()
    
    # ===== 1. Core Temporal Features =====
    model_df['HOUR_OF_DAY'] = model_df['STOP_FRISK_TIME']
    model_df['IS_NIGHT'] = model_df['HOUR_OF_DAY'].between(20, 6).astype(int)
    model_df['IS_WEEKEND'] = model_df['STOP_FRISK_DATE'].dt.dayofweek >= 5
    
    # ===== 2. Demographic Enhancements =====
    model_df['IS_CHILD'] = (model_df['SUSPECT_REPORTED_AGE'] < 18).astype(int)
    model_df['AGE_GROUP'] = pd.cut(
        model_df['SUSPECT_REPORTED_AGE'],
        bins=[0, 12, 18, 30, 50, 100],
        labels=['child', 'teen', 'young_adult', 'adult', 'senior']
    )
    
    # ===== 4. Location Intelligence =====
    # Borough-level force rates
    borough_force_rates = df.groupby('STOP_LOCATION_BORO_NAME')['OFFICER_USED_FORCE'].mean()
    model_df['BOROUGH_FORCE_RATE'] = model_df['STOP_LOCATION_BORO_NAME'].map(borough_force_rates)
    
    # ===== 5. Stop Context Features =====
    model_df['MULTIPLE_PERSONS_STOPPED'] = df['OTHER_PERSON_STOPPED_FLAG'].astype(int)
    
    # ===== 6. Officer Profile Features =====
    model_df['OFFICER_EXPERIENCE'] = np.where(
        df['ISSUING_OFFICER_RANK'].isin(['PO', 'PROBATIONARY POLICE OFFICER']),
        'junior',
        'senior'
    )
    
    # ===== 8. Smart Encoding =====
    # Categorical features with targeted encoding
    categorical_features = {
        'high_cardinality': [
            'SUSPECTED_CRIME_DESCRIPTION',
            'STOP_LOCATION_BORO_NAME',
            'NEIGHBORHOOD'
        ],
        'low_cardinality': [
            # 'SUSPECTED_CRIME_DESCRIPTION',
            # 'STOP_LOCATION_BORO_NAME',
            'SUSPECT_RACE_DESCRIPTION',
            'SUSPECT_SEX',
            'STOP_WAS_INITIATED',
            'OFFICER_EXPERIENCE',
            'AGE_GROUP'
        ]
    }
    
    # Target encoding for high-cardinality features
    for col in categorical_features['high_cardinality']:
        encoder = model_df.groupby(col)['OFFICER_USED_FORCE'].mean()
        model_df[f'{col}_ENCODED'] = model_df[col].map(encoder)
    
    # One-hot encode low-cardinality features
    encoded_df = pd.get_dummies(
        model_df[categorical_features['low_cardinality']],
        drop_first=True
    )
    
    # ===== 9. Final Feature Set =====
    numeric_features = [
        'SUSPECT_REPORTED_AGE',
        'SUSPECT_HEIGHT',
        'HOUR_OF_DAY',
        # 'NUM_FORCE_TYPES',
        'BOROUGH_FORCE_RATE'
    ]
    
    X = pd.concat([
        model_df[numeric_features],
        encoded_df,
        model_df[[
            'OFFICER_IN_UNIFORM_FLAG',
            'IS_NIGHT',
            'IS_WEEKEND',
            'IS_CHILD',
            # 'MULTIPLE_PERSONS_STOPPED'
        ]],
        model_df[[col for col in model_df.columns if '_ENCODED' in col]]
    ], axis=1)
    
    y = model_df['OFFICER_USED_FORCE']
    # y = model_df['OUTCOME_OF_STOP']
    
    # There might be a couple of null values left, clear them out assuming the delta isn't too high
    null_mask = X.isnull().any(axis=1)  # Boolean mask of rows with ANY nulls
    
    if null_mask.any():
        print(f"Dropping {null_mask.sum()} rows with null values")
        X_clean = X[~null_mask].copy()  # Keep only non-null rows
        y_clean = y[~null_mask].copy()
    else:
        X_clean, y_clean = X, y
    
    assert not X_clean.isnull().any().any(), "X still contains null values"
    assert len(X_clean) == len(y_clean), f"Length mismatch: X={len(X_clean)}, y={len(y_clean)}"
    
    return X_clean, y_clean


# Extract and visualize feature importances
def visualize_features(X, model, title='Top 15 Features Predicting Use of Force'):
    feature_names = X.columns
    coefficients = model.coef_[0]
    importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    importance = importance.sort_values('Coefficient', key=abs, ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=importance.head(15))
    plt.title(title)
    plt.tight_layout()
    plt.show()
