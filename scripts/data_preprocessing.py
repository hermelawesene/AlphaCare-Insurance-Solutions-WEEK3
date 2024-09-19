import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

def impute_missing_values(df, numerical_features, categorical_features):
    # Impute numerical features with median
    numerical_imputer = SimpleImputer(strategy='median')
    df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])
    
    # Impute categorical features with the most frequent category
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    
    return df

def remove_missing_rows(df):
    df_clean = df.dropna()
    return df_clean

#Feature Engineering
# Create Interaction Terms
def create_interaction_terms(df, features, new_feature_name):
    df[new_feature_name] = df[features[0]] * df[features[1]]
    return df


def add_polynomial_features(df, feature, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[[feature]])
    poly_df = pd.DataFrame(poly_features, columns=[f'{feature}_poly_{i+1}' for i in range(poly_features.shape[1])])
    df = pd.concat([df, poly_df], axis=1)
    return df

def bin_continuous_feature(df, feature, bins, labels):
    df[feature+'_binned'] = pd.cut(df[feature], bins=bins, labels=labels)
    return df

def one_hot_encode(df, categorical_features):
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_encoded

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def scaler(method, data, columns_scaler):    
    if method == 'standardScaler':        
        Standard = StandardScaler()
        df_standard = data.copy()  # Create a copy of the dataset
        df_standard[columns_scaler] = Standard.fit_transform(df_standard[columns_scaler])        
        return df_standard
        
    elif method == 'minMaxScaler':        
        MinMax = MinMaxScaler()
        df_minmax = data.copy()  # Create a copy of the dataset
        df_minmax[columns_scaler] = MinMax.fit_transform(df_minmax[columns_scaler])        
        return df_minmax
    
    elif method == 'npLog':        
        df_nplog = data.copy()  # Create a copy of the dataset
        df_nplog[columns_scaler] = np.log(df_nplog[columns_scaler])        
        return df_nplog
    
    return data
