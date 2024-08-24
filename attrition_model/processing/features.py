import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



def numerical_transformer(numerical_features):
    # Impute missing values with median and scale the numerical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handling missing values
        ('scaler', StandardScaler())  # Scaling features
    ])
    return numerical_pipeline

def categorical_transformer(categorical_features):
    # Impute missing values with the most frequent value and one-hot encode the categorical features
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handling missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
    ])
    return categorical_pipeline

def create_preprocessor(numerical_features, categorical_features):
    # Combine numerical and categorical transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer(numerical_features), numerical_features),
            ('cat', categorical_transformer(categorical_features), categorical_features)
        ]
    )
    return preprocessor

