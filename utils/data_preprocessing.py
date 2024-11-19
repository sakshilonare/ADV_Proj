import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, 
    identifying data types, and cleaning uninformative columns.
    """
    # Drop columns with >50% missing values
    df = df.dropna(thresh=len(df) * 0.5, axis=1)

    # Fill remaining missing values
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].mean(), inplace=True)  # Fill numeric with mean
    for col in df.select_dtypes(exclude='number').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical with mode

    return df

def categorize_columns(df):
    """
    Categorize columns into numeric, categorical, and binary for visualization.
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = [
        col for col in df.select_dtypes(exclude='number').columns 
        if df[col].nunique() > 2
    ]
    binary_cols = [
        col for col in df.select_dtypes(exclude='number').columns 
        if df[col].nunique() == 2
    ]

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "binary": binary_cols
    }

def identify_key_columns(df):
    """
    Identify key columns for visualization based on variability and missing values.
    """
    key_columns = []
    for col in df.columns:
        # Exclude columns with low variability or too many missing values
        if df[col].nunique() > 1 and df[col].isnull().mean() < 0.5:
            key_columns.append(col)

    return key_columns
