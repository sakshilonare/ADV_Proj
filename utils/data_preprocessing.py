import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values,
    ensuring categorical data is preserved, and cleaning uninformative columns.
    """
    # Drop columns with >50% missing values
    df = df.dropna(thresh=len(df) * 0.5, axis=1)

    # Handle string cleaning for all columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove quotes and extra spaces
            df[col] = df[col].str.replace('"', '', regex=False).str.strip()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':  # Categorical columns
            if df[col].isnull().all():
                df[col] = "Unknown"  # Replace entirely missing columns with "Unknown"
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)  # Fill with mode
        else:  # Numeric columns (if any exist)
            if df[col].isnull().all():
                df[col] = 0  # Replace entirely missing numeric columns with 0
            else:
                df[col].fillna(df[col].mean(), inplace=True)  # Fill with mean

    # Ensure no columns are completely empty after processing
    df.dropna(how='all', axis=1, inplace=True)
    df.dropna(how='all', axis=0, inplace=True)

    return df

def categorize_columns(df):
    """
    Categorize columns into numeric, categorical, and binary for visualization.
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    # Categorical columns should include non-numeric columns with more than 2 unique values
    categorical_cols = [
        col for col in df.select_dtypes(exclude='number').columns 
        if df[col].nunique() > 2
    ]
    
    # Binary columns should include non-numeric columns with exactly 2 unique values
    binary_cols = [
        col for col in df.select_dtypes(exclude='number').columns 
        if df[col].nunique() == 2
    ]

    # Remove columns like IDs or timestamps that should not be considered numeric
    # You can add more specific rules to identify such columns
    non_relevant_columns = ['id', 'timestamp', 'date', 'email', 'version', 'code']  # Add more based on your data
    numeric_cols = [col for col in numeric_cols if col not in non_relevant_columns]

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

def categorize_time_series_columns(df):
    """
    Identify columns that are related to time series or represent years.
    These could include columns like 'year', 'date', or 'timestamp'.
    """
    time_series_cols = []
    
    for col in df.columns:
        # Check if column name contains time-related keywords or is of datetime type
        if 'year' in col.lower() or 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            time_series_cols.append(col)
    
    # Additionally, you can categorize specific columns manually if needed
    # For example, you could add 'year' to the list if it doesn't get captured
    if 'year' in df.columns:
        time_series_cols.append('year')
    
    return time_series_cols