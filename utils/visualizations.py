import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def bar_chart(df, column):
    """Creates a bar chart for a categorical column."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f"Bar Chart of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    return plt

def scatter_plot(df, x_col, y_col):
    """Creates a scatter plot between two numeric columns."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    return plt

def histogram(df, column):
    """Creates a histogram for a numeric column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    return plt

def boxplot(df, x_col, y_col):
    """Creates a boxplot to compare a numeric column across categories."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title(f"Boxplot of {y_col} by {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    return plt

def heatmap(df):
    """Creates a heatmap to visualize correlations in numeric columns."""
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    return plt

def pairplot(df, numeric_columns):
    """Creates pairwise scatter plots for numeric columns."""
    if len(numeric_columns) > 1:
        sns.pairplot(df[numeric_columns], diag_kind="kde")
        plt.suptitle("Pairwise Scatter Plots", y=1.02)
        return plt
    else:
        raise ValueError("Not enough numeric columns for a pairplot.")
