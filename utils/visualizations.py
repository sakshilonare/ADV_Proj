import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.graph_objects as go

def bar_chart(df, column):
    """Creates a bar chart for a categorical column."""
    plt.figure(figsize=(8, 4))  # Reduced size
    sns.countplot(x=column, data=df, palette="viridis")
    plt.title(f"Bar Chart of {column}", fontsize=12)
    plt.xlabel(column, fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.xticks(fontsize=8, rotation=45)
    plt.tight_layout()
    
    return plt

def scatter_plot(df, x_col, y_col):
    """Creates a scatter plot between two numeric columns."""
    plt.figure(figsize=(8, 4))  # Reduced size
    sns.scatterplot(x=x_col, y=y_col, data=df, color="blue", alpha=0.7)
    plt.title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=12)
    plt.xlabel(x_col, fontsize=10)
    plt.ylabel(y_col, fontsize=10)
    plt.tight_layout()
    return plt

def histogram(df, column):
    """Creates a histogram for a numeric column."""
    plt.figure(figsize=(8, 4))  # Reduced size
    sns.histplot(df[column], kde=True, color="green", bins=20)
    plt.title(f"Histogram of {column}", fontsize=12)
    plt.xlabel(column, fontsize=5)
    plt.ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    return plt

def boxplot(df, x_col, y_col):
    """Creates a boxplot to compare a numeric column across categories."""
    plt.figure(figsize=(6, 4))  # Reduced size
    sns.boxplot(x=x_col, y=y_col, data=df, palette="Set2")
    plt.title(f"Boxplot of {y_col} by {x_col}", fontsize=12)
    plt.xlabel(x_col, fontsize=10)
    plt.ylabel(y_col, fontsize=10)
    plt.xticks(fontsize=5, rotation=90)
    plt.tight_layout()
    return plt

def heatmap(df):
    """Creates a heatmap to visualize correlations in numeric columns."""
    plt.figure(figsize=(6, 4))  # Reduced size
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 6})
    plt.title("Correlation Heatmap", fontsize=12)
    plt.tight_layout()
    return plt

def pairplot(df, numeric_columns):
    """Creates pairwise scatter plots for numeric columns."""
    if len(numeric_columns) > 1:
        sns.pairplot(df[numeric_columns], diag_kind="kde", plot_kws={"s": 10})
        plt.subplots_adjust(top=0.9)
        plt.suptitle("Pairwise Scatter Plots", y=0.98, fontsize=12)
        return plt
    else:
        raise ValueError("Not enough numeric columns for a pairplot.")

def wordcloud_plot(text_series):
    """Creates a word cloud from text data."""
    text = ' '.join(text_series.dropna())
    wordcloud = WordCloud(background_color="white", width=400, height=300).generate(text)
    plt.figure(figsize=(6, 4))  # Reduced size
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud", fontsize=12)
    return plt

def time_series_plot(df, time_col, numeric_col):
    """
    Creates a time series plot for a given time column and numeric column.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=time_col, y=numeric_col, data=df, marker='o', linewidth=2)
    plt.title(f"Time Series Plot: {numeric_col} over {time_col}", fontsize=14)
    plt.xlabel(time_col.capitalize(), fontsize=12)
    plt.ylabel(numeric_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt


def sankey_diagram(source, target, value):
    """
    Creates a Sankey diagram to visualize flows.
    :param source: List of source nodes
    :param target: List of target nodes
    :param value: List of values for the flows
    """
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=10,  # Reduced padding
            thickness=15,  # Reduced thickness
            line=dict(color="black", width=0.5),
            label=list(set(source + target)),
            color="cornflowerblue"
        ),
        link=dict(
            source=[list(set(source + target)).index(s) for s in source],
            target=[list(set(source + target)).index(t) for t in target],
            value=value
        )
    )])
    fig.update_layout(
        title_text="Sankey Diagram", 
        font_size=10, 
        height=400, 
        width=600  # Reduced dimensions
    )
    return fig
def regression_plot(df, x_col, y_col):
    """
    Creates a regression plot to visualize the relationship between two numeric columns.
    :param df: DataFrame containing the data
    :param x_col: The independent variable (X-axis)
    :param y_col: The dependent variable (Y-axis)
    """
    plt.figure(figsize=(8, 4))  # Reduced size
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    plt.title(f"Regression Plot: {x_col} vs {y_col}", fontsize=12)
    plt.xlabel(x_col, fontsize=10)
    plt.ylabel(y_col, fontsize=10)
    plt.tight_layout()
    return plt
