# analysis.py - stats and viz functions

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def descriptive_stats(df):
    # get basic stats for numeric cols
    nums = df.select_dtypes(include=[np.number])
    s = nums.describe().T[['min', 'max', 'mean', 'std']]
    s['variance'] = nums.var()
    s['skewness'] = nums.skew()
    s['kurtosis'] = nums.kurtosis()
    return s

def correlation_matrix(df):
    return df.select_dtypes(include=[np.number]).corr()

def covariance_matrix(df):
    return df.select_dtypes(include=[np.number]).cov()

def anova_test(df, cat_col, num_col):
    # one way anova
    groups = df.groupby(cat_col)[num_col].apply(list)
    f, p = stats.f_oneway(*groups)
    return f, p

def ttest(arr1, arr2):
    return stats.ttest_ind(arr1, arr2)

def chi_square(df, c1, c2):
    table = pd.crosstab(df[c1], df[c2])
    chi, p, dof, exp = stats.chi2_contingency(table)
    return chi, p

def plot_heatmap(corr, path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_dist(df, col, path=None):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} Distribution')
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def plot_boxplot(df, x, y, path=None):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(f'{y} by {x}')
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
