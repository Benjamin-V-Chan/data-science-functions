import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def descriptive_statistics(data):
    """
    Returns basic descriptive statistics of a numerical dataset.
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'variance': np.var(data, ddof=1),
        'standard deviation': np.std(data, ddof=1),
        'min': np.min(data),
        'max': np.max(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }

def correlation_matrix(df):
    """
    Returns the correlation matrix of a DataFrame.
    """
    return df.corr()

def normality_test(data):
    """
    Performs the Shapiro-Wilk test for normality.
    """
    stat, p = stats.shapiro(data)
    return {'Shapiro-Wilk Statistic': stat, 'p-value': p, 'normal': p > 0.05}

