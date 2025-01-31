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


def linear_regression(X, y):
    """
    Fits a simple linear regression model and returns coefficients.
    """
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    return {'slope': model.coef_[0], 'intercept': model.intercept_}

def kmeans_clustering(data, k=3):
    """
    Performs K-Means clustering on a dataset and returns cluster labels.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels


def bootstrap_sampling(data, n_samples=1000):
    """
    Generates bootstrap samples and returns confidence intervals.
    """
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_samples)]
    return {'mean': np.mean(means), '95% CI': (np.percentile(means, 2.5), np.percentile(means, 97.5))}

def t_test(data1, data2):
    """
    Performs an independent t-test to compare two datasets.
    """
    stat, p = stats.ttest_ind(data1, data2)
    return {'t-statistic': stat, 'p-value': p, 'significant': p < 0.05}
