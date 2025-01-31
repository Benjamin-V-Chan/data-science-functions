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


def principal_component_analysis(data, n_components=2):
    """
    Performs PCA on a dataset and returns transformed data.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca.explained_variance_ratio_

def monte_carlo_simulation(func, n_simulations=10000):
    """
    Runs a Monte Carlo simulation for a given function.
    """
    results = [func() for _ in range(n_simulations)]
    return {'mean': np.mean(results), 'variance': np.var(results), '95% CI': (np.percentile(results, 2.5), np.percentile(results, 97.5))}


def exponential_smoothing(data, alpha=0.2):
    """
    Applies exponential smoothing to a time series dataset.
    """
    smoothed = [data[0]]  # First value remains the same
    for t in range(1, len(data)):
        smoothed.append(alpha * data[t] + (1 - alpha) * smoothed[t-1])
    return np.array(smoothed)