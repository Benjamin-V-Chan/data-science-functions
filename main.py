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