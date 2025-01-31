# data-science-functions

## **Overview**  
This project contains a collection of Python functions for statistical analysis and data science. The functions perform various tasks such as descriptive statistics, hypothesis testing, regression analysis, clustering, Monte Carlo simulations, and more.  

## **Features**  
- **Descriptive Statistics**: Compute mean, median, variance, standard deviation, skewness, and kurtosis.  
- **Correlation Matrix**: Analyze relationships between variables in a dataset.  
- **Normality Test**: Perform the Shapiro-Wilk test for normality.  
- **Linear Regression**: Fit a simple linear regression model and return coefficients.  
- **K-Means Clustering**: Cluster data using the K-Means algorithm.  
- **Bootstrap Sampling**: Generate bootstrap samples for confidence intervals.  
- **T-Test**: Compare two datasets using an independent t-test.  
- **Principal Component Analysis (PCA)**: Reduce dimensionality of a dataset.  
- **Monte Carlo Simulation**: Run simulations to estimate probabilities.  
- **Exponential Smoothing**: Apply smoothing to time-series data.  

## **Installation**  
Ensure you have Python installed, then install the required libraries:  
```bash
pip install numpy pandas scipy scikit-learn
```

## **Usage**  
Import the functions and use them with numerical datasets:  
```python
from stats_functions import descriptive_statistics, normality_test

data = [10, 20, 30, 40, 50]
print(descriptive_statistics(data))
print(normality_test(data))
```