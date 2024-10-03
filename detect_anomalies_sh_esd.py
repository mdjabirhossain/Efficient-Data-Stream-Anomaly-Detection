import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

def sh_esd(data_stream, period, max_anomalies=0.05):
    """
    Seasonal Hybrid Extreme Studentized Deviate (S-H-ESD) anomaly detection.
    
    Parameters:
    - data_stream: np.array
        The continuous data stream to be analyzed for anomalies.
    
    - period: int
        The seasonal period of the data stream (e.g., 24 for daily data with hourly observations).
    
    - max_anomalies: float, optional (default=0.05)
        The maximum percentage of data points that can be detected as anomalies.

    Returns:
    - anomalies: list of tuples
        A list of detected anomalies where each entry is a tuple (index, value) indicating the index 
        of the anomaly in the data stream and the corresponding anomalous value.
    """
    # Convert data to a pandas Series
    data = pd.Series(data_stream)
    
    # Decompose the data stream into seasonal, trend, and residual components
    decomposition = seasonal_decompose(data, period=period, model='additive', extrapolate_trend='freq')
    
    # Extract the residual component
    residual = decomposition.resid.dropna()
    
    # Perform the ESD test on the residuals
    anomalies = perform_esd_test(residual, max_anomalies)
    
    return anomalies

def perform_esd_test(residuals, max_anomalies):
    """
    Perform the ESD (Extreme Studentized Deviate) test on residuals to detect anomalies.
    
    Parameters:
    - residuals: pd.Series
        The residual component of the decomposed time series.
    
    - max_anomalies: float
        The maximum percentage of data points that can be detected as anomalies.

    Returns:
    - anomalies: list of tuples
        Detected anomalies as (index, value) tuples.
    """
    n = len(residuals)
    max_outliers = int(n * max_anomalies)
    
    anomalies = []
    for i in range(max_outliers):
        mean = residuals.mean()
        std_dev = residuals.std()
        
        # Calculate the modified Z-score for each residual
        z_scores = np.abs((residuals - mean) / std_dev)
        
        # Find the data point with the maximum Z-score
        max_z_score = z_scores.max()
        max_index = z_scores.idxmax()
        
        # Compute the test statistic for ESD
        lambda_value = stats.t.ppf(1 - 0.05 / (2 * (n - i)), df=n - i - 1)
        critical_value = lambda_value * (n - i) / np.sqrt((n - i - 1 + lambda_value**2) * (n - i))
        
        # Check if the maximum Z-score exceeds the critical value (flag as anomaly)
        if max_z_score > critical_value:
            anomalies.append((max_index, residuals[max_index]))
            residuals = residuals.drop(max_index)  # Remove the outlier for next iteration
        else:
            break
    
    return anomalies