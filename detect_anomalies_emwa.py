import numpy as np

def ewma(data_stream, alpha=0.3):
    """
    Compute Exponentially Weighted Moving Average (EWMA).
    
    Parameters:
    - data: list or array of values to smooth.
    - alpha: smoothing factor, determines the weight of recent observations (0 < alpha <= 1).
    
    Returns:
    - smoothed_data: array of smoothed values.
    """
    smoothed_data = np.zeros_like(data_stream)
    smoothed_data[0] = data_stream[0]  # Initialize the first value
    
    for t in range(1, len(data_stream)):
        smoothed_data[t] = alpha * data_stream[t] + (1 - alpha) * smoothed_data[t - 1]
    
    return smoothed_data

def detect_anomalies_ewma(data, alpha=0.3, threshold=3):
    """
    EWMA-based anomaly detection.
    
    Parameters:
    - data: list or array of values.
    - alpha: smoothing factor for EWMA.
    - threshold: the number of standard deviations away from EWMA to flag as an anomaly.
    
    Returns:
    - anomalies: list of (index, value) where anomalies are detected.
    """
    smoothed_data = ewma(data, alpha)
    residuals = np.abs(data - smoothed_data)
    std_dev = np.std(residuals)  # Standard deviation of the residuals
    
    anomalies = []
    for t in range(len(data)):
        if residuals[t] > threshold * std_dev:
            anomalies.append((t, data[t]))
    
    return anomalies