from collections import deque

def detect_anomalies_zscore(data_stream, window_size=50, threshold=3):
    """
    Detects anomalies in a data stream using Z-score-based analysis.
    
    Parameters:
    - data_stream: np.array
        The continuous data stream to be analyzed for anomalies.
    
    - window_size: int, optional (default=50)
        The size of the rolling window used for calculating the moving mean and standard deviation.
    
    - threshold: float, optional (default=3)
        The number of standard deviations above or below the mean to consider as an anomaly.

    Returns:
    - anomalies: list of tuples
        A list of detected anomalies where each entry is a tuple (index, value) indicating the index 
        of the anomaly in the data stream and the corresponding anomalous value.
    """
    # Deque to store a rolling window of data points
    rolling_window = deque(maxlen=window_size)
    
    # List to store detected anomalies
    anomalies = []

    for i, value in enumerate(data_stream):
        # Only calculate Z-score when the rolling window is full
        if len(rolling_window) == window_size:
            mean = np.mean(rolling_window)   # Compute mean of the rolling window
            std_dev = np.std(rolling_window) # Compute standard deviation of the rolling window

            # Calculate the Z-score for the current data point
            z_score = (value - mean) / std_dev if std_dev != 0 else 0

            # If the Z-score exceeds the threshold, mark it as an anomaly
            if abs(z_score) > threshold:
                anomalies.append((i, value))  # Store the index and value of the anomaly
        
        # Add the current value to the rolling window
        rolling_window.append(value)

    return anomalies

def zscore_anomaly_detection_optimized(data_stream, window_size=50, threshold=3):
    """
    Optimized Z-score-based anomaly detection using online calculations.
    """
    rolling_window = deque(maxlen=window_size)
    anomalies = []

    # Initialize rolling statistics
    sum_vals = 0.0
    sum_sq_vals = 0.0

    for i, value in enumerate(data_stream):
        if len(rolling_window) == window_size:
            # Compute mean and std_dev using rolling sums
            mean = sum_vals / window_size
            variance = (sum_sq_vals / window_size) - (mean ** 2)
            std_dev = np.sqrt(variance) if variance > 0 else 0

            z_score = (value - mean) / std_dev if std_dev != 0 else 0

            if abs(z_score) > threshold:
                anomalies.append((i, value))

            # Remove the oldest value from sums
            old_value = rolling_window.popleft()
            sum_vals -= old_value
            sum_sq_vals -= old_value ** 2
        else:
            # If window isn't full yet, just append
            rolling_window.append(value)
            sum_vals += value
            sum_sq_vals += value ** 2
            continue

        # Add the new value to rolling window and sums
        rolling_window.append(value)
        sum_vals += value
        sum_sq_vals += value ** 2

    return anomalies