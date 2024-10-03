import numpy as np

def generate_data_stream(steps=1000):
    """
    Simulates a data stream with seasonal variations, random noise, and injected anomalies.
    
    Parameters:
    - steps: int, optional (default=1000)
        The number of data points to generate in the stream.

    Returns:
    - data_stream: np.array
        The generated data stream containing sinusoidal patterns, noise, and random anomalies.
    """
    time = np.arange(steps)
    
    # Sinusoidal base signal with seasonal variation
    seasonal_signal = 10 * np.sin(2 * np.pi * time / 100)  
    
    # Add random noise to simulate real-world data fluctuations
    noise = np.random.normal(0, 1, steps) 
    
    # Inject random anomalies (sudden spikes)
    anomalies = np.random.choice([0, 50], size=steps, p=[0.98, 0.02])
    
    # Combine seasonal signal, noise, and anomalies
    data_stream = seasonal_signal + noise + anomalies
    
    return data_stream