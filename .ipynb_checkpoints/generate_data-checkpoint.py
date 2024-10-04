import numpy as np
from arch import arch_model
import statsmodels.api as sm

def generate_arima_data(order=(1, 1, 1), steps=1000):
    """
    Generates a synthetic time series using the ARIMA model.
    
    Parameters:
    - order: tuple
        The (p, d, q) order of the ARIMA model.
        p: number of autoregressive terms.
        d: number of differences.
        q: number of moving average terms.
    - steps: int
        Number of time steps to generate.
    
    Returns:
    - arima_data: np.array
        Generated ARIMA time series data.
    """
    # Generate random noise to simulate white noise
    np.random.seed(0)
    noise = np.random.normal(0, 1, steps)
    
    # Fit ARIMA model with given parameters
    arima_model = sm.tsa.ArmaProcess.from_coeffs(order[0], order[2])
    arima_data = arima_model.generate_sample(steps)
    
    return arima_data

def generate_brownian_motion(steps=1000, drift=0.001, volatility=0.05):
    """
    Generates a Brownian motion time series with drift and volatility.
    
    Parameters:
    - steps: int
        Number of time steps to generate.
    - drift: float
        The expected drift per step.
    - volatility: float
        The standard deviation of random noise (volatility).
    
    Returns:
    - brownian_motion: np.array
        Generated Brownian motion time series.
    """
    time = np.arange(steps)
    # Simulate Brownian motion with drift
    brownian_motion = np.cumsum(np.random.normal(drift, volatility, steps))
    
    return brownian_motion

def generate_garch_data(steps=1000):
    """
    Generates time series data using a GARCH(1, 1) model, often used to simulate financial time series.
    
    Parameters:
    - steps: int
        Number of time steps to generate.
    
    Returns:
    - garch_data: np.array
        Generated GARCH time series data.
    """
    # Generate random noise for the GARCH model
    np.random.seed(0)
    random_data = np.random.normal(0, 1, steps)
    
    # Fit a GARCH(1, 1) model to the random data
    garch_model = arch_model(random_data, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp="off")

    # Forecast the next steps using the fitted model
    forecast = garch_fit.forecast(horizon=steps)

    # Get the conditional volatility (sigma) from the forecast
    garch_data = forecast.variance.values[-1]

    return garch_data


def generate_poisson_process(lam=5, steps=1000):
    """
    Generates a time series representing a Poisson process (e.g., transaction counts).
    
    Parameters:
    - lam: float
        The rate or expected number of events per time unit (lambda).
    - steps: int
        Number of time steps to generate.
    
    Returns:
    - poisson_data: np.array
        Generated Poisson process data.
    """
    poisson_data = np.random.poisson(lam, steps)
    
    return poisson_data

def generate_random_walk(steps=1000, drift=0.0, volatility=1.0):
    """
    Generates a random walk time series with optional drift and volatility.
    
    Parameters:
    - steps: int
        Number of time steps to generate.
    - drift: float
        The expected drift per step (representing a steady trend in the data).
    - volatility: float
        The randomness or noise added to each step.

    Returns:
    - random_walk: np.array
        Generated random walk time series.
    """
    # Generate random steps
    random_steps = np.random.normal(drift, volatility, steps)
    
    # Calculate cumulative sum to generate the random walk
    random_walk = np.cumsum(random_steps)

def generate_seasonal_data(steps=1000, seasonality_period=100, noise_std=1.0):
    """
    Generates a seasonal time series with random noise.
    
    Parameters:
    - steps: int
        Number of time steps to generate.
    - seasonality_period: int
        The period of the seasonality (e.g., daily cycle).
    - noise_std: float
        The standard deviation of the added noise.
    
    Returns:
    - seasonal_data: np.array
        Generated seasonal time series with noise.
    """
    time = np.arange(steps)
    seasonal_signal = 10 * np.sin(2 * np.pi * time / seasonality_period)
    noise = np.random.normal(0, noise_std, steps)
    
    seasonal_data = seasonal_signal + noise
    return seasonal_data

def add_anomalies(data, num_anomalies=5, anomaly_factor=5):
    """
    Adds anomalies to a time series by injecting random spikes or dips.
    
    Parameters:
    - data: np.array
        The original time series data.
    - num_anomalies: int
        Number of anomalies to add.
    - anomaly_factor: float
        The factor by which to increase or decrease the value at anomaly points.
    
    Returns:
    - data_with_anomalies: np.array
        Time series data with added anomalies.
    """
    if data is None:
        raise ValueError("Data cannot be None.")
    data_with_anomalies = data.copy()
    anomaly_indices = np.random.choice(len(data), num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Randomly choose whether it's a spike or dip
        anomaly_type = np.random.choice([1, -1])
        data_with_anomalies[idx] += anomaly_type * anomaly_factor * np.std(data)
    
    return data_with_anomalies
