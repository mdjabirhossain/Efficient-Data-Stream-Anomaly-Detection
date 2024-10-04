# Efficient Data Stream Anomaly Detection

This project implements various algorithms to detect anomalies in a continuous data stream. The stream represents real-time sequences of floating-point numbers, simulating various metrics such as financial transactions or system metrics. The goal is to identify unusual patterns such as sudden spikes, deviations from the norm, or changes in trend using several anomaly detection algorithms.

## Features

- **Data Stream Simulation**:

  - Provides five different methods for emulating real-time data streams:
    1. **ARIMA (Autoregressive Integrated Moving Average)**: Used for simulating time series data with trends and autocorrelations.
    2. **Brownian Motion**: A model often used to simulate random movements, such as stock price movements over time.
    3. **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**: Used for simulating time series with changing volatility, such as financial market returns.
    4. **Poisson Process for Transaction Counts**: Simulates discrete events like transaction counts or number of arrivals per unit of time.
    5. **Seasonal Data with Noise**: Generates data with a repeating seasonal pattern and added noise, simulating seasonal trends.

- **Algorithms Implemented**:

  - **Z-Score Anomaly Detection**: Uses statistical thresholds to detect anomalies based on z-scores.
  - **Seasonal Hybrid Extreme Studentized Deviate (S-H-ESD)**: Detects anomalies in seasonal time series data.
  - **Exponentially Weighted Moving Average (EWMA)**: Tracks trends in time series data and flags significant deviations.

- **Real-Time Visualization**:
  - Visualizes the data stream with anomalies flagged for easy analysis.

## Technologies Used

- **Python 3.11**
- **Jupyter Notebook** for visualization and development
- **NumPy** for numerical calculations
- **Pandas** for data manipulation
- **Matplotlib** for visualization
- **SciPy** for statistical functions
- **Statsmodels** for time series decomposition and forecasting
- **ARCH** for GARCH model implementation

## Project Structure

```bash
Efficient-Data-Stream-Anomaly-Detection/
│
├── .gitignore                     # Files and folders to ignore in Git
├── anomaly_detection.ipynb         # Jupyter notebook with experiments
├── anomaly_detection.py            # Main script for running the anomaly detection
├── detect_anomalies_ewma.py        # EWMA anomaly detection implementation
├── detect_anomalies_sh_esd.py      # S-H-ESD anomaly detection implementation
├── detect_anomalies_zscore.py      # Z-Score anomaly detection implementation
├── example.txt                     # Example data or documentation
├── generate_data.py                # Functions for generating time series data
├── README.md                       # Project description and instructions (this file)
├── requirements.txt                # List of required packages
└── visualize_data.py               # Visualization utility for anomaly detection
```

## Setup Instructions

### Prerequisites

Before setting up the project, ensure that you have Python 3.x and `pip` installed on your machine. This project uses Python 3.11.

- If Python or `pip` is not installed, you can download Python from [python.org](https://www.python.org/downloads/) and follow the installation instructions.

git clone https://github.com/yourusername/Efficient-Data-Stream-Anomaly-Detection.git
cd Efficient-Data-Stream-Anomaly-Detection

## Setup Instructions

### 1. Clone the Repository:

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/mdjabirhossain/Efficient-Data-Stream-Anomaly-Detection.git
cd Efficient-Data-Stream-Anomaly-Detection
```

### 2. Create and Activate the Virtual Environment:

It’s recommended to use a virtual environment to isolate the project’s dependencies.

- On **macOS/Linux**:

  ```
  python3 -m venv venv
  source venv/bin/activate

  ```

- On **Windows**:
  ```
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install the Required Packages:

Once the virtual environment is activated, install the required packages using `pip`. Make sure you are in the project directory where `requirements.txt` is located.

```
pip install -r requirements.txt
```

### 4. Run the Anomaly Detection Scripts:

You can now run the anomaly detection scripts provided in the project. For example, to run the anomaly detection application, use the following command:

```
python anomaly_detection.py
```

#### Console Menu System:

Once the script is running, you will be presented with an interactive menu that looks like this:

```
Welcome to the Anomaly Detection System!
Please select an option:
1. Generate random time series data points
2. Choose a file
3. Exit
```

- **Option 1**: Allows you to generate random time series data using different methods, such as ARIMA, Brownian Motion, GARCH, etc.
- **Option 2**: Lets you analyze a pre-existing file.
- **Option 3**: Exits the program.

Once you select a method to generate data points, you'll be guided through further steps to run anomaly detection using the available algorithms, including:

- Z-Score
- Seasonal Hybrid Extreme Studentized Deviate (S-H-ESD)
- Exponentially Weighted Moving Average (EWMA)

#### Example Flow of Running the Application:

- You start the program:

  ```bash
  python anomaly_detection.py
  ```

- You select **1** to generate random data, choose **Brownian Motion** as the generation method, specify **1000 data points**, and inject **10 anomalies**.
- You then select **Z-Score** for anomaly detection and choose **Real-Time** visualization.
- The program displays the anomalies in real-time as they are detected.

### 5. Handling Errors:

If you enter invalid inputs during any of the steps, the system will guide you to re-enter a valid input. For example, if you enter an invalid number for generating data points, you will be asked to try again.

## Contributing

Contributions are welcome! If you would like to contribute, please fork the repository and create a pull request with your proposed changes. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
