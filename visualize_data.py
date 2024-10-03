import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize_anomalies_static(data_stream, anomalies_list, algorithm_name):
    """
    Visualizes the data stream and highlights detected anomalies in a static plot.
    
    Parameters:
    - data_stream: np.array
        The continuous data stream.
    
    - anomalies_list: list of tuples
        A list of detected anomalies from the anomaly detection algorithm.
    
    - algorithm_name: str
        The name of the anomaly detection algorithm used.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the data stream
    ax.plot(data_stream, label='Data Stream')
    
    # Extract anomaly indices and values
    anomaly_x = [idx for idx, _ in anomalies_list]
    anomaly_y = [value for _, value in anomalies_list]
    
    # Highlight anomalies
    ax.scatter(anomaly_x, anomaly_y, color='red', label='Anomalies', zorder=5)
    
    # Add title and labels
    ax.set_title(f'Anomaly Detection using {algorithm_name}')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Display the plot
    plt.show()


def visualize_anomalies(data_stream, anomalies_list, algorithm_name):
    """
    Visualizes the data stream and highlights detected anomalies.
    
    Parameters:
    - data_stream: np.array
        The continuous data stream.
    
    - anomalies_list: list of tuples
        A list of detected anomalies from the anomaly detection algorithm.
    
    - algorithm_name: str
        The name of the anomaly detection algorithm used.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    x_data = []
    y_data = []
    anomaly_x = [idx for idx, _ in anomalies_list]
    anomaly_y = [value for _, value in anomalies_list]

    def animate(i):
        x_data.append(i)
        y_data.append(data_stream[i])
        ax.clear()
        ax.plot(x_data, y_data, label='Data Stream')
        ax.scatter(anomaly_x, anomaly_y, color='red', label='Anomalies')
        ax.set_title(f'Real-Time Anomaly Detection using {algorithm_name}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.set_xlim(0, len(data_stream))
        ax.set_ylim(min(data_stream) - 10, max(data_stream) + 10)

    ani = animation.FuncAnimation(fig, animate, frames=len(data_stream), interval=10, repeat=False)
    plt.show()
