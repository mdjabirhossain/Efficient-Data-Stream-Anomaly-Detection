import numpy as np
from detect_anomalies_zscore import detect_anomalies_zscore, zscore_anomaly_detection_optimized
from detect_anomalies_emwa import detect_anomalies_ewma
from detect_anomalies_sh_esd import sh_esd, perform_esd_test
from visualize_data import visualize_anomalies, visualize_anomalies_static
from generate_data import generate_arima_data, generate_brownian_motion, generate_garch_data, generate_poisson_process, generate_random_walk, generate_seasonal_data, add_anomalies

def main():
    # This menu system allows users to interactively select options for generating and analyzing time series data.
    # It provides options to generate random data points using different methods, choose a file for analysis,
    # or exit the program. The system guides the user through the selection process, ensuring valid inputs
    # and providing feedback for invalid choices.
    data_stream = None
    print("Welcome to the Anomaly Detection System!")

    while True:
        print("Please select an option:")
        print("1. Generate random time series data points")
        print("2. Choose a file")
        print("3. Exit")
        user_input = input("Your choice (1-3): ")
        try:
            choice = int(user_input)
            if choice == 1:
                print("Please select a method to generate the data points:\n")
                print("1. ARIMA (Autoregressive Integrated Moving Average)")
                print("2. Brownian Motion")
                print("3. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)")
                print("4. Poisson Process for Transaction Counts")
                print("5. Seasonal Data with Noise")
                generation_method = input("Your choice (1-5): ")

                while True:
                    try:
                        method_choice = int(generation_method)
                        if method_choice < 1 or method_choice > 6:
                            raise ValueError
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid data generation method between 1 and 6.\n")
                        print("1. ARIMA (Autoregressive Integrated Moving Average)")
                        print("2. Brownian Motion")
                        print("3. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)")
                        print("4. Poisson Process for Transaction Counts")
                        print("5. Seasonal Data with Noise")
                        generation_method = input("Please enter your choice (1-6): ")

                method_choice = int(generation_method)

                num_data_points = input("How many random data points do you want to generate? ")
                while True:
                    try:
                        num_data_points = int(num_data_points)
                        if not isinstance(num_data_points, int) or num_data_points <= 5:
                            raise ValueError
                        num_data_points = int(num_data_points)
                        break
                    except ValueError:
                        num_data_points = input("Invalid input. Please enter an integer greater than 5 for the number of data points: ")
                
                num_anomalies = input("How many anomalies do you want to add? (Between 1 and {} inclusive): ".format(num_data_points // 10))
                while True:
                    try:
                        num_anomalies = int(num_anomalies)
                        if num_anomalies < 1 or num_anomalies > (num_data_points // 10) + 1:
                            raise ValueError
                        break
                    except ValueError:
                        num_anomalies = input("Invalid input. Please enter a valid number of anomalies between 1 and {}: ".format(num_data_points // 10))

                print("Generating random data points...")
                try:
                    if method_choice == 1:
                        data_stream = generate_arima_data(steps=num_data_points)
                    elif method_choice == 2:
                        data_stream = generate_brownian_motion(steps=num_data_points)
                    elif method_choice == 3:
                        data_stream = generate_garch_data(steps=num_data_points)
                    elif method_choice == 4:
                        data_stream = generate_poisson_process(steps=num_data_points)
                    elif method_choice == 5:
                        data_stream = generate_seasonal_data(steps=num_data_points)
                except Exception as e:
                    print(f"An error occurred during data generation: {e}")
                    continue
                
                try:
                    data_stream = add_anomalies(data=data_stream, num_anomalies=num_anomalies)
                except ValueError as e:
                    print(f"An error occurred while adding anomalies: {e}")
                    continue
                
            elif choice == 2:
                print("Please ensure the file is in the correct format.")
                print("The file should contain a 1D time series of floating point numbers,")
                print("each on a new line. Please see the example.txt file for reference.")
                file_path = input("Please provide the path to the file: ")
                try:
                    with open(file_path, 'r') as file:
                        data_stream = np.genfromtxt(file, skip_header=1, dtype=None, encoding=None)
                except FileNotFoundError:
                    raise ValueError("The file path is not valid. Please ensure the file exists.")
            elif choice == 3:
                break
            else:
                raise ValueError
        except ValueError:
            print("Invalid input. Please enter a valid option.")
            continue
        
        detection_methods = ["Z-Score", "EWMA", "Seasonal Hybrid ESD"]
        detection_method = input("Choose a method to detect anomalies: \n1. Z-Score \n2. EWMA \n3. Seasonal Hybrid ESD \nYour choice (1-3): ")
        while True:
            try:
                detection_method = int(detection_method)
                if detection_method < 1 or detection_method > 3:
                    raise ValueError
                break
            except ValueError:
                detection_method = input("Invalid input. Please enter a valid number for the detection method (1-3): ")
                
        if detection_method == 1:
            anomalies = detect_anomalies_zscore(data_stream)
        elif detection_method == 2:
            anomalies = detect_anomalies_ewma(data_stream)
        elif detection_method == 3:
            try:
                anomalies = sh_esd(data_stream, period=100, max_anomalies=0.05)
            except ValueError as e:
                # Handle the error (log, print, or fallback) and continue execution
                print(f"An error occurred during anomaly detection: {e}")
                continue

        visualization_mode = input("Choose a visualization mode: \n1. Static \n2. Real-Time \nYour choice (1-2): ")
        while True:
            try:
                visualization_mode = int(visualization_mode)
                if visualization_mode < 1 or visualization_mode > 2:
                    raise ValueError
                break
            except ValueError:
                visualization_mode = input("Invalid input. Please enter a valid number for the visualization mode (1-2): ")
        
        if visualization_mode == 1:
            visualize_anomalies_static(data_stream, anomalies, detection_methods[detection_method-1])
        elif visualization_mode == 2:
            visualize_anomalies(data_stream, anomalies, detection_methods[detection_method-1])

    print("Thank you for using the Anomaly Detection Tool. Have a great day!")

if __name__ == "__main__":
    main()