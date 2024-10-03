from generate_data_stream import generate_data_stream
from detect_anomalies_zscore import detect_anomalies_zscore, zscore_anomaly_detection_optimized
from detect_anomalies_emwa import detect_anomalies_ewma
from detect_anomalies_sh_esd import sh_esd, perform_esd_test
from visualize_data import visualize_anomalies

def main():
    data_stream = generate_data_stream(1000)
    zscore_anoms = detect_anomalies_zscore(data_stream)
    ewma_anoms = detect_anomalies_ewma(data_stream)
    sh_esd_anomalies = sh_esd(data_stream, period=100, max_anomalies=0.05)
    print("zscore anomalies", zscore_anoms)
    print()
    print("emwa anomalies", ewma_anoms)
    print()
    print("sh_esd anomalies", sh_esd_anomalies)

if __name__ == "__main__":
    main()