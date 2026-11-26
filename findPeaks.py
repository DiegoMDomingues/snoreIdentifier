class FindPeaks:
    @staticmethod
    def calculate(signal, threshold):
        x_peaks = []
        y_peaks = []

        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
                x_peaks.append(i)
                y_peaks.append(signal[i])

        return x_peaks, y_peaks
