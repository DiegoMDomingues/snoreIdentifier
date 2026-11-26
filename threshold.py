import numpy as np


class Threshold:
    # Número de bins no histograma
    bins = 100

    @staticmethod
    def calculate_threshold(signal):
        # Histogram
        if len(signal) == 0:
            return 0

        max_val = max(signal)
        min_val = min(signal)

        step = (max_val - min_val) / Threshold.bins
        aux = 0.0

        n_elements = [0.0] * Threshold.bins
        between_class_variance = [0.0] * Threshold.bins
        centers = [0.0] * Threshold.bins

        # Preenchendo os bins com os valores do sinal
        for i in range(len(signal)):
            aux = min_val + step
            for j in range(Threshold.bins):
                if signal[i] < aux:
                    n_elements[j] += 1
                    break
                aux += step

        # Calculando os centros dos bins
        for i in range(Threshold.bins):
            centers[i] = (min_val + i * step + min_val + (i + 1) * step) / 2

        sum_n_elements = sum(n_elements)

        # Calculando a variância entre as classes
        for i in range(Threshold.bins):
            sum_wb = 0.0
            sum_wf = 0.0
            for s in range(i):
                sum_wb += n_elements[s]
            for s in range(i, len(n_elements)):
                sum_wf += n_elements[s]

            Wb = sum_wb / sum_n_elements
            Wf = sum_wf / sum_n_elements

            sum_ub = 0.0
            for s in range(i):
                sum_ub += centers[s] * n_elements[s]
            ub = sum_ub / sum_wb if sum_wb != 0 else 0

            sum_uf = 0.0
            for s in range(i, len(n_elements)):
                sum_uf += centers[s] * n_elements[s]
            uf = sum_uf / sum_wf if sum_wf != 0 else 0

            between_class_variance[i] = Wb * Wf * (ub - uf) ** 2

        max_variance = max(between_class_variance)
        index = between_class_variance.index(max_variance)
        threshold = centers[index]

        return threshold