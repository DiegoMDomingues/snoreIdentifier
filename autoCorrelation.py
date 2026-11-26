import math


class AutoCorrelation:
    @staticmethod
    def calculate(signal):
        max_delay = len(signal)

        # mx is the mean of the signal
        mx = sum(signal) / max_delay

        # sx is the sum of the square deviations of the signal
        sx_aux = [(x - mx) ** 2 for x in signal]
        sx = sum(sx_aux)

        # The denominator is the standard deviation
        denom = math.sqrt(sx * sx)

        # r[delay] is the correlation coefficient at "delay"
        r = [0] * max_delay
        aux = 0

        for delay in range(max_delay):
            sxy = 0.0
            for i in range(max_delay):
                j = i + delay
                if j < 0 or j >= max_delay:
                    continue
                else:
                    sxy += (signal[i] - mx) * (signal[j] - mx)
            r[aux] = sxy / denom
            aux += 1

        return r
