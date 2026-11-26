import numpy as np
from snoreConstants import SnoreConstants


class Energy:
    # Número de amostras em uma janela de amostras
    WINDOW_LENGTH_FRAMES = int(SnoreConstants.WINDOW_LENGTH_SECS * SnoreConstants.SAMPLE_RATE_HZ)

    def __init__(self):
        self.energy = [0.0] * (int(SnoreConstants.AUDIO_BLOCK_LENGTH_SECS /
                                   (SnoreConstants.WINDOW_LENGTH_SECS * SnoreConstants.OVERLAPPING)) )

    def calculate(self, audio: np.ndarray):
        aux = int(SnoreConstants.WINDOW_LENGTH_SECS * SnoreConstants.OVERLAPPING * SnoreConstants.SAMPLE_RATE_HZ)

        for i in range(len(self.energy)):
            frame_start = i * aux
            sum_energy = 0.0

            for j in range(0, Energy.WINDOW_LENGTH_FRAMES, 2):
                index = frame_start + j
                if 0 <= index < len(audio):
                    x = float(audio[index])
                    sum_energy += x * x

            self.energy[i] = float(np.log10(2 * sum_energy / Energy.WINDOW_LENGTH_FRAMES + 1))