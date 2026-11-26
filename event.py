import numpy as np
from typing import List
from snoreConstants import SnoreConstants
from autoCorrelation import AutoCorrelation
from findPeaks import FindPeaks

class Event:
    def __init__(self):
        self.eventStart = 0.0
        self.eventEnd = 0.0
        self.energyAnt = 0.0
        self.energyPos = 0.0
        self.corr = 0.0
        self.intlNNGrade = 0.0
        self.features = []

    class Peak:
        def __init__(self, x_value: int, y_value: float):
            self.xValue = x_value
            self.yValue = y_value

    @staticmethod
    def find_peaks_in_cross_correlation(sig: np.ndarray) -> List['Event.Peak']:
        peaks = []
        for i in range(1, len(sig) - 1):
            if sig[i] > sig[i - 1] and sig[i] > sig[i + 1] and sig[i] > 0.1:
                peaks.append(Event.Peak(i, sig[i]))
        return peaks

    @staticmethod
    def find_events(energy_array: np.ndarray, threshold: float,
                    t_start: float = 10.0, t_end: float = 70.0) -> List['Event']:
        events = []

        # Passo 1 - Pré-processamento
        media = np.mean(energy_array)
        if threshold <= media:
            return events

        max_energy_value = np.max(energy_array)
        min_energy_value = np.min(energy_array)

        if max_energy_value - min_energy_value < 1:
            return events

        # Passo 2 - Achar eventos
        energy_array_ths = np.copy(energy_array)
        energy_array_ths[energy_array_ths < threshold] = 0.0

        event = None
        dt = SnoreConstants.WINDOW_LENGTH_SECS * SnoreConstants.OVERLAPPING

        for i in range(len(energy_array_ths) - 1):
            t = i * dt
            if energy_array_ths[i] == 0 and energy_array_ths[i + 1] != 0 and t_start <= t <= t_end and event is None:
                event = Event()
                event.eventStart = t
            if energy_array_ths[i] != 0 and energy_array_ths[i + 1] == 0 and t_start <= t <= (t_end + 3) and event is not None:
                event.eventEnd = t
                events.append(event)
                event = None

        if not events:
            return events

        if len(events) >= 60:
            events.clear()
            return events

        # Passo 3 - Pós-processamento
        events = [e for e in events if 0.2 <= (e.eventEnd - e.eventStart) <= 3]

        i = 0
        while i < len(events) - 1:
            if events[i + 1].eventStart - events[i].eventEnd < 1:
                events[i].eventEnd = events[i + 1].eventEnd
                events.pop(i + 1)
            else:
                i += 1

        # Filtros
        k = len(energy_array) / SnoreConstants.AUDIO_BLOCK_LENGTH_SECS
        n_energy = len(energy_array)

        i = 0
        while i < len(events):
            idx_ant_start = max(0, int(np.ceil((events[i].eventStart - 2) * k)))
            idx_ant_end   = max(0, int(np.floor((events[i].eventStart - 0.5) * k)))
            idx_ev_start  = max(0, int(np.ceil(events[i].eventStart * k)))
            idx_ev_end    = min(n_energy, int(np.floor(events[i].eventEnd * k)))
            idx_pos_start = min(n_energy, int(np.ceil((events[i].eventEnd + 0.5) * k)))
            idx_pos_end   = min(n_energy, int(np.floor((events[i].eventEnd + 2) * k)))

            slice_ant    = energy_array[idx_ant_start:idx_ant_end]
            slice_evento = energy_array[idx_ev_start:idx_ev_end]
            slice_pos    = energy_array[idx_pos_start:idx_pos_end]

            soma_ant    = float(np.mean(slice_ant))    if len(slice_ant)    > 0 else 0.0
            soma_evento = float(np.mean(slice_evento)) if len(slice_evento) > 0 else 0.0
            soma_pos    = float(np.mean(slice_pos))    if len(slice_pos)    > 0 else 0.0

            if soma_evento != 0:
                events[i].energyAnt = soma_ant / soma_evento
                events[i].energyPos = soma_pos / soma_evento

            if soma_evento < soma_ant * 1.1 and soma_evento < soma_pos * 1.1:
                events.pop(i)
            else:
                i += 1

        # Processamento final
        for event in events:
            event_center = (event.eventStart + event.eventEnd) / 2
            time_interval = 10
            n_energy = len(energy_array)

            while time_interval > 0:
                ten_sec_before_idx = int(
                    (n_energy * (event_center - time_interval)) / SnoreConstants.AUDIO_BLOCK_LENGTH_SECS) + 1
                ten_sec_after_idx = int(
                    (n_energy * (event_center + time_interval)) / SnoreConstants.AUDIO_BLOCK_LENGTH_SECS) - 1

                if 0 <= ten_sec_before_idx < ten_sec_after_idx < n_energy:
                    break
                time_interval -= 1
            else:
                # Não foi possível achar uma janela válida — usa o array inteiro
                ten_sec_before_idx = 0
                ten_sec_after_idx = n_energy - 1

            # Proteção final contra índices inválidos
            ten_sec_before_idx = max(0, ten_sec_before_idx)
            ten_sec_after_idx = min(n_energy - 1, ten_sec_after_idx)
            if ten_sec_after_idx <= ten_sec_before_idx:
                continue

            corte_corr = energy_array[ten_sec_before_idx:ten_sec_after_idx]

            if len(corte_corr) < 2:
                continue
            correlacao = AutoCorrelation.calculate(corte_corr)
            peaks = Event.find_peaks_in_cross_correlation(correlacao)

            peaks = [p for p in peaks if (p.xValue * 21 / len(correlacao)) >= 1]

            if peaks:
                try:
                    coef_ang = (1 - peaks[0].yValue) / (0 - (peaks[0].xValue - 1))
                    y = np.array([coef_ang * j + 1 for j in range(peaks[0].xValue)])
                    area = np.mean((y - correlacao[:peaks[0].xValue]) ** 2)
                    event.corr = area * peaks[0].yValue
                except Exception:
                    pass

        return events