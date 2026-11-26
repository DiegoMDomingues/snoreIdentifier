class SnoreConstants:
    # Constants for snore recording
    WINDOW_LENGTH_SECS = 0.250
    SAMPLE_RATE_HZ = 8000 #44100  # The sampling frequency set in Hertz TODO
    FRAME_SIZE_BYTES = 2
    REC_BUFFER_SIZE_SAMPLES = SAMPLE_RATE_HZ // 10  # The size of the buffer for the recording, in samples
    AUDIO_BLOCK_LENGTH_SECS = 80
    OVERLAPPING = 0.5
    timeIntervalPerAudioSample = 1800  # Record a sample Audio each 30 minutes
    maxTestDurationinSeconds = 12 * 3600  # 12 hours in seconds

    minBatteryLevel2Recording = 0.2  # Minimum battery level to continue recording
    stopRecordingIfBiggerThan = 12.0  # If the audio recording exceeds 12 hours, stop recording
