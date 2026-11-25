import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.fft import fft, rfftfreq, rfft
from scipy.io import wavfile
import math
from scipy.signal import spectrogram
from scipy.signal import ShortTimeFFT, spectrogram
from scipy.signal.windows import gaussian
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.fft import fft, rfftfreq, rfft
from scipy.io import wavfile
import math
import librosa



# Extracts the segments out of the original signal for concatiatinating
def Extract_Segments(Signal, boundaries, Fs):
    segments = []

    for start, end in boundaries:
        a = round(start * Fs/4000)
        b = round(end * Fs/4000)

        segment = Signal[a:b]

        if len(segment) > 0:
            segments.append(segment)

    return segments

# Finds the boundaries of the peaks of S1 and S2, for extracting 
def Find_Boundaries(see, peaks, threshold=0.2, max_duration=0.2):
    boundaries = []
    Fs_see = 4000
    max_samples = int(max_duration*Fs_see)

    for p in peaks:
        limit = threshold * see[p]

        start = p
        while(start > 0 and p-start < max_samples and see[start] > limit):
            start -= 1

        end = p
        while(end < len(see) and end - p < max_samples and see[end] > limit):
            end += 1

        boundaries.append((start,end))

    return boundaries


# Removing The unreliable pairs of S1 and S2 detected in Segmentation()
def Remove_Unreliable_Pairs(S1, S2, unreliable, Fs_see=4000):

    margin = int(0.05 * Fs_see) 
    S1_clean = []
    S2_clean = []

    for s1,s2 in zip(S1, S2):
        valid = True

        for start, end in unreliable:
            if start - margin <= s1 <= end + margin or start - margin <= s2 <= end + margin:
                valid = False
                break

        if valid:
            S1_clean.append(s1)
            S2_clean.append(s2)



    return np.array(S1_clean, dtype=int), np.array(S2_clean, dtype=int)


# Determining the best quality channel to perform peak analysis on, using quality ratio which consists
# Of the median of the peak_strengths divided by the median of the background noise
# In Segmentation() the channel with the highest quality gets selected and used for analisys

def Channel_Quality(data, Fs):

    # Bandpass filtering
    b, a = signal.butter(2, [10, 800], 
                         btype='bandpass', fs=Fs)
    filtered = signal.filtfilt(b, a, data.astype(float))

    # Downsampling to 4000 Hz
    g = math.gcd(Fs, 4000)

    filtered_ds = signal.resample_poly(
        filtered,
        4000 // g,
        Fs // g
    )

    Fs = 4000

    # Normalize for Shannon energy
    scale = np.percentile(np.abs(filtered_ds), 99)

    nor_filtered_ds = 0.6 * filtered_ds / (scale + 1e-12)

    nor_filtered_ds = np.clip(nor_filtered_ds, -0.6, 0.6)

    # Shannon energy
    x2 = np.square(nor_filtered_ds)
    sh_energy = -x2 * np.log10(np.maximum(x2, 1e-12))

    # Shannon energy envelope
    d, c = signal.butter(2, 15, btype='lowpass', fs=Fs)
    see = signal.filtfilt(d, c, sh_energy)
    see = np.maximum(see, 0)

    see = see / (np.max(see) + 1e-12)

    # Peak detection
    peaks, properties = signal.find_peaks(
        see,
        height=0.1,
        distance=int(0.15 * Fs),
        prominence=0.05
    )

    if len(peaks) < 3:
        return 0, see, peaks

    peak_strength = np.median(properties["prominences"])

    # Background noise estimation
    mask = np.ones(len(see), dtype=bool)

    for p in peaks:
        start = max(0, p - int(0.1 * Fs))
        end = min(len(see), p + int(0.1 * Fs))
        mask[start:end] = False

    if not np.any(mask):
        return 0, see, peaks

    background = np.median(see[mask])

    quality = peak_strength / (background + 1e-6)

    return quality, see, peaks


def Segmentation(Signal, Fs):

    # Support mono and multichannel signals
    if Signal.ndim == 1:
        Signal = Signal[:, np.newaxis]

    n_channels = Signal.shape[1]

    qualities = []
    all_see = []
    all_peaks = []

    Fs_see = 4000

    # Determine best channel
    for ch in range(n_channels):

        quality, see, peaks = Channel_Quality(
            Signal[:, ch], Fs
        )

        qualities.append(quality)
        all_see.append(see)
        all_peaks.append(peaks)

        print(f"Channel {ch+1}: quality = {quality:.2f}")
        print(f"Detected peaks: {len(peaks)}")

    best_channel = np.argmax(qualities)

    print(f"\nBest channel: {best_channel+1}")

    see = all_see[best_channel]
    peaks = all_peaks[best_channel]

    # Calculate intervals
    intervals = np.diff(peaks) / Fs_see

    print("Peak times:", peaks / Fs_see)
    print("Intervals:", intervals)

    # Plot detected peaks for debugging
    t_see = np.arange(len(see)) / Fs_see

    plt.figure(figsize=(15, 4))
    plt.plot(t_see, see)
    plt.plot(t_see[peaks], see[peaks], "rx")
    plt.xlabel("Time [s]")
    plt.title("Shannon energy envelope")
    plt.grid()
    plt.show()

    # Check sufficient number of peaks
    if len(peaks) < 3:
        raise ValueError(
            "Too few peaks detected to estimate heartbeat intervals"
        )

    # Estimate systole and diastole
    threshold = (
        np.percentile(intervals, 25) +
        np.percentile(intervals, 75)
    ) / 2

    short = intervals[intervals < threshold]
    long = intervals[intervals >= threshold]

    if len(short) == 0 or len(long) == 0:
        raise ValueError(
            "Cannot distinguish systole and diastole"
        )

    S = np.median(short)
    D = np.median(long)
    T = S + D

    print(f"Systole: {S:.3f} s")
    print(f"Diastole: {D:.3f} s")
    print(f"Heartbeat period: {T:.3f} s")
    print(f"Heart rate: {60/T:.1f} BPM")

    # Classify S1 and S2
    S1 = []
    S2 = []

    tolerance = 0.25
    i = 0

    while i < len(peaks) - 1:

        dt = (peaks[i+1] - peaks[i]) / Fs_see

        if abs(dt - S) < tolerance * S:

            S1.append(peaks[i])
            S2.append(peaks[i+1])

            i += 2

        else:
            i += 1

    S1 = np.array(S1, dtype=int)
    S2 = np.array(S2, dtype=int)

    # Flag suspicious intervals
    unreliable = []

    for i in range(len(peaks) - 1):

        dt = (peaks[i+1] - peaks[i]) / Fs_see

        valid_s = abs(dt - S) < tolerance * S
        valid_d = abs(dt - D) < tolerance * D

        if not (valid_s or valid_d):
            unreliable.append((peaks[i], peaks[i+1]))

    # Remove unreliable pairs
    S1_clean, S2_clean = Remove_Unreliable_Pairs(
        S1, S2, unreliable
    )

    print("Reliable S1:", len(S1_clean))
    print("Reliable S2:", len(S2_clean))

    # Find boundaries
    S1_boundaries = Find_Boundaries(see, S1_clean)
    S2_boundaries = Find_Boundaries(see, S2_clean)

    # Extract original signal segments
    S1_segments = Extract_Segments(
        Signal, S1_boundaries, Fs
    )

    S2_segments = Extract_Segments(
        Signal, S2_boundaries, Fs
    )

    # Check before concatenating
    if len(S1_segments) == 0 or len(S2_segments) == 0:
        raise ValueError("No reliable S1/S2 segments found")

    # Concatenate segments
    S1_recording = np.concatenate(S1_segments, axis=0)
    S2_recording = np.concatenate(S2_segments, axis=0)

    print("S1 shape:", S1_recording.shape)
    print("S2 shape:", S2_recording.shape)

    return S1_recording, S2_recording


# Fs, Data = wavfile.read("Module_1/Recordings/Model_Heart.wav")

# # Convert to floating point
# Data = Data.astype(float)

# # Normalize
# Data = Data / (np.max(np.abs(Data)) + 1e-12)

# # Ensure shape is (samples, channels)
# if Data.ndim == 1:
#     Data = Data[:, np.newaxis]


Data, Fs = librosa.load(
    "Module_1/Recordings/TestRecording.mp3",
    sr=None,
    mono=False
)

# Ensure shape is (samples, channels)
if Data.ndim == 1:
    Data = Data[:, np.newaxis]
else:
    Data = Data.T

# Normalize
Data = Data / (np.max(np.abs(Data)) + 1e-12)

S1_recording, S2_recording = Segmentation(Data, Fs)


fig, axes = plt.subplots(2, 1, figsize=(15, 6))

t1 = np.arange(len(S1_recording)) / Fs
t2 = np.arange(len(S2_recording)) / Fs

axes[0].plot(t1, S1_recording[:, 0])
axes[0].set_title("S1 Recording")

axes[1].plot(t2, S2_recording[:, 0])
axes[1].set_title("S2 Recording")

plt.tight_layout()
plt.show()