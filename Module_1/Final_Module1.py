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


def Extract_Segments(Signal, boundaries, Fs):
    segments = []

    for start, end in boundaries:
        a = round(start * Fs/4000)
        b = round(end * Fs/4000)

        segment = Signal[a:b]

        if len(segment) > 0:
            segments.append(segment)

    return segments


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

def Channel_Quality(data, Fs):
    b, a = signal.butter(2, [10, 800], btype='bandpass', fs=Fs)
    filtered = signal.filtfilt(b, a, data.astype(float))

    g = math.gcd(Fs, 4000)

    filtered_ds = signal.resample_poly(filtered,
        4000 // g,
        Fs // g
    )

    Fs = 4000

    scale = np.percentile(np.abs(filtered_ds), 99)

    nor_filtered_ds = filtered_ds / (scale + 1e-12)
    nor_filtered_ds = np.clip(nor_filtered_ds, -1, 1)
    x2 = np.square(nor_filtered_ds)
    sh_energy = -(x2)*np.log10(np.maximum(x2, 1e-12))

    d, c = signal.butter(2, 15, btype='lowpass', fs=4000)
    see = signal.filtfilt(d, c, sh_energy)
    see = np.maximum(see, 0)

    see = see/(np.max(see)+ 1e-12)

    peaks, properties = signal.find_peaks(see, height=0.4,distance=int(0.1*Fs), prominence=0.2)
    if len(peaks) < 2:
        return 0, see, peaks

    peak_strength = np.median(properties["prominences"])

    mask = np.ones(len(see), dtype=bool)
    for p in peaks:
        start = max(0, p - int(0.1*Fs))
        end = min(len(see), p + int(0.1*Fs))
        mask[start:end] = False
    if not np.any(mask):
        return 0, see, peaks

    background = np.median(see[mask])

    quality = peak_strength / (background + 1e-6)

    return quality, see, peaks


def Segmentation(Signal, Fs):

    # We start off with a quality check of the best channel to use for detection
    n_channels = Signal.shape[1]
    qualities = []
    all_see= []
    all_peaks =[]

    for ch in range(n_channels):
        quality, see, peaks = Channel_Quality(Signal[:,ch], Fs)
        qualities.append(quality)
        all_see.append(see)
        all_peaks.append(peaks)

        print(f"Channel {ch+1}: quality = {quality:.2f}")

    best_channel = np.argmax(qualities)

    print(f"\nBest channel: {best_channel+1}")


    see = all_see[best_channel]
    peaks = all_peaks[best_channel]


    # This part detect peaks s1 and s2 very roughly, prone to errors
    Fs_see = 4000

    S1 = []
    S2 = []

    i = 0

    while i < len(peaks) - 2:

        interval1 = peaks[i+1] - peaks[i]
        interval2 = peaks[i+2] - peaks[i+1]

        if interval1 < interval2:
            S1.append(peaks[i])
            S2.append(peaks[i+1])
            i += 2
        else:
            i += 1

    S1 = np.array(S1, dtype=int)
    S2 = np.array(S2, dtype=int)

    intervals = np.diff(peaks)/Fs_see

    S = np.median(intervals[intervals < np.median(intervals)])
    D = np.median(intervals[intervals >= np.median(intervals)])

    T = S + D

    print(f"Systole: {S:.3f} s")
    print(f"Diastole: {D:.3f} s")
    print(f"Heartbeat period: {T:.3f} s")


    #flagging suspicious intervals


    tolerance =0.25
    unreliable = []

    for i in range(len(peaks) - 1):
        dt = (peaks[i+1] - peaks[i])/Fs_see

        valid_s = np.abs(dt - S) < tolerance * S
        valid_d = np.abs(dt - D) < tolerance * D

        if not(valid_d or valid_s):
            unreliable.append((peaks[i], peaks[i+1]))


    S1_clean, S2_clean = Remove_Unreliable_Pairs(
        S1, S2, unreliable
    )

    print("Reliable S1:", len(S1_clean))
    print("Reliable S2:", len(S2_clean))


    S1_boundaries = Find_Boundaries(see, S1_clean)
    S2_boundaries = Find_Boundaries(see, S2_clean)

    S1_segments = Extract_Segments(Signal, S1_boundaries, Fs)
    S2_segments = Extract_Segments(Signal, S2_boundaries, Fs)


    S1_recording = np.concatenate(S1_segments, axis=0)
    S2_recording = np.concatenate(S2_segments, axis=0)

    print("S1 shape:", S1_recording.shape)
    print("S2 shape:", S2_recording.shape)

    return S1_recording, S2_recording


Data, Fs = librosa.load("Module_1/TestRecording.mp3", sr=None, mono=False)

if Data.ndim == 2:
    Data = Data.T

left = Data[:, 0]
right = Data[:, 1]

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