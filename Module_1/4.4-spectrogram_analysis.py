
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


Fs, av = wavfile.read("Chapter_2_Assignment/85340_AV.wav")
Fs, mv = wavfile.read("Chapter_2_Assignment/85340_MV.wav")
Fs, pv = wavfile.read("Chapter_2_Assignment/85341_PV.wav")
Fs, tv = wavfile.read("Chapter_2_Assignment/85340_TV.wav")


low = 10/(Fs/2)
high = 800/(Fs/2)
b, a = signal.butter(2, [low, high], btype='band')
filtered_data_av = signal.filtfilt(b,a,av)
filtered_data_mv = signal.filtfilt(b,a,mv)
filtered_data_pv = signal.filtfilt(b,a,pv)
filtered_data_tv = signal.filtfilt(b,a,tv)

w, H = signal.freqz(b, a, worN=[2000], fs=Fs)

time_av = np.arange(filtered_data_av.size)/Fs
time_mv = np.arange(filtered_data_mv.size)/Fs
time_pv = np.arange(filtered_data_pv.size)/Fs
time_tv = np.arange(filtered_data_tv.size)/Fs



F_av = rfft(filtered_data_av)
F_mv = rfft(filtered_data_mv)
F_pv = rfft(filtered_data_pv)
F_tv = rfft(filtered_data_tv)

def downsampling(Fs, pv):

    M = int(Fs/4000)

    pv_new = pv[::M]

    return pv_new

Fs=4000
down_filter_pv = downsampling(Fs, filtered_data_pv)
normalized_pv = down_filter_pv/np.max(abs(down_filter_pv))


x2 = np.square(normalized_pv)
sh_energy = -x2 * np.log10(np.maximum(x2, 1e-12))


d, c = signal.butter(2, 15, btype='lowpass', fs=Fs)
see = signal.filtfilt(d, c, sh_energy)
see = see[0:15*Fs]
time_pv = time_pv[0:15*Fs]
normalized_see = see/np.max(abs(see))

peaks, properties = signal.find_peaks(normalized_see, 0.1, distance=int(0.15 * Fs), prominence=0.05)
peak_times = peaks / Fs
intervals = np.diff(peak_times)

print("Peak times:", peak_times)
print("Intervals:", intervals)


def search_missing_peak(see, predicted_time, Fs, search_width=0.03):
    center = int(predicted_time * Fs)
    half_width = int(search_width * Fs)

    start = max(0, center - half_width)
    end = min(len(see), center + half_width + 1)

    if start >= end:
        return None

    local_peak = start + np.argmax(see[start:end])

    return local_peak

S1 = []
S2 = []

i = 0

while i < len(peaks) - 2:
    interval1 = peak_times[i+1] - peak_times[i]
    interval2 = peak_times[i+2] - peak_times[i+1]

    if interval1 < interval2:
        S1.append(peaks[i])
        S2.append(peaks[i+1])
        i += 2
    else:
        i+=1


S1 = np.array(S1, dtype=int)
S2 = np.array(S2, dtype=int)

S1_times = time_pv[S1]

heart_period = np.median(np.diff(S1_times))



valid_intervals = intervals[
    (intervals > 0.15) & (intervals < 0.8)
]


threshold = np.median(valid_intervals)

short_intervals = valid_intervals[valid_intervals < threshold]
long_intervals = valid_intervals[valid_intervals >= threshold]

S = np.median(short_intervals)
D = np.median(long_intervals)

# Expected heartbeat timing
T = S + D
tolerance = 0.15

tolerance = 0.25

unreliable_intervals = []
possible_missing_intervals = []

for i in range(len(peaks) - 1):

    gap = (peaks[i+1] - peaks[i]) / Fs

    matches_S = abs(gap - S) < tolerance * S
    matches_D = abs(gap - D) < tolerance * D
    matches_T = abs(gap - T) < tolerance * T

    start = peaks[i] / Fs
    end = peaks[i+1] / Fs

    if matches_S or matches_D:
        continue

    elif matches_T:
        possible_missing_intervals.append((start, end))

    else:
        unreliable_intervals.append((start, end))

print("Unreliable intervals:", len(unreliable_intervals))
print("Possible missing-peak intervals:", len(possible_missing_intervals))



peak_times = peaks / Fs

missing_candidates = []
missing_times = []

for i in range(len(peaks) - 1):

    gap = peak_times[i+1] - peak_times[i]

    # A gap approximately equal to a complete heartbeat
    if abs(gap - T) < tolerance * T:

        # Two possibilities:
        # 1. S1 -> missing S2 -> S1
        # 2. S2 -> missing S1 -> S2

        predicted_times = [
            peak_times[i] + S,
            peak_times[i] + D
        ]

        for predicted_time in predicted_times:

            center = int(predicted_time * Fs)
            half_width = int(0.08 * Fs)

            start = max(0, center - half_width)
            end = min(len(normalized_see), center + half_width)

            segment = normalized_see[start:end]

            if len(segment) < 3:
                continue

            # Search for peaks with a lower threshold
            local_peaks, properties = signal.find_peaks(
                segment,
                prominence=0.02,
                height=0.03
            )

            if len(local_peaks) > 0:

                # Select peak closest to predicted position
                nearest = np.argmin(
                    abs((local_peaks + start) - center)
                )

                recovered_peak = local_peaks[nearest] + start

                # Only add peaks that were not already detected
                if recovered_peak not in peaks:
                    missing_candidates.append(recovered_peak)
                    missing_times.append(recovered_peak / Fs)

missing_candidates = np.unique(
    np.array(missing_candidates, dtype=int)
)

print("Possible missing peaks:", missing_candidates / Fs)
print(f"S = {S:.3f} s")
print(f"D = {D:.3f} s")
print(f"T = {T:.3f} s")


def extract_sound_interval(see, peak, threshold_fraction=0.20):

    threshold = threshold_fraction * see[peak]

    # Search left
    left = peak
    while left > 0 and see[left] > threshold:
        left -= 1

    # Search right
    right = peak
    while right < len(see)-1 and see[right] > threshold:
        right += 1

    return left, right

S1_intervals = []
S2_intervals = []

for p in S1:
    left, right = extract_sound_interval(normalized_see, p)
    S1_intervals.append((left, right))

for p in S2:
    left, right = extract_sound_interval(normalized_see, p)
    S2_intervals.append((left, right))



start, end = int(2.1 * Fs), int(3.3 *Fs)
data_sh_plt = normalized_see[start:end]
time_pv_plt = time_pv[start:end]



plt.plot(time_pv, normalized_see)
plt.plot(time_pv[peaks], normalized_see[peaks], "rx", label="Detected peaks")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude ")


plt.plot(
    time_pv[S1],
    normalized_see[S1],
    "go",
    markersize=8,
    label="S1"
)

plt.plot(
    time_pv[S2],
    normalized_see[S2],
    "ro",
    markersize=8,
    label="S2"
)

if len(missing_candidates) > 0:
    plt.plot(
        time_pv[missing_candidates],
        normalized_see[missing_candidates],
        "mo",
        markersize=10,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Possible missing peaks"
    )

for p1 in S1:
    later_s2 = S2[S2 > p1]

    if len(later_s2) == 0:
        continue

    p2 = later_s2[0]

    # Only connect plausible consecutive detected peaks
    if p2 not in peaks[np.where(peaks == p1)[0][0] + 1:
                       np.where(peaks == p1)[0][0] + 2]:
        continue

    t1 = time_pv[p1]
    t2 = time_pv[p2]

    plt.annotate(
        "", xy=(t2, 0.12), xytext=(t1, 0.12),
        arrowprops=dict(arrowstyle="<->", color="red", lw=1.5)
    )

    plt.text((t1+t2)/2, 0.15, "S",
            color="red", ha="center", fontsize=12)

# Annotate diastole (S2 -> next S1)
for p2 in S2:
    later_s1 = S1[S1 > p2]

    if len(later_s1) == 0:
        continue

    p1 = later_s1[0]

    t2 = time_pv[p2]
    t1 = time_pv[p1]

    plt.annotate(
        "", xy=(t1, 0.08), xytext=(t2, 0.08),
        arrowprops=dict(arrowstyle="<->", color="blue", lw=1.5)
    )

    plt.text((t1+t2)/2, 0.10, "D",
            color="blue", ha="center", fontsize=12)

for j, (start, end) in enumerate(unreliable_intervals):

    plt.axvspan(
        start, end,
        color="orange",
        alpha=0.2,
        label="Unreliable interval" if j == 0 else None
    )
for i, (start, end) in enumerate(possible_missing_intervals):
    plt.axvspan(
        start, end,
        color="purple",
        alpha=0.12,
        label="Possible missing peak" if i == 0 else None
    )

for i, (left, right) in enumerate(S1_intervals):
    plt.axvspan(
        time_pv[left], time_pv[right],
        color="green", alpha=0.25,
        label="S1 interval" if i == 0 else None
    )

for i, (left, right) in enumerate(S2_intervals):
    plt.axvspan(
        time_pv[left], time_pv[right],
        color="red", alpha=0.25,
        label="S2 interval" if i == 0 else None
    )


plt.xlim(2.1, 3.3)
plt.legend()
plt.show()












# time_av = np.arange(av.size)/Fs

# N = int(len(filtered_data_pv))
# win = ('gaussian', 1e-2 * Fs) 
# SFT = ShortTimeFFT.from_window(win, Fs, nperseg=256, noverlap=220,
# fft_mode='centered', scale_to='psd', phase_shift=None)


# Sx2 = SFT.spectrogram(filtered_data_pv)
# t_lo, t_hi = SFT.extent(N)[:2] 
# Sx_dB = 10 * np.log10(np.fmax(Sx2, 1e-4))

# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].imshow(Sx_dB, origin='lower', aspect='auto', extent=SFT.extent(N))
# ax[0].set_xlim(2.1, 3.3)     
# ax[0].set_ylim(0, 1500)  

# start, end = int(2.1 * Fs), int(3.3 *Fs)
# data_pv_plt = filtered_data_pv[start:end]
# time_pv_plt = time_pv[start:end]
# ax[1].plot(time_pv_plt, data_pv_plt/data_pv_plt.max())
# ax[1].set_xlim(2.1, 3.3)



# ax[1].set_xlabel("Time [s]")
# ax[1].set_ylabel("Amplitude ")
# ax[1].set_title("Heart Sound Time Plot")
# plt.show()

