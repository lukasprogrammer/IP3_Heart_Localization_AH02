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
from pathlib import Path


def model_3d (M, T, A, P, t):

    valve = np.array([[6.37, 3.15, 6.00],
                     [0.94, 2.07, 5.50],
                     [5.50, 3.50, 3.60],
                     [3.90, 4.00, 4.50]])/100

    microphone = np.array([[0, 0, 0],
                          [0, 5, 0],
                          [5, 0, 0],
                          [5, 5, 0],
                          [10, 0, 0],
                          [10, 5, 0]])/100

    d_vm = np.zeros((4, 6))

    for i in range(4):
        for m in range(6):
            d_vm[i, m] = np.linalg.norm(valve[i] - microphone[m])

    v = 60 
    G = 1/d_vm
    delay = d_vm/v
        
    s = np.vstack((M, T, A, P)) 

    X = np.zeros((6, len(t)))

    for m in range(6):
        for i in range(4):
            X[m, :] += G[i, m] * np.interp(t - delay[i, m], t, s[i],left=0, right=0)

    return X

Fs = 48000

def ValveModel(duration, Freq, Rel_Amp, ons_del, Fs):


    duration = duration/1000
    ons_del = ons_del/1000

    alpha = -(np.log(0.01))/duration
    w0 = 2 * np.pi * Freq

    num = [Rel_Amp, Rel_Amp*alpha]
    den = [1, 2*alpha, alpha**2 + w0**2]

    system = signal.TransferFunction(num, den)


    t = np.arange(0, duration, 1/Fs)
    t, h = signal.impulse(system, T=t)


    delay_samples = int(round(ons_del*Fs))
    h = np.pad(h, (delay_samples, 0))

    # Update time axis
    t = np.arange(len(h)) / Fs
    # t = np.arange(0, duration, 1/Fs)

    # h = Rel_Amp * np.exp(-alpha*t) * np.cos(2*np.pi*Freq*t)
    # delay_samples = int(round(ons_del*Fs))
    # h = np.pad(h, (delay_samples, 0))
    # t = np.arange(len(h)) / Fs
    return t, h, system

def PlotTime(t, h):
    plt.figure(figsize=(10, 4))
    plt.plot(t, h)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Time-domain response")
    plt.grid()
    plt.show()

def Bandpass_Filter(data, Fs):

    b, a = signal.butter(
        2, [10, 800],
        btype='bandpass',
        fs=Fs
    )

    filtered = signal.filtfilt(b, a, data)

    return filtered
Mt,M, sys_M = ValveModel(60, 140, 1, 70, Fs)
Tt,T, sys_T = ValveModel(50, 150, 0.5, 90, Fs)
At,A, sys_A= ValveModel(50, 100, 0.4, 400, Fs)
Pt,P, sys_P = ValveModel(40, 60, 0.3, 420, Fs)


N = max(len(M), len(T), len(A), len(P))

M = np.pad(M, (0, N-len(M)))
T = np.pad(T, (0, N-len(T)))
A = np.pad(A, (0, N-len(A)))
P = np.pad(P, (0, N-len(P)))

S1 = M + T
S2 = A+P
S = S1+S2

BPM = 66.67

T_cycle = 0.9
N_cycle = int(round(T_cycle * Fs))

N_beats = 10



impulses = np.zeros(N_beats * N_cycle)

impulses[::N_cycle] = 1

M_rep = signal.fftconvolve(impulses, M)[:len(impulses)]
T_rep = signal.fftconvolve(impulses, T)[:len(impulses)]
A_rep = signal.fftconvolve(impulses, A)[:len(impulses)]
P_rep = signal.fftconvolve(impulses, P)[:len(impulses)]

S = np.pad(S, (0, N_cycle - len(S)))
HeartSignal = signal.fftconvolve(impulses, S)

HeartSignal = HeartSignal[:len(impulses)]


HeartSignalfilt = Bandpass_Filter(HeartSignal, Fs)

base = Path(__file__).resolve().parent
path = base / "../Module_1/Recordings/TestRecording.mp3"

recorded, Fs_rec = librosa.load(str(path), sr=None)

recorded_filtered = Bandpass_Filter(recorded, Fs_rec)

t_recorded = np.arange(len(recorded_filtered)) / Fs_rec
model = HeartSignalfilt / np.max(np.abs(HeartSignalfilt))

recorded = recorded_filtered / np.max(np.abs(recorded_filtered))

H_model = np.abs(rfft(model))
H_recorded = np.abs(rfft(recorded))

f_recorded = np.fft.rfftfreq(len(recorded), d=1/Fs_rec)
f_model = np.fft.rfftfreq(len(model), d=1/Fs)

H_recorded = H_recorded / np.max(H_recorded)
H_model = H_model / np.max(H_model)

t = np.arange(len(impulses)) / Fs


# Calculate six microphone signals
X = model_3d(M_rep, T_rep, A_rep, P_rep, t)

# Apply preprocessing filter
X = Bandpass_Filter(X, Fs)

for m in range(6):
    print(f"Microphone {m+1}")
    print("Maximum:", np.max(np.abs(X[m])))
    print("RMS:", np.sqrt(np.mean(X[m]**2)))
    print()


X_norm = X / np.max(np.abs(X))
print(X.shape)


fig, ax = plt.subplots(3, 2, figsize=(12, 6))

ax[0,0].plot(t, X[0])
ax[0,0].set_title("Recorded heart sound 1")
ax[0,0].set_xlim(0, 3)
ax[0,0].grid()

ax[0,1].plot(t, X[1])
ax[0,1].set_title("Simulated heart sound 2")
ax[0,1].set_xlim(0, 3)
ax[0,1].grid()

ax[1,0].plot(t, X[2])
ax[1,0].set_title("Recorded heart sound 3")
ax[1,0].set_xlim(0, 3)
ax[1,0].grid()

ax[1,1].plot(t, X[3])
ax[1,1].set_title("Simulated heart sound 4")
ax[1,1].set_xlim(0, 3)
ax[1,1].grid()

ax[2,0].plot(t, X[4])
ax[2,0].set_title("Recorded heart sound 5")
ax[2,0].set_xlim(0, 3)
ax[2,0].grid()

ax[2,1].plot(t, X[5])
ax[2,1].set_title("Simulated heart sound 6")
ax[2,1].set_xlim(0, 3)
ax[2,1].grid()
# plt.figure(figsize=(14, 5))

# for m in range(6):
#     plt.plot(t, X[m], label=f"Mic {m+1}")

# plt.xlim(0.069, 0.085)
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.title("Microphone arrival time comparison")
# plt.legend()


plt.tight_layout()
plt.show()

