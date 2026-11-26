import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, rfftfreq, rfft
from scipy.io import wavfile
import math
from scipy.signal import ShortTimeFFT


fs = 48000
nperseg = 1024
t = np.arange(int(1.0 * fs)) / fs
x = np.cos(2*np.pi*2000*t)                 # 1 s of 2 kHz

SFT = ShortTimeFFT.from_window('hann', fs, nperseg, noverlap= nperseg//2)   # 50% overlap?
Sx = SFT.stft(x)
print(Sx.shape)          
print(SFT.f[:5])        



N = 8                         # 8 samples
n = np.arange(N)              # n = 0,1,...,7
x = np.cos(2*np.pi*1*n/N)     # a cosine completing exactly 1 cycle in 8 samples

# --- the DFT written exactly like the formula ---
X = np.zeros(N, dtype=complex)
for k in range(N):                                   # for each frequency bin k
    test_tone = np.exp(-1j*2*np.pi*k*n/N)            # e^{-j 2π k n / N}
    X[k] = np.sum(x * test_tone)                     # Σ_n x[n] · test tone

print("my DFT:   ", np.round(X, 3))
print("np.fft:   ", np.round(np.fft.fft(x), 3))