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

def ValveModel(A, alpha, f0, Fs, duration):
    t = np.arange(0, duration, 1/Fs)

    h = A * np.exp(-alpha*t) * np.cos(2*np.pi*f0*t)

    return t, h

def PlotTime(t, h):
    plt.figure(figsize=(10, 4))
    plt.plot(t, h)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Time-domain response")
    plt.grid()
    plt.show()

t,h = ValveModel(1, 30, 60, 48000, 0.5)
PlotTime(t, h)

