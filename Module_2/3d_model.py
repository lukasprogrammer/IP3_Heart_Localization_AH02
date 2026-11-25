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

def model_3d (M, T, A, P, t):

    valve = np.array([6.37, 3.15, 6.00],
                     [0.94, 2.07, 5.50],
                     [5.50, 3.50, 3.60],
                     [3.90, 4.00, 4.50])/100

    microphone = np.array([0, 0, 0],
                          [0, 5, 0],
                          [5, 0, 0],
                          [5, 5, 0],
                          [10, 0, 0],
                          [10, 5, 0])/100

    d_vm = np.zeros((4, 6))

    for i in range(4):
        for m in range(6):
            d_vm[i, m] = np.linalg.norm(valve[i] - microphone[m])

    v = 60 
    G = 1/d_vm
    delay = d_vm/v
        
    s = np.vstack(M, T, A, P)

    X = np.zeros((6, len(t)))

    for m in range(6):
        for i in range(4):
            X[m, :] += G[i, m] * np.interp(t - delay[i, m], t, s[i],left=0, right=0)

    return X