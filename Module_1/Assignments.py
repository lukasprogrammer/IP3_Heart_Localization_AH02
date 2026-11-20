import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.fft import fft, rfftfreq, rfft
from scipy.io import wavfile
import math

Fs, av = wavfile.read("Chapter_2_Assignment/85340_AV.wav")
Fs, mv = wavfile.read("Chapter_2_Assignment/85340_MV.wav")
Fs, pv = wavfile.read("Chapter_2_Assignment/85340_PV.wav")
Fs, tv = wavfile.read("Chapter_2_Assignment/85340_TV.wav")


