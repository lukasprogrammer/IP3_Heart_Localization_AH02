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

time_av = np.arange(av.size)/Fs
f_axis_av = np.fft.rfftfreq(av.size, d=1/Fs)
F_av = rfft(av)


fig, ax = plt.subplots(4, 2, figsize=(16, 12))

ax[0, 0].plot(time_av, av)
ax[0, 0].set_title("AV_Time")
ax[0, 0].set_xlabel("Time [s]")
ax[0, 0].set_ylabel("Amplitude")
ax[0, 0].grid()

ax[0,1].plot(f_axis_av, np.abs(F_av))
ax[0,1].set_title("AV_Frequency")
ax[0,1].set_xlabel("Frequency [Hz]")
ax[0,1].set_xlim(0, 500)
ax[0,1].set_ylabel("Amplitude")
ax[0,1].grid()

time_mv = np.arange(mv.size)/Fs
f_axis_mv = np.fft.rfftfreq(mv.size, d=1/Fs)
F_mv = rfft(mv)

ax[1,0].plot(time_mv, mv)
ax[1,0].set_title("MV_Time")
ax[1,0].set_xlabel("Time [s]")
ax[1,0].set_ylabel("Amplitude")
ax[1,0].grid()

ax[1,1].plot(f_axis_mv, np.abs(F_mv))
ax[1,1].set_title("MV_Frequency")
ax[1,1].set_xlabel("Frequency [Hz]")
ax[1,1].set_ylabel("Amplitude")
ax[1,1].set_xlim(0, 500)
ax[1,1].grid()


time_pv = np.arange(pv.size)/Fs
f_axis_pv = np.fft.rfftfreq(pv.size, d=1/Fs)
F_pv = rfft(pv)

ax[2,0].plot(time_pv, pv)
ax[2,0].set_title("PV_Time")
ax[2,0].set_xlabel("Time [s]")
ax[2,0].set_ylabel("Amplitude")
ax[2,0].grid()

ax[2,1].plot(f_axis_pv, np.abs(F_pv))
ax[2,1].set_title("PV_Frequency")
ax[2,1].set_xlabel("Frequency [Hz]")
ax[2,1].set_ylabel("Amplitude")
ax[2,1].set_xlim(0, 500)
ax[2,1].grid()

time_tv = np.arange(tv.size)/Fs
f_axis_tv = np.fft.rfftfreq(tv.size, d=1/Fs)
F_tv = rfft(tv)

ax[3,0].plot(time_tv, tv)
ax[3,0].set_title("TV_Time")
ax[3,0].set_xlabel("Time [s]")
ax[3,0].set_ylabel("Amplitude")
ax[3,0].grid()

ax[3,1].plot(f_axis_tv, np.abs(F_tv))
ax[3,1].set_title("TV_Frequency")
ax[3,1].set_xlabel("Frequency [Hz]")
ax[3,1].set_ylabel("Amplitude")
ax[3,1].set_xlim(0, 500)
ax[3,1].grid()


plt.subplots_adjust(hspace=0.9, wspace=0.4)
# plt.tight_layout()
plt.show()