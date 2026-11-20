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


print(Fs)
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

f_axis_av = np.fft.rfftfreq(filtered_data_av.size, d=1/Fs)
f_axis_mv = np.fft.rfftfreq(filtered_data_mv.size, d=1/Fs)
f_axis_pv = np.fft.rfftfreq(filtered_data_pv.size, d=1/Fs)
f_axis_tv = np.fft.rfftfreq(filtered_data_tv.size, d=1/Fs)

F_av = rfft(filtered_data_av)
F_mv = rfft(filtered_data_mv)
F_pv = rfft(filtered_data_pv)
F_tv = rfft(filtered_data_tv)

F_av_nonFilter = rfft(av)
F_mv_nonFilter = rfft(mv)
F_pv_nonFilter = rfft(pv)
F_tv_nonFilter = rfft(tv)


fig, ax = plt.subplots(4, 2, figsize=(10, 8))

damping = -20 * np.log10(abs(H[0]))
print(f"Damping at 2000Hz: {damping * 2}dB")

ax[0, 0].plot(f_axis_av, np.abs(F_av_nonFilter))
ax[0, 0].set_title("AV_NonFilter_Frequency")
ax[0, 0].set_xlabel("freq [Hz]")
ax[0, 0].set_ylabel("Amplitude")
ax[0, 0].grid()

ax[0,1].plot(f_axis_av, np.abs(F_av))
ax[0,1].set_title("AV_Frequency")
ax[0,1].set_xlabel("Frequency [Hz]")
ax[0,1].set_xlim(0, 2000)
ax[0,1].set_ylabel("Amplitude")
ax[0,1].grid()



ax[1,0].plot(f_axis_mv, np.abs(F_mv_nonFilter))
ax[1,0].set_title("MV_NonFilter_Frequency")
ax[1,0].set_xlabel("freq [Hz]")
ax[1,0].set_ylabel("Amplitude")
ax[1,0].grid()

ax[1,1].plot(f_axis_mv, np.abs(F_mv))
ax[1,1].set_title("MV_Frequency")
ax[1,1].set_xlabel("Frequency [Hz]")
ax[1,1].set_xlim(0, 2000)
ax[1,1].set_ylabel("Amplitude")
ax[1,1].grid()

ax[2,0].plot(f_axis_pv, np.abs(F_pv_nonFilter))
ax[2,0].set_title("PV_NonFilter_Frequency")
ax[2,0].set_xlabel("freq [Hz]")
ax[2,0].set_ylabel("Amplitude")
ax[2,0].grid()

ax[2,1].plot(f_axis_pv, np.abs(F_pv))
ax[2,1].set_title("MV_Frequency")
ax[2,1].set_xlabel("Frequency [Hz]")
ax[2,1].set_xlim(0, 2000)
ax[2,1].set_ylabel("Amplitude")
ax[2,1].grid()

ax[3,0].plot(f_axis_tv, np.abs(F_tv_nonFilter))
ax[3,0].set_title("TV_NonFilter_Frequency")
ax[3,0].set_xlabel("freq [Hz]")
ax[3,0].set_ylabel("Amplitude")
ax[3,0].grid()

ax[3,1].plot(f_axis_tv, np.abs(F_tv))
ax[3,1].set_title("TV_Frequency")
ax[3,1].set_xlabel("Frequency [Hz]")
ax[3,1].set_xlim(0, 2000)
ax[3,1].set_ylabel("Amplitude")
ax[3,1].grid()
plt.show()

