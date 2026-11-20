
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



start, end = int(2.1 * Fs), int(3.3 *Fs)
data_sh_plt = see[start:end]
time_pv_plt = time_pv[start:end]
plt.plot(time_pv_plt, data_sh_plt)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude ")
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

