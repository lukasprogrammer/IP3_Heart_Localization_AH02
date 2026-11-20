from scipy.signal import ShortTimeFFT, spectrogram
from scipy.signal.windows import gaussian

from scipy.signal import spectrogram
f, t, Sxx = spectrogram(signal, Fs, nperseg=x, noverlap=y)

plt.imshow(t,f,Sxx)

Sx_dB = 10*np.log10(Sxx)