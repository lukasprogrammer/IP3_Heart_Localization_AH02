import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

Fs = 48000
N = Fs * 10

uwn = np.random.uniform(-1, 1, N)
scaleduwn = np.int16(uwn / np.max(np.abs(uwn)) * 32767)
write('Module_1/Recordings/Uwn.wav', Fs, scaleduwn)

gwn = np.random.normal(0, 1, N)
scaledgwn = np.int16(gwn / np.max(np.abs(gwn)) * 32767)
write('Module_1/Recordings/Gwn.wav', Fs, scaledgwn)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(gwn)
plt.title('White Noise (Uniformly Distributed)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.show()