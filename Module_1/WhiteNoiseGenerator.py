import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as sci

Fs = 48000
N = Fs * 10

uwn = np.random.uniform(-1, 1, N)
uwn1 = np.zeros(N)

scaleduwn = np.int16(uwn / np.max(np.abs(uwn)) * 32767)
scaleduwn1 = np.int16(uwn1 / np.max(np.abs(uwn1)) * 32767)

result = np.vstack((scaleduwn, scaleduwn1)).T
sci.write('Module_1/Recordings/UwnMono.wav', Fs, result)
samplerate, data = sci.read('Module_1/Recordings/UwnMono.wav')

print(data.shape)

# gwn = np.random.normal(0, 1, N)
# scaledgwn = np.int16(gwn / np.max(np.abs(gwn)) * 32767)
# sci.write('Module_1/Recordings/Gwn2.wav', Fs, scaledgwn)

# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.plot(gwn)
# plt.title('White Noise (Uniformly Distributed)')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')

# plt.show()