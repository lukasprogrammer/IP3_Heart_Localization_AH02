from scipy.signal import ShortTimeFFT
import numpy as np


def narrowband_Rx(X,Fs,nperseg):

  SFT = ShortTimeFFT.from_window('hann', Fs, nperseg, noverlap=nperseg//2)
  Sx_all = SFT.stft(X) # complex, shape (M, K, L)
  f_bins = SFT.f # F_k = k*Fs/nperseg
  M,K,L=Sx_all.shape
  print(Sx_all.shape)

  Rx_all = np.zeros((K,M,M), dtype = complex)
  for k in range(K):
      Xk = Sx_all[:,k,:]    
      Rx_all[k] = Xk @ Xk.conj().T/L
    

  return f_bins, Rx_all


def generate_source(N):
   return (np.random.randn(N) + 1j*np.random.randn(N))/ np.sqrt(2)

def datamodel(M,N,theta0)
    theta
    Q = len(


