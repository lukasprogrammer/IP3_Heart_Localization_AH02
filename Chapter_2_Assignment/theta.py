import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, rfftfreq, rfft
from scipy.io import wavfile
import math
from scipy.signal import ShortTimeFFT
from pathlib import Path




v = 343; # m/s
F0 = 500 # Hz
M = 6 
snr = 10 
Q = 2 # how many sources we have
lam = v/F0
delta = 0.5; 
#d = lam * delta; # m

d = 0.10
Nw = 2048

theta = np.deg2rad(np.linspace(-90, 90, 721))


def alin(theta, M, d, v, F0):

    th = np.atleast_1d(theta)       
    tau = (d / v) * np.sin(th) 
    m = np.arange(M)

    Am =  np.exp(-1j*2 *np.pi *F0 * m[:, None] * tau[None, :])

    if np.ndim(theta) == 0:                   
        Am = Am[:, 0]    
    
    return Am

def spatialSpectrum(P, A0, theta, M, d, v, F0):
    
    for i, angle in enumerate(theta):
        A = alin(angle,M, d, v, F0)
        P[i] = np.abs(A.conj().T @ A0).item() ** 2

    return P


A0 = alin(0, M, d, v, F0)
P = np.zeros(len(theta))
Spectrum = spatialSpectrum(P, A0, theta, M, d, v, F0)

#plt.plot(theta, P)
#plt.xlabel("angle [deg]")
#plt.ylabel("spatial spectrum")
#plt.show()

#        --part 2 two sources --

def giveRx(SNR, theta0, M, d, v, F0):
    
    sigma_n = 10**(-SNR/20)
    
    A = alin(theta0, M, d, v, F0)
    A_H = A.conj().T
    R = A @ A_H                            
    Rn = np.eye(M,M)*sigma_n**2 
    Rx = R + Rn  

    return Rx

def matchedBeamFormer(pmbf, Rx, theta, M, d, v, F0):

    for i, angle in enumerate(theta):
        a = alin(angle, M, d, v, F0)
        pmbf[i] = np.real(a.conj().T @ Rx @ a)

    return pmbf


def MVDR(pmvdr, Rx, theta, M, d, v, F0):
    Rxinv = np.linalg.solve(Rx,np.eye(M))
    for i, angle in enumerate(theta):
        a = alin(angle, M, d, v, F0)
        
        pmvdr[i] = np.real(1/(a.conj().T @ Rxinv @ a))
    
    return pmvdr

def sourceDirection(Pmbf, theta, Q):
    peaks, _ = find_peaks(Pmbf)

    sourceIndices = peaks[np.argsort(Pmbf[peaks])[-Q:]] 
    return  sourceIndices


# calculating Rx

theta0 = np.deg2rad([0,15]) 
Rxx = giveRx(snr, theta0, M, d, v, F0)

# calculating pmbf

pmbf = np.zeros(len(theta))
#Pmbf  = matchedBeamFormer(pmbf, Rxx, theta, M, d, v, F0)

# calculating pmvdr

pmvdr = np.zeros(len(theta))
#Pmvdr = MVDR(pmvdr, Rxx, theta, M, d, v, F0)

# finding argument theta

'''
sourcePointsMbf = sourceDirection(Pmbf, theta, Q)
thetaDirMbf = np.sort(np.rad2deg(theta[sourcePointsMbf]))
print( thetaDirMbf)

# finding argument theta

sourcePointsMvdr = sourceDirection(Pmvdr, theta, Q)
thetaDirMvdr = np.sort(np.rad2deg(theta[sourcePointsMvdr]))
print( thetaDirMvdr)

plt.figure(figsize=(14, 4))

# bmf plot

plt.subplot(1, 2, 1)
plt.plot(np.rad2deg(theta), Pmbf)
plt.title("(a) matched beamformer")
plt.xlabel("angle [deg]")
plt.ylabel("spatial spectrum")
plt.grid()

# mvdr plot
plt.subplot(1, 2, 2)
plt.plot(np.rad2deg(theta), Pmvdr)
plt.title("mvdr")
plt.xlabel("angle [deg]")
plt.ylabel("spatial spectrum")
plt.grid()

plt.tight_layout()
plt.show()
'''



base = Path(__file__).resolve().parent

folder = base / "recordings" / "2026-11-24_17-43-45_single_source"

files = [
    folder / f"recording_2026-11-24_17-43-45_channel_{i}.wav"
    for i in range(1, 7)
]

X = []

for f in files:
    Fs, x = wavfile.read(f)
    X.append(x)

X = np.array(X, dtype=float)

def narrowband_Rx(X, Fs, nperseg):

    SFT = ShortTimeFFT.from_window('hann', Fs, nperseg, noverlap=nperseg//2)
    Sx_all = SFT.stft(X)
    f_bins = SFT.f 
    M = Sx_all.shape[0]
    K = Sx_all.shape[1]                 
    L = Sx_all.shape[2]
      
    print(Sx_all.shape)  
    print(M,K,L)     
    
            
    Rx_all = np.zeros((K, M, M), dtype=complex)
    for k in range(K):
        Xk = Sx_all[:,k,:]    
        Rx_all[k] = Xk @ Xk.conj().T/L

    return f_bins, Rx_all

f_bins, Rx_all = narrowband_Rx(X, Fs, Nw)

k = np.argmin(np.abs(f_bins - 2000))
Fk = f_bins[k]


Pmbf  = matchedBeamFormer(pmbf, Rx_all[k], theta, M, d, v, Fk)
Pmvdr = MVDR(pmvdr, Rx_all[k], theta, M, d, v, Fk)

print("bin", round(Fk), "Hz -> mbf:", np.rad2deg(theta[np.argmax(Pmbf)]),
      " mvdr:", np.rad2deg(theta[np.argmax(Pmvdr)]))

plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.plot(np.rad2deg(theta), Pmbf)
plt.title("matched beamformer, " + str(round(Fk)) + " Hz")
plt.xlabel("angle [deg]")
plt.ylabel("spatial spectrum")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(np.rad2deg(theta), Pmvdr)
plt.title("mvdr, " + str(round(Fk)) + " Hz")
plt.xlabel("angle [deg]")
plt.ylabel("spatial spectrum")
plt.grid()
plt.tight_layout()
plt.show()


def bandAverage(Rx_all, f_bins, F1, F2, method):

    P = np.zeros(len(theta))
    count = 0

    for k in range(len(f_bins)):
        if f_bins[k] >= F1 and f_bins[k] <= F2:
            p = np.zeros(len(theta))
            if method == "mbf":
                p = matchedBeamFormer(p, Rx_all[k], theta, M, d, v, f_bins[k])
            else:
                p = MVDR(p, Rx_all[k], theta, M, d, v, f_bins[k])
            P = P + p
            count = count + 1

    return P / count

F1 = 300
F2 = 1600       # below v/(2d) = 1715 Hz, so no spatial aliasing

Pmbf_band = bandAverage(Rx_all, f_bins, F1, F2, "mbf")
Pmvdr_band = bandAverage(Rx_all, f_bins, F1, F2, "mvdr")

print("band", F1, "-", F2, "Hz -> mbf:", np.rad2deg(theta[np.argmax(Pmbf_band)]),
      " mvdr:", np.rad2deg(theta[np.argmax(Pmvdr_band)]))

plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.plot(np.rad2deg(theta), Pmbf_band)
plt.title("matched beamformer, band " + str(F1) + "-" + str(F2) + " Hz")
plt.xlabel("angle [deg]")
plt.ylabel("spatial spectrum")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(np.rad2deg(theta), Pmvdr_band)
plt.title("mvdr, band " + str(F1) + "-" + str(F2) + " Hz")
plt.xlabel("angle [deg]")
plt.ylabel("spatial spectrum")
plt.grid()
plt.tight_layout()
plt.show()

