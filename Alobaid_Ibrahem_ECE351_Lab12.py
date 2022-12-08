import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sig
import scipy.fftpack as sfft
import control as con 
import pandas as pd 

# ==================== load input signal ====================
df = pd.read_csv ('NoisySignal.csv') 

t = df ['0'].values
sensor_sig = df ['1'].values

plt.figure(figsize = (10, 7))
plt.plot (t, sensor_sig ) 
plt.grid ()
plt.title ('Noisy Input Signal')
plt.xlabel ('Time [s]') 
plt.ylabel ('Amplitude [V]') 
plt.show ()

# ==================== .stem() Workaround ====================
def make_stem(ax, x, y, color = 'k', style = 'solid', label ='', linewidths = 2.5,** kwargs):
    
    ax.axhline(x[0], x[-1], 0, color = 'r')
    ax.vlines(x, 0, y, color = color, linestyles = style, label = label, linewidths = linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])
    
# ==================== Fast-fourier transform ====================
def FFT(x, fs):
    
    N = len(x) 
    X_fft = sfft.fft(x) 
    X_fft_shifted = sfft.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2) * fs/N 

    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    for i in range(len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0 

    return freq, X_mag, X_phi

# ==================== Plots of noisy signal ====================
fs = 1e6
freq, X_mag, X_phi  = FFT(sensor_sig, fs)

fig, ax = plt.subplots(figsize=(10, 7))
make_stem(ax, freq, X_mag)
plt.grid ()
plt.title("Fast Fourier Transform of Noisy Signal")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')
plt.show ()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_mag)
plt.grid ()
plt.xlim(1800, 2000)
plt.title("($1.8kHz \leq f \leq 2.0kHz$)")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')

fig, ax = plt.subplots(figsize=(10, 7))
make_stem(ax, freq, X_mag)
plt.grid ()
plt.xlim(0, 1800)
plt.title("($0Hz \leq f \leq 1.8kHz$)")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')
plt.xlabel('f(Hz)')

fig, ax = plt.subplots(figsize=(10, 7))
make_stem(ax, freq, X_mag)
plt.grid ()
plt.xlim(2000, 100000)
plt.title("($2.0kHz \leq f \leq 100kHz$)")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')
plt.xlabel('f(Hz)')

# ==================== Filter characteristics ====================
steps = 1
W = np. arange(1e3, 1e6 + steps, steps ) 

L = 1
fc1 = 1.8e3
fc2 =  2e3

Wc1 = 2 * np.pi * fc1
Wc2 = 2 * np.pi * fc2
W0 = np.sqrt(Wc1 * Wc2)
B = Wc2 - Wc1

R = B * L
C = 1/(L * W0**2)

num = [0, (R/L), 0]
den = [1, (R/L), 1/(L*C)]

print("wc1 = ", Wc1, ", wc2 = ", Wc2)
print("w0 = ", W0)
print("R = ", R, ", L = ", L, ", C = ", C)


plt.figure(figsize = (10, 7))
sys = con.TransferFunction(num, den)
_ = con.bode(sys, W, dB = True, Hz = True, deg = True, Plot = True)

plt.figure(figsize = (10, 7))
sys = con.TransferFunction(num, den)
_ = con.bode(sys, W, dB = True, Hz = True, deg = True, Plot = True)
plt.xlim(1800, 2000)

plt.figure(figsize = (10, 7))
sys = con.TransferFunction(num, den)
_ = con.bode(sys, W, dB = True, Hz = True, deg = True, Plot = True)
plt.xlim(0, 1800)

plt.figure(figsize = (10, 7))
sys = con.TransferFunction(num, den)
_ = con.bode(sys, W, dB = True, Hz = True, deg = True, Plot = True)
plt.xlim(2000, 100000)

# ==================== Filtered signal ====================

numZ, denZ = sig.bilinear(num, den, fs) 

y = sig.lfilter(numZ, denZ, sensor_sig)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.title("Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [V]")
plt.show()

# ==================== Plots of filtered signal ====================
freq, X_mag, X_phi  = FFT(y, fs)

fig, ax = plt.subplots(figsize=(10, 7))
plt.grid ()
make_stem(ax, freq, X_mag)
plt.title("Fast Fourier Transform of Filtered Signal")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')
plt.show ()

fig, ax = plt.subplots(figsize =(10, 7))
make_stem(ax, freq, X_mag)
plt.grid ()
plt.xlim(1800, 2000)
plt.title("($1.8kHz \leq f \leq 2.0kHz$)")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')

fig, ax = plt.subplots(figsize=(10, 7))
make_stem(ax, freq, X_mag)
plt.grid ()
plt.xlim(0, 1800)
plt.title("($0Hz \leq f \leq 1.8kHz$)")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')
plt.xlabel('f(Hz)')

fig, ax = plt.subplots(figsize=(10, 7))
make_stem(ax, freq, X_mag)
plt.grid ()
plt.xlim(2000, 100000)
plt.title("($2.0kHz \leq f \leq 100kHz$)")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')
plt.xlabel('f(Hz)')

