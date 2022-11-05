import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as sfft

def FFT(x, fs):
    N = len(x)
    X_fft = sfft.fft(x)
    X_fft_shifted = sfft.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2) * fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    return freq, X_mag, X_phi

# ==================== Part 1 - Task 1 ====================
fs = 100
T = 1/fs
t = np.arange(0 , 2, T)

x = np.cos(2*np.pi*t)
freq, mag, phi = FFT(x, fs)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t(s)")
plt.title("Task 1 - $x(t) = cos(2 \pi t)$")

plt.subplot(3, 2, 3)
plt.stem(freq, mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, phi) 
plt.grid()
plt.ylabel('Phase (deg)')
plt.xlabel('f(Hz)')

plt.subplot(3, 2, 6)
plt.stem(freq, phi)
plt.grid()
plt.xlim(-3, 3)
plt.xlabel('f(Hz)')

# ==================== Part 1 - Task 2 ====================
x = 5 * np.sin(2*np.pi*t)
freq, mag, phi = FFT(x, fs)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t(s)")
plt.title("Task 2 - $x(t) = 5sin(2 \pi t)$")

plt.subplot(3, 2, 3)
plt.stem(freq, mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, phi) 
plt.grid()
plt.ylabel('Phase (deg)')
plt.xlabel('f(Hz)')

plt.subplot(3, 2, 6)
plt.stem(freq, phi)
plt.grid()
plt.xlim(-3, 3)
plt.xlabel('f(Hz)')

# ==================== Part 1 - Task 3 ====================
x = 2 * np.cos((2*np.pi*2*t) - 2) + (np.sin((2*np.pi*6*t) + 3))**2
freq, mag, phi = FFT(x, fs)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t(s)")
plt.title("Task 3 - $x(t) = 2cos((2 \pi \cdot 2t)-2) + sin^2((2 \pi \cdot 6t)+3)$")

plt.subplot(3, 2, 3)
plt.stem(freq, mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, mag)
plt.grid()
plt.xlim(-20, 20)

plt.subplot(3, 2, 5)
plt.stem(freq, phi) 
plt.grid()
plt.ylabel('Phase (deg)')
plt.xlabel('f(Hz)')

plt.subplot(3, 2, 6)
plt.stem(freq, phi)
plt.grid()
plt.xlim(-20, 20)
plt.xlabel('f(Hz)')

# ==================== Part 1 - Task 4 ====================
def newFFT(x, fs):
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

# ==================== Part 1 - Task 4(a) ====================
x = np.cos(2*np.pi*t)
freq, mag, phi = newFFT(x, fs)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t(s)")
plt.title("Task 4(a) - $x(t) = cos(2 \pi t)$")

plt.subplot(3, 2, 3)
plt.stem(freq, mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, phi) 
plt.grid()
plt.ylabel('Phase (deg)')
plt.xlabel('f(Hz)')

plt.subplot(3, 2, 6)
plt.stem(freq, phi)
plt.grid()
plt.xlim(-3, 3)
plt.xlabel('f(Hz)')

# ==================== Part 1 - Task 4(b) ====================
x = 5 * np.sin(2*np.pi*t)
freq, mag, phi = newFFT(x, fs)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t(s)")
plt.title("Task 4(b) - $x(t) = 5sin(2 \pi t)$")

plt.subplot(3, 2, 3)
plt.stem(freq, mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, phi) 
plt.grid()
plt.ylabel('Phase (deg)')
plt.xlabel('f(Hz)')

plt.subplot(3, 2, 6)
plt.stem(freq, phi)
plt.grid()
plt.xlim(-3, 3)
plt.xlabel('f(Hz)')

# ==================== Part 1 - Task 4(b) ====================
x = 2 * np.cos((2*np.pi*2*t) - 2) + (np.sin((2*np.pi*6*t) + 3))**2
freq, mag, phi = newFFT(x, fs)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t(s)")
plt.title("Task 4(c) - $x(t) = 2cos((2 \pi \cdot 2t)-2) + sin^2((2 \pi \cdot 6t)+3)$")

plt.subplot(3, 2, 3)
plt.stem(freq, mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, mag)
plt.grid()
plt.xlim(-20, 20)

plt.subplot(3, 2, 5)
plt.stem(freq, phi) 
plt.grid()
plt.ylabel('Phase (deg)')
plt.xlabel('f(Hz)')

plt.subplot(3, 2, 6)
plt.stem(freq, phi)
plt.grid()
plt.xlim(-20, 20)
plt.xlabel('f(Hz)')

# ==================== Part 1 - Task 5 ====================
T1 = 8
w = 2*np.pi/T1
steps = 1e-3
t=np.arange(0, 16, T)

def FFTs(T,t):
    y = 0
    for k in range(15):
        b = 2/((k+1)*np.pi) * (1-np.cos((k+1)*np.pi))
        x = b * np.sin((k+1)*w*t)
        
        y += x
    return y

x = FFTs(T1, t)
freq, mag, phi = newFFT(x, fs)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t(s)")
plt.title("Task 5 - Fourier series approximation of Lab8 using (N=15)")

plt.subplot(3, 2, 3)
plt.stem(freq, mag)
plt.grid()
plt.xlim(-2, 2)
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, mag)
plt.grid()
plt.xlim(-2, 2)

plt.subplot(3, 2, 5)
plt.stem(freq, phi) 
plt.grid()
plt.xlim(-2, 2)
plt.ylabel('Phase (deg)')
plt.xlabel('f(Hz)')

plt.subplot(3, 2, 6)
plt.stem(freq, phi)
plt.grid()
plt.xlim(-2, 2)
plt.xlabel('f(Hz)')