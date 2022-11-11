import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as sfft
import control as con 

# ==================== Part 1 - Task 1 ====================
steps = 1
W = np. arange (1e3, 1e6 + steps, steps ) 

R = 1000
L = 27e-3
C = 100e-9

phase = 90-np.arctan((W/(R*C))/(1/(L*C)-(W**2)))*(180/np.pi)
for i in range(len(phase)):
    if (phase[i] > 90):
        phase[i] = phase[i] - 180

mag = (W/(R*C))/(np.sqrt((1/(L*C)-(W**2))**2+(W/(R*C))**2))
magDB = 20*np.log10(mag)

plt.figure(figsize = (10, 7))

plt.subplot(2, 1, 1)
plt.semilogx(W, magDB)
plt.grid()
plt.ylabel("Mag (dB)")
plt.title("H(jw) user-defined")

plt.subplot(2, 1, 2)
plt.semilogx(W, phase)
plt.grid()
plt.ylabel("Phase (deg)")
plt.xlabel("w (rad/s)")

# ==================== Part 1 - Task 2 ====================
num = [0, 1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

W1, mag, phase = sig.bode((num,den), W)

plt.figure(figsize = (10, 7))

plt.subplot(2, 1, 1)
plt.semilogx(W1, mag)
plt.grid()
plt.ylabel("Mag (dB)")
plt.title("H(jw) using scipy.signal.bode()")

plt.subplot(2, 1, 2)
plt.semilogx(W1, phase)
plt.grid()
plt.ylabel("Phase (deg)")
plt.xlabel("w (rad/s)")

# ==================== Part 1 - Task 3 ====================
f = W/(2*np.pi)

plt.figure(figsize = (10, 7))

sys = con.TransferFunction(num, den)
_ = con.bode(sys, W, dB = True, Hz = True, deg = True, Plot = True)

# ==================== Part 2 - Task 1 ====================
fs = (2*50000*np.pi)
steps = 1 / fs
t = np.arange(0, 0.01+steps, steps)

x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure(figsize = (10, 7))
plt.plot(t, x)
plt.grid()
plt.ylabel("x(t)")
plt.xlabel("t(s)")
plt.title("Input Signal")

# ==================== Part 2 - Task 2 ====================
numZ, denZ = sig.bilinear(num, den, fs)

# ==================== Part 2 - Task 3 ====================
y = sig.lfilter(numZ, denZ, x)

# ==================== Part 2 - Task 4 ====================
plt.figure(figsize = (10, 7))

plt.plot(t, y)
plt.grid()
plt.ylabel("y(t)")
plt.xlabel("t(s)")
plt.title("Output Signal")

plt.show()

