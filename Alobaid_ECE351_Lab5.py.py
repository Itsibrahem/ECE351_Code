import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math

steps = 1e-7

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1
    return y
 
# ==================== part 1 - Task 1 ====================

R = 1000
L = 27e-3
C = 100e-9

t = np.arange(0 , 1.2e-3 + steps , steps)

def h(R, L, C, t):
    
    w = (1/2) * np.sqrt((1/(R*C))**2 -4*(1/(np.sqrt(L*C)))**2 + 0j) * u(t)
    alpha = -1/(2*R*C)
    
    p = alpha + w
    
    g = 1/(R*C) * p
    g_abs = np.abs(g)
    g_ang = np.angle(g)

    y = (g_abs/np.abs(w)) * np.exp(alpha * t) * np.sin(np.abs(w) * t + g_ang)

    return y

# ==================== part 1 - Task 2 ====================

num = [0, 1/(R*C), 0]
den = [1, 1/(R*C), 1/(C*L)]

tout, yout = sig.impulse((num, den), T = t)

plt.figure(figsize = (10 ,7))
plt.subplot(2, 1, 1)
plt.plot(t , h(R,L,C,t))
plt.grid()
plt.ylabel('hand-solved')
plt.title('Figure 1: impulse response')

plt.subplot(2, 1, 2)
plt.plot(tout , yout)
plt.grid()
plt.ylabel('using scipy.signal.impulse()')
plt.xlabel ('t')

# ======================== part 2 =========================

tout, yout = sig.step((num, den), T = t)

plt.figure(figsize = (10 ,7))
plt.plot(tout , yout)
plt.grid()
plt.title('Figure 2: step response')
plt.ylabel('using scipy.signal.step()')
plt.xlabel ('t')

plt.show()