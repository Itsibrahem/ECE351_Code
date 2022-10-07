import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math
import sympy

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1
    return y
 
# ==================== part 1 - Task 1 ====================
def y(t):
    y = (np.exp(-6*t) - 0.5*np.exp(-4*t) + 0.5)*u(t)
    return y

steps = 1e-3

t = np.arange(0 , 2+steps , steps)

plt.figure(figsize = (10 ,7))
plt.subplot(2, 1, 1)
plt.plot(t , y(t))
plt.grid()
plt.ylabel('user-defined function')
plt.title('Figure 1: step response')

# ==================== part 1 - Task 2 ====================
numH = [1, 6, 12]
denH = [1, 10, 24]

tout, yout = sig.step((numH, denH), T = t)

plt.subplot(2, 1, 2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('scipy.signal.step()')
plt.xlabel('t')

# ==================== part 1 - Task 3 ====================
denH.append(0)
R, P, K = sig.residue(numH, denH)

print('Part1:', '\nresidue: ', R, '\npoles: ', P, '\ngain: ', K)

# ==================== part 2 - Task 1 ====================
numY = [25250]
denY = [1, 18, 218, 2036, 9085, 25250, 0]

R, P, K = sig.residue(numY, denY)

print('\nPart2:', '\nresidue: ', R, '\npoles: ', P, '\ngain: ', K)

# ==================== part 2 - Task 2 ====================
t = np.arange(0 , 4.5+steps , steps)

def cosine(R, P, K):
    y = 0
    
    for i in range(len(R)):
        R_mag = np.abs(R[i])
        R_ang = np.angle(R[i])
        alpha = np.real(P[i])
        omega = np.imag(P[i])
        
        y += (R_mag*np.exp(alpha*t) * np.cos(omega*t + R_ang) * u(t))
        
    return y

plt.figure(figsize = (10 ,7))
plt.subplot(2, 1, 1)
plt.plot(t , cosine(R, P, K))
plt.grid()
plt.ylabel('cosine method')
plt.title('Figure 2: step response')

# ==================== part 2 - Task 3 ====================
denY.pop()

tout, yout = sig.step((numY, denY), T = t)

plt.subplot(2, 1, 2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('scipy.signal.step()')
plt.xlabel('t')




