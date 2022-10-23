import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math


# ==================== Part 1 - Task 1 ====================

a = np.zeros((4, 1))
b = np.zeros((4, 1))

for k in np.arange(1, 4):
    b[k] = 2/(k*np.pi) * (1-np.cos(k*np.pi))

for k in np.arange(1, 4):
    a[k] = 0
    
print("a_0 = ", a[0], "\na_1 = ", a[1])
print("b_1 = ", b[1], "\nb_2 = ", b[2], "\nb_3 = ", b[3])

# ==================== Part 1 - Task 2 ====================
T = 8
w = 2*np.pi/T
steps = 1e-3
t=np.arange(0 , 20 + steps , steps)

N = [1, 3, 15, 50, 150, 1500]
y = [0, 0, 0, 0, 0, 0]

for i in range(len(N)):
    for k in range(N[i]):
        b = 2/((k+1)*np.pi) * (1-np.cos((k+1)*np.pi))
        x = b * np.sin((k+1)*w*t)
        
        y[i] += x
        

plt.figure(figsize = (10 ,7))
plt.subplot(3, 1, 1)
plt.plot(t, y[0])
plt.grid()
plt.ylabel('N = 1')
plt.title('Figure 1: Fourier Series Approximations of x(t) for N ={1, 3, 15}')

plt.subplot(3, 1, 2)
plt.plot(t, y[1])
plt.grid()
plt.ylabel('N = 3')
plt.xlabel('t')

plt.subplot(3, 1, 3)
plt.plot(t, y[2])
plt.grid()
plt.ylabel('N = 15')
plt.xlabel('t')

plt.figure(figsize = (10 ,7))
plt.subplot(3, 1, 1)
plt.plot(t, y[3])
plt.grid()
plt.ylabel('N = 50')
plt.title('Figure 2: Fourier Series Approximations of x(t) for N ={50, 150, 1500}')

plt.subplot(3, 1, 2)
plt.plot(t, y[4])
plt.grid()
plt.ylabel('N = 150')
plt.xlabel('t')

plt.subplot(3, 1, 3)
plt.plot(t, y[5])
plt.grid()
plt.ylabel('N = 1500')
plt.xlabel('t')
    

    

