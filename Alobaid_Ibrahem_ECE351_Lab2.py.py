import numpy as np
import matplotlib.pyplot as plt
import math 

plt.rcParams['font.size'] = 14 

steps = 1e-2
t = np.arange(-5, 10+steps, steps) # changes with plots

def r(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y
# =============================================================================
def step(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1
    return y
# =============================================================================

def func1(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
            y[i] = np.cos(t[i])
    return y
# =============================================================================

def func(t):
    return r(t) - r(t-3) + 5*step(t-3) - 2*step(t-6) - 2 * r(t-6)

#y = r(t)
# =============================================================================
y = func(t) # changes for each operation

dy = np.diff(func(t))

# =============================================================================
#x = np.arange(-5, 10, steps)

plt.figure(figsize = (10, 7))
plt.plot(t, dy)
plt.legend()
plt.grid()

plt.ylabel('dy(t)')
plt.xlabel('t')
plt.title('Derivative of the user-defined function')

plt.show()