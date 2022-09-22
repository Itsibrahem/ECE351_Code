import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams['font.size'] = 14 

steps = 1e-2

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1
    return y
 
# ==================== part 1 - Task 1 ====================

t = np.arange(-10, 10 + steps , steps)

f = 0.25
w = 2 * math.pi * f

def h1(t):
    return (np.exp(-2*t)) * (u(t) - u(t-3))

def h2(t):
    return u(t-2) - u(t-6)

def h3(t):
    return np.cos(w*t) * u(t)

# ==================== part 1 - Task 2 ====================

plt.figure(figsize = (10, 7))
plt.subplot(3 , 1 , 1)
plt.plot(t , h1(t))
plt.grid()
plt.ylabel('h1(t)')
plt.title('Figure 1')
 
plt.subplot(3 , 1 , 2)
plt.plot(t , h2(t))
plt.grid()
plt.ylabel('h2(t)')
 
plt.subplot(3 , 1 , 3)
plt.plot(t , h3(t))
plt.grid()
plt.ylabel('h3(t)')
plt.xlabel('t')

# ========================- part 2 -========================
def my_conv(h1, h2):
    
    Nh1 = len(h1)
    Nh2 = len(h2)
     
    h1Extended = np.append(h1, np.zeros((1, Nh2-1))) 
    h2Extended = np.append(h2, np.zeros((1, Nh1-1)))
     
    result = np.zeros(h1Extended.shape)
     
    for i in range((Nh2 + Nh1)-2): 
        result[i] = 0  
          
        for j in range(Nh1): 
            if ((i-j)+1 > 0): 
                try: 
                    result[i] += h1Extended[j] * h2Extended[i-j+1]
                except:
                    print(i-j)                 
    return result

y1 = my_conv(h1(t), u(t)) 
y2 = my_conv(h2(t), u(t)) 
y3 = my_conv(h3(t), u(t))

tExtended = np.arange(2*t[0], 2*t[len(t)-1]+steps, steps)

plt.figure(figsize = (10, 7))
plt.subplot(3 , 1 , 1)
plt.plot(tExtended, y1)
plt.grid()
plt.ylabel('Responce1')
plt.title('Figure 2')

plt.subplot(3 , 1 , 2)
plt.plot(tExtended, y2)
plt.grid()
plt.ylabel('Responce2')

plt.subplot(3 , 1 , 3)
plt.plot(tExtended, y3)
plt.grid()
plt.ylabel('Responce3')
plt.xlabel('t')

# ======================= part 2(c) ========================

y1c =  0.5*(1-np.exp(-2*tExtended))*u(tExtended)-0.5*(1-np.exp(-2*(tExtended-3)))*u(tExtended-3)
y2c = ((tExtended-2) * u(tExtended-2)) - ((tExtended-6) * u(tExtended-6))
y3c = (1 / w) * (np.sin(w*tExtended) * u(tExtended))

plt.figure( figsize = (10 , 7))
plt.subplot(3, 1, 1)
plt.plot(tExtended, y1c)
plt.grid()
plt.ylabel('Responce1')
plt.title('Step Respnoce (Hand Calculated)')

plt.subplot(3, 1, 2)
plt.plot(tExtended, y2c)
plt.grid()
plt.ylabel('Responce2')

plt.subplot(3, 1, 3)
plt.plot(tExtended, y3c)
plt.grid()
plt.ylabel('Responce3')
plt.xlabel('t')

plt.show()


