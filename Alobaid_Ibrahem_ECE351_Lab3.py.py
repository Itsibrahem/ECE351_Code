import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.rcParams['font.size'] = 14 

steps = 1e-2

def r(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y
# =============================================================================
def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1
    return y
# =============================================================================

 

# ==================== part 1 - Task 1 ====================
t = np.arange(0, 20 + steps , steps)

def f1(t):
    return u(t-2) - u(t-9)

def f2(t):
    return (np.exp(-t))

def f3(t):
    return r(t-2) * (u(t-2) - u(t-3)) + r(4-t) * (u(t-3) - u(t-4))

# ==================== part 1 - Task 2 ====================

plt.figure(figsize = (10, 7))
plt.subplot(3 , 1 , 1)
plt.plot(t , f1(t))
plt.grid()
plt.ylabel('f1(t)')
plt.title('Figure 1')
 
 
plt.subplot(3 , 1 , 2)
plt.plot(t , f2(t))
plt.grid()
plt.ylabel('f2(t)')
 
plt.subplot(3 , 1 , 3)
plt.plot(t , f3(t))
plt.grid()
plt.ylabel('f3(t)')
plt.xlabel('t')
 
# ==================== part 2 - Task 1 ====================

def my_conv(f1, f2):
    
    Nf1 = len(f1) # create an array of length f1
    Nf2 = len(f2)
     
    f1Extended = np.append(f1, np.zeros((1, Nf2-1))) # attach zeros to the end
                                                     # of list of size Nf2
    f2Extended = np.append(f2, np.zeros((1, Nf1-1)))
     
    result = np.zeros(f1Extended.shape)
     
    for i in range((Nf2 + Nf1)-2): # iterate until size of (Nf2+Nf1)
        result[i] = 0 # initialize list 
          
        for j in range(Nf1): 
            if ((i-j)+1 > 0): 
                try: 
                    result[i] += f1Extended[j] * f2Extended[i-j+1]
                except:
                    print(i-j)
# The nested for loop should perform a summition of the productes of 
# f1 and f2 extended list and store the result in the "result" array.                  
    return result
    
# ==================== part 2 - Task 2 ====================
 
Nt = len(t)
tExtended = np.arange(0, 2*t[Nt-1], steps)         
 
convf1_f2 = my_conv(f1(t), f2(t))
 
 
plt.figure(figsize = (10, 7))
plt.subplot(3 , 1 , 1)
plt.plot(tExtended , convf1_f2)
plt.grid()
plt.ylabel('convf1_f2')
plt.title('Figure 2')
 
# ==================== part 2 - Task 3 ====================
 
convf2_f3 = my_conv(f2(t), f3(t))
 
plt.subplot(3 , 1 , 2)
plt.plot(tExtended , convf2_f3)
plt.grid()
plt.ylabel('convf2_f3')
 
# ==================== part 2 - Task 4 ====================
 
convf1_f3 = my_conv(f1(t), f3(t))
 
plt.subplot(3 , 1 , 3)
plt.plot(tExtended , convf1_f3)
plt.grid()
plt.ylabel('convf1_f3')
plt.xlabel('t')

# ==================== part 2b ====================
 
plt.figure(figsize = (10, 7))
plt.subplot(3 , 1 , 1)
plt.plot(tExtended , sig.convolve(f1(t), f2(t)))
plt.grid()
plt.ylabel('v_convf1_f2')
plt.title('Figure 2')
 
# ==================== part 2b ====================
 
plt.subplot(3 , 1 , 2)
plt.plot(tExtended , sig.convolve(f2(t), f3(t)))
plt.grid()
plt.ylabel('v_convf2_f3')
 
# ==================== part 2b ====================

plt.subplot(3 , 1 , 3)
plt.plot(tExtended , sig.convolve(f1(t), f3(t)))
plt.grid()
plt.ylabel('v_convf1_f3')
plt.xlabel('t')

plt.show()