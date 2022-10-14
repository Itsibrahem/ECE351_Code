import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math

# ==================== part 1 - Task 2 ====================
numG, denG = [1, 9], [1, -2, -40, -64]

Gz, Gp, Gk = sig.tf2zpk(numG, denG)                  
print("G(s):", "\nzeros: ", Gz, "\npoles: ", Gp)

numA, denA = [1, 4], [1, 4, 3]

Az, Ap, Ak = sig.tf2zpk(numA, denA)
print("\nA(s):", "\nzeros: ", Az, "\npoles: ", Ap)

numB = [1, 26, 168]

Bz = np.roots(numB)
print("\nB(s):", "\nzeros: ", Bz)


# ==================== part 1 - Task 5 ====================
numO = sig.convolve(numG, numA)
denO = sig.convolve(denG, denA)

print("\nOpen-loop transfer function:")
print("Numerator: ", numO, "\nDenominator: ", denO)

steps = 1e-2

t = np.arange(0 , 2+steps , steps)
tout, yout = sig.step((numO, denO), T = t)

plt.figure(figsize = (10 ,7))
plt.subplot(1, 1, 1)
plt.plot(tout , yout)
plt.grid()
plt.ylabel('response')
plt.xlabel('t')
plt.title('Figure 1: open-loop step response')

# ==================== part 2 - Task 2 ====================
numC = sig.convolve(numA, numG)
denc0 = sig.convolve(denA, denG)
denc1 = sig.convolve(denA, numG)
denc2 = sig.convolve(denc1, numB)

denC = denc0 + denc2

print("\nClosed-loop transfer function:")
print("Numerator: ", numC, "\nDenominator: ", denC)

Cz, Cp, Ck = sig.tf2zpk(numC, denC)
print("\nnzeros: ", Cz, "\npoles: ", Cp)


# ==================== part 2 - Task 4 ====================

t = np.arange(0 , 10+steps , steps)
tout, yout = sig.step((numC, denC), T = t)

plt.figure(figsize = (10 ,7))
plt.subplot(1, 1, 1)
plt.plot(tout , yout)
plt.grid()
plt.ylabel('response')
plt.xlabel('t')
plt.title('Figure 2: closed-loop step response')





