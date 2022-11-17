import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import patches    
    
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()

    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    return z, p, k
# ==================== Part 1 - Task 3 ====================
Num = [2, -40]
Den = [1, -10, 16]

R, P, K = sig.residuez(Num, Den)

print("H(z): using scipy.signal.residuez() function")
print("Residue = ", R, "\nPoles = ", P, "\nK = ", K)

# ==================== Part 1 - Task 4 ====================
NumZ = [2, -40]
DenZ = [1, -10, 16]

Z, P, K = zplane(NumZ, DenZ)

print("\nH(z): using zplane() function")
print("Zeros = ", Z, "\nPoles = ", P, "\nK = ", K)

# ==================== Part 1 - Task 5 ====================
w, h = sig.freqz(NumZ, DenZ, whole = True)

magDB = 20*np.log10(np.abs(h))
phase = np.angle(h)*180/np.pi
f = w/(2*np.pi)

plt.figure(figsize = (10, 7))

plt.subplot(2, 1, 1)
plt.plot(f, magDB)
plt.grid()
plt.ylabel("Mag (dB)")
plt.title("Magnitude & Phase Responses of H(z)")

plt.subplot(2, 1, 2)
plt.plot(f, phase)
plt.grid()
plt.ylabel("Phase (deg)")
plt.xlabel("f(Hz)")


