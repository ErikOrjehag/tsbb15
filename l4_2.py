import cvl_labs.lab4 as lab4
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.signal import convolve2d as conv2
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


Ig = None
gI_norm = None
T_norm = None
HL_norm = None

def image_gradient(I, ksize=3, sigma=3):


    lp = np.atleast_2d(np.exp(-0.5*(np.arange(-ksize,ksize+1)/sigma)**2))
    lp = lp / np.sum(lp)
    df = np.atleast_2d(-1.0/np.square(sigma) * np.arange(-ksize,ksize+1) * lp)
    global Ig
    #Ig = conv2(conv2(I, lp, mode="same"), lp.T, mode="same")

    #dx = conv2(Ig, df, mode="same")
    #dy = conv2(Ig, df.T, mode="same")

    dx = conv2(conv2(I, df, mode="same"), lp.T , mode="same")
    dy = conv2(conv2(I, lp, mode="same"), df.T , mode="same")

    #df = np.atleast_2d(np.array([1/2, 0, -1/2]))

    #dx = conv2(I, df, mode="same")
    #dy = conv2(I, df.T, mode="same")

    gI = np.array([dx, dy])
    global gI_norm
    gI_norm = np.linalg.norm(gI, axis=0)

    T = np.empty(np.shape(I) + (2, 2))
    T[...,0,0] = dx**2
    T[...,0,1] = dx*dy
    T[...,1,0] = T[...,0,1]
    T[...,1,1] = dy**2

    global T_norm
    T_norm = np.linalg.norm(T, axis=(-1,-2))

    h11 = np.array([
        [0, 0, 0],
        [1, -2, 1],
        [0, 0, 0]
    ])
    h12 = np.array([
        [1/2, 0, -1/2],
        [0, 0, 0],
        [-1/2, 0, 1/2]
    ])
    h22 = h11.T
    HL = np.empty(np.shape(I) + (2, 2))
    HL[...,0,0] = conv2(I, h11, mode="same")
    HL[...,0,1] = conv2(I, h12, mode="same")
    HL[...,1,0] = HL[...,0,1]
    HL[...,1,1] = conv2(I, h22, mode="same")

    global HL_norm
    HL_norm = np.linalg.norm(HL, axis=(-1,-2))

    return gI, T, HL

a = None
b = None
step = None

def update_u(u, g, Xor, alpha, lam):

    gI, T, HL = image_gradient(u)

    inner1 = (HL[...,0,0] * T[...,1,1] -
            2 * HL[...,0,1] * T[...,0,1] +
            HL[...,1,1] * T[...,0,0])
    inner2 = 1e-10 + np.linalg.norm(gI, axis=0) ** 3

    inner = inner1 / inner2

    global a, b, step
    a = alpha * Xor * (u - g)
    b = alpha * lam*inner
    step = - (a - b)

    return u + step

# Green-black-red color map for gradients
gkr_col = np.zeros((255, 4))
gkr_col[:,3] = 1
gkr_col[:127,1] = np.linspace(1.0, 0.0, 127, False)
gkr_col[127:,0] = np.linspace(0.0, 1.0, 128, True)
gkr_col = ListedColormap(gkr_col)

#img = lab4.make_circle(256/2, 256/2, 256/3)
img = lab4.get_cameraman().astype(float) / 255

#Xor = np.ones_like(img)
#g = img * Xor

#Xor = np.random.randint(2, size=img.shape)
#Xor = np.ones_like(img)
#Xor[10,10] = 0
#Xor[25:27,25:27] = 0
#Xor[50:55,50:55] = 0
#Xor[40:41,10:20] = 0
#Xor[10:20,40:41] = 0
#Xor[30:32,10:20] = 0
#Xor[10:20,30:32] = 0
#g = img * Xor

Xor = np.zeros_like(img)

for row in range(img.shape[0]):
    for col in range(row %2, img.shape[1], 2):
        Xor[row,col] = 1

g = img * Xor

plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.subplot(2, 3, 2), plt.imshow(g, cmap='gray')
plt.subplot(2, 3, 3), plt.imshow(Xor, cmap='gray')

Ig = np.zeros_like(img)
a = np.zeros_like(img)
b = np.zeros_like(img)
step = np.zeros_like(img)
gI_norm = np.zeros_like(img)
T_norm = np.zeros_like(img)
HL_norm = np.zeros_like(img)

u = g.copy()
n = 500
#alpha = 0.0005
#lam = 0.005
alpha = 0.01
lam = 0.001
for i in range(12*n):
    if i%n == 0:
        plt.figure(2)
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(Ig, cmap='gray', vmin=0, vmax=1)
        plt.figure(3)
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(u, cmap='gray', vmin=0, vmax=1)
        plt.figure("a")
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(a, cmap=gkr_col, vmin=-1, vmax=1)
        plt.figure("b")
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(b, cmap=gkr_col, vmin=-1, vmax=1)
        plt.figure("step")
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(step, cmap=gkr_col, vmin=-1, vmax=1)
        plt.figure("norm")
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(gI_norm, cmap='gray', vmin=0, vmax=0.1)
        plt.figure("T_norm")
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(T_norm, cmap='gray', vmin=0, vmax=0.01)
        plt.figure("HL_norm")
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(HL_norm, cmap='gray', vmin=0, vmax=0.5)

    u = update_u(u, g, Xor, alpha, lam)
    u[u<0] = 0
    u[u>1] = 1

plt.show()
