import cvl_labs.lab4 as lab4
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.signal import convolve2d as conv2


def image_gradient(I):
    df = np.atleast_2d(np.array([-1,1]))

    dx = conv2(I, df, mode="same")
    dy = conv2(I, df.T, mode="same")

    gI = np.array([dx, dy])

    T = np.empty(np.shape(I) + (2, 2))
    T[...,0,0] = dx**2
    T[...,0,1] = dx*dy
    T[...,1,0] = T[...,0,1]
    T[...,1,1] = dy**2

    h11 = np.atleast_2d(np.array([1, -2, 1]))
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

    return gI, T, HL

def update_u(u, g, Xor, alpha, lam):

    gI, T, HL = image_gradient(u)

    inner = (HL[...,0,0] * T[...,1,1] -
            2 * HL[...,0,1] * T[...,0,1] +
            HL[...,1,1] * T[...,0,0]) / np.linalg.norm(gI, axis=0) ** 3

    return u - alpha * (Xor * (u - g) - lam*inner)

#img = lab4.make_circle(256/2, 256/2, 256/3)
img = lab4.get_cameraman().astype(float)

g = np.zeros((img.shape[0]*2, img.shape[1]*2))
Xor = np.zeros_like(g)

for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        g[row*2,col*2] = img[row, col]
        Xor[row*2,col*2] = 1

plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.subplot(2, 3, 2), plt.imshow(g, cmap='gray')
plt.subplot(2, 3, 3), plt.imshow(Xor, cmap='gray')

u = g
plt.figure(2)
n = 1
alpha = 1
lam = 0.5
for i in range(12*n):
    if i%n == 0:
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(u, cmap='gray')
    u = update_u(u, g, Xor, alpha, lam)

plt.show()
