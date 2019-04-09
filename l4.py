import cvl_labs.lab4 as lab4
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.signal import convolve2d as conv2

def add_noise(img, SNR):
    var = np.var(img) / (10 ** (SNR / 10))
    return img + np.random.normal(scale=sqrt(var), size=np.shape(img))

def mult_noise(img, SNR):
    var = np.var(img) / (10 ** (SNR / 10))
    noise = np.random.normal(scale=sqrt(var), size=np.shape(img))
    return np.power(10, np.log10(img) + noise)

def image_gradient(I, ksize, sigma):
    lp = np.atleast_2d(np.exp(-0.5*(np.arange(-ksize,ksize+1)/sigma)**2))
    lp = lp / np.sum(lp)
    df = np.atleast_2d(-1.0/np.square(sigma) * np.arange(-ksize,ksize+1) * lp)
    Ig = conv2(conv2(I, lp, mode="same"), lp.T, mode="same")
    Gdx = conv2(Ig, df, mode="same")
    Gdy = conv2(Ig, df.T, mode="same")
    return Ig, Gdx, Gdy

def estimate_T(Gdx, Gdy, window_size):
    Gdx2 = np.square(Gdx)
    Gdxy = np.multiply(Gdx, Gdy)
    Gdy2 = np.square(Gdy)
    window = np.ones(window_size)
    T = np.empty(np.shape(Gdx) + (2, 2))
    T[...,0,0] = conv2(Gdx2, window, mode="same")
    T[...,0,1] = conv2(Gdxy, window, mode="same")
    T[...,1,0] = T[...,0,1]
    T[...,1,1] = conv2(Gdy2, window, mode="same")
    return T

def D_from_T(T, m=1):
    lam, e = np.linalg.eig(T)
    alp = np.exp(-lam / m)
    first = alp[...,0,None,None] * np.matmul(e[...,0,None], np.swapaxes(e[...,0,None], -1, -2))
    second = alp[...,1,None,None] * np.matmul(e[...,1,None],  np.swapaxes(e[...,1,None], -1, -2))
    return first + second

def compute_HL(L):
    h11 = np.atleast_2d(np.array([1, -2, 1]))
    h12 = np.array([
        [1/2, 0, -1/2],
        [0, 0, 0],
        [-1/2, 0, 1/2]
    ])
    h22 = h11.T
    HL = np.empty(np.shape(L) + (2, 2))
    HL[...,0,0] = conv2(L, h11, mode="same")
    HL[...,0,1] = conv2(L, h12, mode="same")
    HL[...,1,0] = HL[...,0,1]
    HL[...,1,1] = conv2(L, h22, mode="same")
    return HL

def update_L(L, ds, D, HL):
    return L + ds / 2 * np.trace(np.matmul(D, HL), axis1=-1, axis2=-2)

#img = lab4.make_circle(256/2, 256/2, 256/3)
img = lab4.get_cameraman().astype(float)

noisy_add = add_noise(img, 15)
noisy_add[noisy_add < 0] = 0
noisy_add[noisy_add > 255] = 255

noisy_mult = mult_noise(img, 60)
noisy_mult[noisy_mult < 0] = 0
noisy_mult[noisy_mult > 255] = 255

noisy_img = noisy_add
#noisy_img = noisy_mult

Ig, Gdx, Gdy = image_gradient(noisy_img, ksize=3, sigma=3)
T = estimate_T(Gdx, Gdy, window_size=np.array([3, 3]))
D = D_from_T(T, m=10)

plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.subplot(2, 3, 2), plt.imshow(noisy_add, cmap='gray')
plt.subplot(2, 3, 3), plt.imshow(noisy_mult, cmap='gray')
plt.subplot(2, 3, 4), plt.imshow(np.linalg.norm(T, axis=(2, 3)), cmap='gray')
plt.subplot(2, 3, 5), plt.imshow(np.linalg.norm(D, axis=(2, 3)), cmap='gray')

ds = 0.1
L = noisy_img
plt.figure(2)
n = 4
for i in range(12*n):
    if i%n == 0:
        plt.subplot(3, 4, (i//n)+1).title.set_text(i), plt.imshow(L, cmap='gray')
    HL = compute_HL(L)
    L = update_L(L, ds, D, HL)

plt.show()
