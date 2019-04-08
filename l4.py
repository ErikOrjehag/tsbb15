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

img = lab4.make_circle(256/2, 256/2, 256/3)
noisy_add = add_noise(img, 10)
noisy_mult = mult_noise(img, 10)
Ig, Gdx, Gdy = image_gradient(noisy_add, ksize=6, sigma=0.75)
T = estimate_T(Gdx, Gdy, window_size=np.array([15, 15]))
D = D_from_T(T, m=1)

plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.subplot(2, 3, 2), plt.imshow(noisy_add, cmap='gray')
plt.subplot(2, 3, 3), plt.imshow(noisy_mult, cmap='gray')
plt.subplot(2, 3, 4), plt.imshow(np.linalg.norm(T, axis=(2, 3)), cmap='gray')
plt.subplot(2, 3, 5), plt.imshow(np.linalg.norm(D, axis=(2, 3)), cmap='gray')

plt.show()
