import lab1, lab2
import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot  as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Circle

def image_gradient(I, J, ksize, sigma):
    img = I / 2 + J / 2
    lp = np.atleast_2d(np.exp(-0.5*(np.arange(-ksize,ksize+1)/sigma)**2))
    lp = lp / np.sum(lp)
    df = np.atleast_2d(-1.0/np.square(sigma) * np.arange(-ksize,ksize+1) * lp)
    Ig = conv2(conv2(I, lp, mode="same"), lp.T, mode="same")
    Jg = conv2(conv2(J, lp, mode="same"), lp.T, mode="same")
    Gg = conv2(conv2(img, lp, mode="same"), lp.T, mode="same")
    Gdx = conv2(Gg, df, mode="same")
    Gdy = conv2(Gg, df.T, mode="same")
    return Ig, Jg, Gdx, Gdy

def estimate_Z(Gdx, Gdy, window_size):
    Gdx2 = np.square(Gdx)
    Gdxy = np.multiply(Gdx, Gdy)
    Gdy2 = np.square(Gdy)
    filter = np.ones(window_size)
    Z = np.empty(np.shape(Gdx) + (2, 2))
    Z[...,0,0] = conv2(Gdx2, filter, mode="same")
    Z[...,0,1] = conv2(Gdxy, filter, mode="same")
    Z[...,1,0] = Z[:,:,0,1]
    Z[...,1,1] = conv2(Gdy2, filter, mode="same")
    return Z

def estimate_e(Ig, Jg, Gdx, Gdy, window_size):
    e = np.empty(np.shape(Gdx) + (2,))
    filter = np.ones(window_size)
    ij = Ig - Jg
    e[...,0] = conv2(np.multiply(ij, Gdx), filter, mode="same")
    e[...,1] = conv2(np.multiply(ij, Gdy), filter, mode="same")
    return e

def LK_equation(I, J, window_size):
    Ig, Jg, Gdx, Gdy = image_gradient(I, J, 6, 0.5)
    #plt.figure("Ig"), plt.imshow(Ig, cmap = 'gray')
    #plt.figure("Jg"), plt.imshow(Jg, cmap = 'gray')
    #plt.figure("Gdx"), plt.imshow(Gdx, vmin = -100, vmax = 100, cmap = gkr_col)
    #plt.figure("Gdy"), plt.imshow(Gdy, vmin = -100, vmax = 100, cmap = gkr_col)
    Z = estimate_Z(Gdx, Gdy, window_size)
    e = estimate_e(Ig, Jg, Gdx, Gdy, window_size)
    d = np.linalg.solve(Z, e)
    return d


# Green-black-red color map for gradients
gkr_col = np.zeros((255, 4))
gkr_col[:,3] = 1
gkr_col[:127,1] = np.linspace(1.0, 0.0, 127, False)
gkr_col[127:,0] = np.linspace(0.0, 1.0, 128, True)
gkr_col = ListedColormap(gkr_col)

I, J, dTrue = lab1.get_cameraman()
d = LK_equation(I, J, (70, 40))
print("Estimate d = ", d[85,120,:])
print("True d = ", dTrue)

I = lab1.load_lab_image("chessboard_1.png")
J = lab1.load_lab_image("chessboard_2.png")
d = LK_equation(I, J, (20, 20))
plt.figure("X"), plt.imshow(d[...,0], cmap='gray')
plt.figure("Y"), plt.imshow(d[...,1], cmap='gray')

lab2.gopimage(d)

plt.show()
