import lab1
import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot  as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def image_gradient(I, ksize, sigma):
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    ky = np.transpose(kx)
    Igdx = conv2(I, kx, mode="same")
    Igdy = conv2(I, ky, mode="same")
    Ig = np.sqrt(np.add(np.square(Igdx), np.square(Igdy)))
    Hx = np.atleast_2d(np.exp(-0.5*(np.arange(-ksize,ksize+1)/sigma)**2))
    Hx = Hx / np.sum(Hx)
    Hy = np.transpose(Hx)
    Ig = conv2(conv2(Ig, Hx, mode="same"), Hy, mode="same")
    Igdx = conv2(Igdx, Hx, mode="same")
    Igdy = conv2(Igdy, Hy, mode="same")
    return Ig, Igdx, Igdy

def estimate_T(Jgdx, Jgdy, x, y, window_size):
    T = np.zeros((2,2))
    col_from = max(0, x - window_size[0] // 2)
    col_to = min(np.shape(Jgdx)[1], x + (window_size[0] - 1) // 2)
    row_from = max(0, y - window_size[1] // 2)
    row_to = min(np.shape(Jgdy)[0], y + (window_size[1] - 1) // 2)
    T[0,0] = np.sum(np.square(Jgdx[row_from:row_to,col_from:col_to]))
    T[0,1] = np.sum(np.multiply(Jgdx[row_from:row_to,col_from:col_to],Jgdy[row_from:row_to,col_from:col_to]))
    T[1,0] = T[0,1]
    T[1,1] = np.sum(np.square(Jgdy[row_from:row_to,col_from:col_to]))
    return T

def estimate_e(I, J, Jgdx, Jgdy, x, y, window_size):
    e = np.zeros((2,1))
    col_from = max(0, x - window_size[0] // 2)
    col_to = min(np.shape(Jgdx)[1], x + (window_size[0] - 1) // 2)
    row_from = max(0, y - window_size[1] // 2)
    row_to = min(np.shape(Jgdy)[0], y + (window_size[1] - 1) // 2)
    
    ij = I[row_from:row_to,col_from:col_to].astype(int) - J[row_from:row_to,col_from:col_to]

    e[0] = np.sum(np.multiply(ij, Jgdx[row_from:row_to,col_from:col_to]))
    e[1] = np.sum(np.multiply(ij, Jgdy[row_from:row_to,col_from:col_to]))
    return e

def estimate_d(I, J, x, y):
    #Ig, _, _ = image_gradient(I, 6, 0.5)
    #plt.figure("Ig"), plt.imshow(Ig, cmap = 'gray')
    Jg, Jgdx, Jgdy = image_gradient(J, 6, 0.5)
    #plt.figure("Jg"), plt.imshow(Jg, cmap = 'gray')
    #plt.figure("Jgdx"), plt.imshow(Jgdx, cmap = gkr_col)
    #plt.figure("Jgdy"), plt.imshow(Jgdy, cmap = gkr_col)
    dtot = np.zeros((2, 1))
    xcoords = np.arange(0, np.shape(J)[0], 1)
    ycoords = np.arange(0, np.shape(J)[1], 1)
    width = np.shape(J)[1]
    height = np.shape(J)[0]
    Jd = J
    Jgdxd = Jgdx
    Jgdyd = Jgdy
    window = (70, 40)
    for _ in range(100):
        T = estimate_T(Jgdxd, Jgdyd, x, y, window)
        e = estimate_e(I, Jd, Jgdxd, Jgdyd, x, y, window)
        d = np.linalg.solve(T, e)
        dtot = dtot + d
        if np.linalg.norm(d) < 0.01:
            break
        Jd = scipy.interpolate.RectBivariateSpline(xcoords, ycoords, J)(
            np.arange(dtot[1], height + dtot[1]),
            np.arange(dtot[0], width + dtot[0]), grid=True)
        Jgdxd = scipy.interpolate.RectBivariateSpline(xcoords, ycoords, Jgdx)(
            np.arange(dtot[1], height + dtot[1]),
            np.arange(dtot[0], width + dtot[0]), grid=True)
        Jgdyd = scipy.interpolate.RectBivariateSpline(xcoords, ycoords, Jgdy)(
            np.arange(dtot[1], height + dtot[1]),
            np.arange(dtot[0], width + dtot[0]), grid=True)
    return dtot

def orientation_tensor(img, grad_ksize, grad_sigma, window_size):
    _, dx, dy = image_gradient(img, grad_ksize, grad_sigma)
    T = np.empty(np.shape(img) + (2, 2))
    for x in range(np.shape(img)[1]):
        for y in range(np.shape(img)[0]):
            T[y, x,:,:] = estimate_T(dx, dy, x, y, window_size)
    return T

def harris(img, grad_ksize, grad_sigma, ksize, kappa):
    T = orientation_tensor(img, grad_ksize, grad_sigma, (3,3))
    H = np.empty(np.shape(img))
    
    # H = T11*T22 - T12^2 - k(T11 + T22)^2
    H[:,:] = T[:,:,0,0]*T[:,:,1,1] - T[:,:,0,1]**2 - 0.05*(T[:,:,0,0] + T[:,:,1,1])**2
    return H


# Green-black-red color map for gradients
gkr_col = np.zeros((255, 4))
gkr_col[:,3] = 1
gkr_col[:127,1] = np.linspace(1.0, 0.0, 127, False)
gkr_col[127:,0] = np.linspace(0.0, 1.0, 128, True)
gkr_col = ListedColormap(gkr_col)


I = lab1.load_lab_image("chessboard_1.png")
J = lab1.load_lab_image("chessboard_2.png")

plt.figure("I"), plt.imshow(I, cmap='gray', vmin = 0, vmax = 255)
plt.figure("J"), plt.imshow(J, cmap='gray', vmin = 0, vmax = 255)

#d = estimate_d(I, J, 388, 311)

#print(d)

H = harris(I, 6, 0.5, 1, 1)
HH = H > np.amax(H) * 0.2
HH_max = scipy.signal.order_filter(HH, np.ones((3, 3)), 9-1)
[row, col] = np.nonzero(HH == HH_max)

plt.figure("H"), plt.imshow(H, cmap='gray')
plt.figure("HH"), plt.imshow(HH, cmap='gray')

print(np.array([row, col]))

plt.show()
