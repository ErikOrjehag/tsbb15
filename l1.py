import lab1
import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot  as plt

def image_gradient(I, ksize, sigma):
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
    Igdx = conv2(I, kx, mode="same")
    Igdy = conv2(I, ky, mode="same")
    Ig = np.sqrt(np.add(np.square(Igdx), np.square(Igdy)))
    H = np.atleast_2d(np.exp(-0.5*(np.arange(-ksize,ksize+1)/sigma)**2))
    H = H / np.sum(H)
    Ig = conv2(conv2(Ig, H, mode="same"), np.transpose(H), mode="same")
    Igdx = conv2(Igdx, H, mode="same")
    Igdy = conv2(Igdy, np.transpose(H), mode="same")
    return Ig, Igdx, Igdy

def estimate_T(Jgdx, Jgdy, x, y, window_size):
    T = np.array([[0.0,0.0], [0.0,0.0]])
    col_from = max(0, x - window_size[0])
    col_to = min(np.shape(Jgdx)[1], x + window_size[0])
    row_from = max(0, y - window_size[1])
    row_to = min(np.shape(Jgdx)[0], y + window_size[1])
    for col in range(col_from, col_to):
        for row in range(row_from, row_to):
            T += np.array([[Jgdx[row,col]*Jgdx[row,col],Jgdx[row,col]*Jgdy[row,col]],
                           [Jgdx[row,col]*Jgdy[row,col],Jgdy[row,col]*Jgdy[row,col]]])
    return T

def estimate_e(I, J, Jgdx, Jgdy, x, y, window_size):
    e = np.array([[0.0], [0.0]])
    col_from = max(0, x - window_size[0])
    col_to = min(np.shape(Jgdx)[1], x + window_size[0])
    row_from = max(0, y - window_size[1])
    row_to = min(np.shape(Jgdx)[0], y + window_size[1])
    for col in range(col_from, col_to):
        for row in range(row_from, row_to):
            e += (I[row,col]-J[row,col]) * np.array([[Jgdx[row,col]], [Jgdy[row,col]]])
    return e

def estimate_d(I, J, x, y):
    Ig, _, _ = image_gradient(I, 6, 0.5)
    Jg, Jgdx, Jgdy = image_gradient(J, 6, 0.5)
    #plt.figure("Ig"), plt.imshow(Ig)
    #plt.figure("Jg"), plt.imshow(Jg)
    #plt.figure("Jgdx"), plt.imshow(Jgdx)
    #plt.figure("Jgdy"), plt.imshow(Jgdy)
    dtot = np.array([[0.0], [0.0]])
    xcoord = np.arange(0, np.shape(J)[0], 1)
    ycoord = np.arange(0, np.shape(J)[1], 1)
    Jd = J
    Jgdxd = Jgdx
    Jgdyd = Jgdy
    for _ in range(100):
        T = estimate_T(Jgdxd, Jgdyd, x, y, (35, 20))
        e = estimate_e(I, Jd, Jgdxd, Jgdyd, x, y, (35, 20))
        d = np.linalg.solve(T, e)
        dtot = dtot + d
        if np.linalg.norm(d) < 0.01:
            break
        Jd = scipy.interpolate.RectBivariateSpline(xcoord, ycoord, J)(
            np.arange(dtot[1], np.shape(J)[0] + dtot[1], 1),
            np.arange(dtot[0], np.shape(J)[1]+ dtot[0], 1), grid=True)
        Jgdxd = scipy.interpolate.RectBivariateSpline(xcoord, ycoord, Jgdx)(
            np.arange(dtot[1], np.shape(J)[0] + dtot[1], 1),
            np.arange(dtot[0], np.shape(J)[1]+ dtot[0], 1), grid=True)
        Jgdyd = scipy.interpolate.RectBivariateSpline(xcoord, ycoord, Jgdy)(
            np.arange(dtot[1], np.shape(J)[0] + dtot[1], 1),
            np.arange(dtot[0], np.shape(J)[1]+ dtot[0], 1), grid=True)

    return dtot

def orientation_tensor(img, grad_ksize, grad_sigma, window_size):
    _, dx, dy = image_gradient(img, grad_ksize, grad_sigma)
    T = np.empty(np.shape(img) + (2, 2))
    for x in range(np.shape(img)[1]):
        for y in range(np.shape(img)[0]):
            T[y, x,:,:] = estimate_T(dx, dy, x, y, window_size)
    return T
    #H = np.fromfunction(lambda x,y : T(x,y)(1, 1)*, np.shape(img))

def harris(img, grad_ksize, grad_sigma, ksize, kappa):
    T = orientation_tensor(img, grad_ksize, grad_sigma, (1,1))
    H = np.empty(np.shape(img))
    for x in range(np.shape(img)[1]):
        for y in range(np.shape(img)[0]):
            t = T[y,x,:,:]
            H[y, x] = t[0,0]*t[1,1] - t[0,1]*t[0,1] - 0.05*(t[0,0] + t[1,1])**2
    return H

I = lab1.load_lab_image("chessboard_1.png").astype(np.float64)
J = lab1.load_lab_image("chessboard_2.png").astype(np.float64)

plt.figure("I"), plt.imshow(I)
plt.figure("J"), plt.imshow(J)

#d = estimate_d(I, J, 388, 311)

#print(d)

H = harris(I, 6, 0.5, 1, 1)
HH = H > np.amax(H) * 0.2
HH_max = scipy.signal.order_filter(HH, np.ones((3, 3)), 9-1)
[row, col] = np.nonzero(HH == HH_max)

plt.figure("H"), plt.imshow(H)
plt.figure("HH"), plt.imshow(HH)

print(np.array([row, col]))

plt.show()
