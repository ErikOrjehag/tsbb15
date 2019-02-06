import lab1
import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot  as plt

def image_gradient(I, J, ksize, sigma):
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
    Igdx = conv2(I, kx, mode="same")
    Igdy = conv2(I, ky, mode="same")
    Jgdx = conv2(J, kx, mode="same")
    Jgdy = conv2(J, ky, mode="same")
    Ig = np.sqrt(np.add(np.square(Igdx), np.square(Igdy)))
    Jg = np.sqrt(np.add(np.square(Jgdx), np.square(Jgdy)))
    H = np.atleast_2d(np.exp(-0.5*(np.arange(-ksize,ksize+1)/sigma)**2))
    H = H / np.sum(H)
    Ig = conv2(conv2(Ig, H, mode="same"), np.transpose(H), mode="same")
    Jg = conv2(conv2(Jg, H, mode="same"), np.transpose(H), mode="same")
    Jgdx = conv2(Jgdx, H, mode="same")
    Jgdy = conv2(Jgdy, np.transpose(H), mode="same")
    return Ig, Jg, Jgdx, Jgdy

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

#I = lab1.load_lab_image("chessboard_1.png").astype(np.float64)
#J = lab1.load_lab_image("chessboard_2.png").astype(np.float64)

I, J, dTrue = lab1.get_cameraman()
I = I.astype(np.float64)
J = J.astype(np.float64)


plt.figure("I"), plt.imshow(I)
plt.figure("J"), plt.imshow(J)

Ig, Jg, Jgdx, Jgdy = image_gradient(I, J, 9, 2)

#plt.figure("Ig"), plt.imshow(Ig)
#plt.figure("Jg"), plt.imshow(Jg)
#plt.figure("Jgdx"), plt.imshow(Jgdx)
#plt.figure("Jgdy"), plt.imshow(Jgdy)

dtot = np.array([[0.0], [0.0]])

x = np.arange(0, np.shape(J)[0], 1)
y = np.arange(0, np.shape(J)[1], 1)

Jd = J
Jgdxd = Jgdx
Jgdyd = Jgdy
for _ in range(20):
    T = estimate_T(Jgdxd, Jgdyd, 120, 85, (35, 20))
    e = estimate_e(I, Jd, Jgdxd, Jgdyd, 120, 85, (35, 20))

    d = np.linalg.solve(T, e)
    dtot = dtot + d
    if np.linalg.norm(d) < 0.1:
        break

    Jd = scipy.interpolate.RectBivariateSpline(x, y, J)(
        np.arange(dtot[1], np.shape(J)[0] + dtot[1], 1),
        np.arange(dtot[0], np.shape(J)[1]+ dtot[0], 1), grid=True)

    Jgdxd = scipy.interpolate.RectBivariateSpline(x, y, Jgdx)(
        np.arange(dtot[1], np.shape(J)[0] + dtot[1], 1),
        np.arange(dtot[0], np.shape(J)[1]+ dtot[0], 1), grid=True)

    Jgdyd = scipy.interpolate.RectBivariateSpline(x, y, Jgdy)(
        np.arange(dtot[1], np.shape(J)[0] + dtot[1], 1),
        np.arange(dtot[0], np.shape(J)[1]+ dtot[0], 1), grid=True)

print(T)
print(e)
print(_)
print(d)
print(dtot)
print(dTrue)

plt.show()
