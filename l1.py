import lab1
import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot  as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Circle

def image_gradient(I, ksize, sigma):
    lp = np.atleast_2d(np.exp(-0.5*(np.arange(-ksize,ksize+1)/sigma)**2))
    lp = lp / np.sum(lp)
    df = np.atleast_2d(-1.0/np.square(sigma) * np.arange(-ksize,ksize+1) * lp)
    Ig = conv2(conv2(I, lp, mode="same"), lp.T, mode="same")
    Igdx = conv2(Ig, df, mode="same")
    Igdy = conv2(Ig, df.T, mode="same")
    return Ig, Igdx, Igdy

def estimate_T(Jgdx, Jgdy, x, y, window_size):
    T = np.zeros((2,2))
    col_from = max(0, x - window_size[0] // 2)
    col_to = min(np.shape(Jgdx)[1], x + int((window_size[0] - 1) / 2))
    row_from = max(0, y - window_size[1] // 2)
    row_to = min(np.shape(Jgdy)[0], y + int((window_size[1] - 1) / 2))
    T[0,0] = np.sum(np.square(Jgdx[row_from:row_to,col_from:col_to]))
    T[0,1] = np.sum(np.multiply(Jgdx[row_from:row_to,col_from:col_to],Jgdy[row_from:row_to,col_from:col_to]))
    T[1,0] = T[0,1]
    T[1,1] = np.sum(np.square(Jgdy[row_from:row_to,col_from:col_to]))
    return T

def estimate_e(Ig, Jg, Jgdx, Jgdy, x, y, window_size):
    e = np.zeros((2,1))
    col_from = max(0, x - window_size[0] // 2)
    col_to = min(np.shape(Jgdx)[1], x + int((window_size[0] - 1) / 2))
    row_from = max(0, y - window_size[1] // 2)
    row_to = min(np.shape(Jgdy)[0], y + int((window_size[1] - 1) / 2))
    ij = Ig[row_from:row_to,col_from:col_to] - Jg[row_from:row_to,col_from:col_to]
    e[0] = np.sum(np.multiply(ij, Jgdx[row_from:row_to,col_from:col_to]))
    e[1] = np.sum(np.multiply(ij, Jgdy[row_from:row_to,col_from:col_to]))
    return e

def estimate_d(I, J, x, y, window_size):
    Ig, _, _ = image_gradient(I, 6, 0.5)
    #plt.figure("Ig"), plt.imshow(Ig, cmap = 'gray')
    Jg, Jgdx, Jgdy = image_gradient(J, 6, 0.5)
    #plt.figure("Jg"), plt.imshow(Jg, cmap = 'gray')
    #plt.figure("Jgdx"), plt.imshow(Jgdx, vmin = -100, vmax = 100, cmap = gkr_col)
    #plt.figure("Jgdy"), plt.imshow(Jgdy, vmin = -100, vmax = 100, cmap = gkr_col)
    dtot = np.zeros((2, 1))
    xcoords = np.arange(0, np.shape(Jg)[0])
    ycoords = np.arange(0, np.shape(Jg)[1])
    width = np.shape(J)[1]
    height = np.shape(J)[0]
    Jgd = Jg
    Jgdxd = Jgdx
    Jgdyd = Jgdy
    for _ in range(100):
        T = estimate_T(Jgdxd, Jgdyd, x, y, window_size)
        e = estimate_e(Ig, Jgd, Jgdxd, Jgdyd, x, y, window_size)
        d = np.linalg.solve(T, e)
        dtot = dtot + d
        if np.linalg.norm(d) < 0.1:
            break
        Jgd = scipy.interpolate.RectBivariateSpline(xcoords, ycoords, Jg)(
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
    T = orientation_tensor(img, grad_ksize, grad_sigma, (ksize,ksize))
    H = np.empty(np.shape(img))
    # H = T11*T22 - T12^2 - k(T11 + T22)^2
    H[:,:] = T[:,:,0,0]*T[:,:,1,1] - T[:,:,0,1]**2 - kappa*(T[:,:,0,0] + T[:,:,1,1])**2
    return H


# Green-black-red color map for gradients
gkr_col = np.zeros((255, 4))
gkr_col[:,3] = 1
gkr_col[:127,1] = np.linspace(1.0, 0.0, 127, False)
gkr_col[127:,0] = np.linspace(0.0, 1.0, 128, True)
gkr_col = ListedColormap(gkr_col)

I = lab1.load_lab_image("chessboard_1.png")
size = np.shape(I)

#plt.figure("I"), plt.imshow(I, cmap='gray', vmin = 0, vmax = 255)

H = harris(I, 6, 0.5, 3, 0.05)
mask = (H > np.amax(H) * 0.3)
H = H * mask
H_max = scipy.signal.order_filter(H, np.ones((3, 3)), 9-1)
[row, col] = np.nonzero((H == H_max)*mask)
trackers = list(zip(col, row))
trackers.sort(key = lambda p: (p[0]-size[1]/2)**2 + (p[1]-size[0]/2)**2)
trackers = np.array(trackers[0:5])

print(trackers)

H_bin = np.zeros(np.shape(H))
H_bin[row,col] = 1

plt.figure("H"), plt.imshow(H, cmap='gray')
plt.figure("H_max"), plt.imshow(H_max, cmap='gray')
plt.figure("H_bin"), plt.imshow(H_bin, cmap='gray')

window = 30

_, ax = plt.subplots(1, num="J1")
ax.imshow(I, cmap='gray')
for tracker in trackers:
    ax.add_patch(Circle(tuple(tracker), window/2, fill=False, edgecolor='red', linewidth=1))

for i in range(2, 11):
    J = lab1.load_lab_image("chessboard_%d.png" % i)
    _, ax = plt.subplots(1, num="J%d" % i)
    ax.imshow(J, cmap='gray')
    for tracker in trackers:
        d = estimate_d(I, J, tracker[0], tracker[1], (window, window))
        tracker[0] = np.round(tracker[0] + d[0])
        tracker[1] = np.round(tracker[1] + d[1])
        ax.add_patch(Circle(tuple(tracker), window/2, fill=False, edgecolor='red', linewidth=1))
    I = J

plt.show()
