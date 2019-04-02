import lab3
import numpy as np
from matplotlib import pyplot  as plt

img1, img2 = lab3.load_stereo_pair()

plt.figure(1)
plt.subplot(3,2,1), lab3.imshow(img1, cmap='gray')
plt.subplot(3,2,2), lab3.imshow(img2, cmap='gray')

harr1 = lab3.harris(img1, 3, 5)
harr2 = lab3.harris(img2, 3, 5)

plt.subplot(3,2,3), lab3.imshow(harr1, cmap='gray')
plt.subplot(3,2,4), lab3.imshow(harr2, cmap='gray')

sup = 15
sup1 = lab3.non_max_suppression(harr1, sup)
sup2 = lab3.non_max_suppression(harr2, sup)

sup1 /= np.max(sup2)
sup2 /= np.max(sup2)

msize = 30
mask1 = np.zeros_like(sup1)
mask1[msize:-msize,msize:-msize] = 1
mask2 = np.zeros_like(sup2)
mask2[msize:-msize,msize:-msize] = 1

pts1 = np.zeros_like(sup1)
pts2 = np.zeros_like(sup2)
tresh = 0.0001
pts1[sup1 > tresh] = 1
pts2[sup2 > tresh] = 1
pts1 *= mask1
pts2 *= mask2

color1 = np.stack((img1,)*3, axis=-1)
color1[pts1.astype(np.bool)] = (255, 0, 0)
color2 = np.stack((img2,)*3, axis=-1)
color2[pts2.astype(np.bool)] = (255, 0, 0)

plt.subplot(3,2,5), lab3.imshow(color1, cmap='gray')
plt.subplot(3,2,6), lab3.imshow(color2, cmap='gray')

royrc1 = np.nonzero(pts1)[::-1]
royrc2 = np.nonzero(pts2)[::-1]
rois1 = np.array(lab3.cut_out_rois(img1, *royrc1, 25), dtype=float)
rois2 = np.array(lab3.cut_out_rois(img2, *royrc2, 25), dtype=float)

r11 = rois1[0,...]
#print(rois2 - r11[None,...])
#print((rois2 - r11[None,...]).shape)

corr = np.array([np.linalg.norm(rois2 - r1[None,...], axis=(1,2)) for r1 in rois1])

plt.figure(11), lab3.imshow(corr, cmap='gray')

#print(corr.shape)

vals, ri, ci = lab3.joint_min(corr)
vrc = np.array(list(zip(vals, ri, ci)))

maxi = -1
best_F = None
best_std = 1e20
best_inliers = None

_pl = np.array([royrc1[0][ri.astype(int)], royrc1[1][ri.astype(int)]])
_pr = np.array([royrc2[0][ci.astype(int)], royrc2[1][ci.astype(int)]])

bpl = None
bpr = None

print("LENGTHS VALS", len(vals))

this_happened = 0

for iter in range(10000):
    pick = vrc[np.random.choice(np.arange(len(vrc)), 8)]

    pl = np.array([royrc1[0][pick[:,1].astype(int)], royrc1[1][pick[:,1].astype(int)]])
    pr = np.array([royrc2[0][pick[:,2].astype(int)], royrc2[1][pick[:,2].astype(int)]])

    F = lab3.fmatrix_stls(pl, pr)

    residuals = lab3.fmatrix_residuals(F, _pl, _pr)

    dists = np.mean(np.abs(residuals), axis=0)
    tresh = 5
    inliers = dists < tresh
    n_inliers = np.sum(inliers)

    std = np.std(residuals[:,dists < tresh].flatten())
    if n_inliers > maxi or (n_inliers == maxi and std < best_std):
        maxi = n_inliers
        best_inliers = inliers
        best_F = F
        best_std = std
        bpl = pl
        bpr = pr
        this_happened += 1

print("THIS HAPPENED", this_happened)

print("Max inliers %d -> %.2f" % (maxi, maxi / len(vals)))

plt.figure(2)
plt.subplot(1,2,1), lab3.imshow(img1), lab3.plot_eplines(best_F, _pr, img1.shape)
plt.subplot(1,2,2), lab3.imshow(img2), lab3.plot_eplines(best_F.T, _pl, img2.shape)

lab3.show_corresp(img1, img2, bpl, bpr, vertical=True)

#plt.show()

# INSERT CODE HERE!

from scipy.optimize import least_squares

Fhat = best_F

P, Pprim = lab3.fmatrix_cameras(Fhat)

ipl = _pl[:,best_inliers]
ipr = _pr[:,best_inliers]
print(_pl.shape)
print(ipl.shape)

X = np.array([lab3.triangulate_optimal(P, Pprim, ipl[:,i], ipr[:,i]) for i in range(ipl.shape[1])])

params = np.hstack((P.ravel(), X.ravel()))

print(P.shape)
print(X.shape)
print(params.shape)

solution = least_squares(
    fun=lab3.fmatrix_residuals_gs,
    x0=params,
    args=(ipl, ipr)
)

print(solution.success)
print(solution.message)
print(solution.status)
print(solution.x[0:12].reshape((3,4)))

C1 = solution.x[0:12].reshape((3,4))
C2 = Pprim

Fg = lab3.fmatrix_from_cameras(C1, C2)


print(lab3.fmatrix_residuals(best_F, ipl, ipr))
print(lab3.fmatrix_residuals(Fg, ipl, ipr))

plt.figure(3)
plt.subplot(1,2,1), lab3.imshow(img1), lab3.plot_eplines(Fg, ipr, img1.shape)
plt.subplot(1,2,2), lab3.imshow(img2), lab3.plot_eplines(Fg.T, ipl, img2.shape)

plt.show()
