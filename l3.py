import lab3
import numpy as np
from matplotlib import pyplot  as plt

img1, img2 = lab3.load_stereo_pair()

plt.figure(1)
plt.subplot(3,2,1), lab3.imshow(img1, cmap='gray')
plt.subplot(3,2,2), lab3.imshow(img2, cmap='gray')

harr1 = lab3.harris(img1, 10, 5)
harr2 = lab3.harris(img2, 10, 5)

plt.subplot(3,2,3), lab3.imshow(harr1, cmap='gray')
plt.subplot(3,2,4), lab3.imshow(harr2, cmap='gray')

sup1 = lab3.non_max_suppression(harr1, 5)
sup2 = lab3.non_max_suppression(harr2, 5)

sup1 /= np.max(sup2)
sup2 /= np.max(sup2)

msize = 20
mask1 = np.zeros_like(sup1)
mask1[msize:-msize,msize:-msize] = 1
mask2 = np.zeros_like(sup2)
mask2[msize:-msize,msize:-msize] = 1

pts1 = np.zeros_like(sup1)
pts2 = np.zeros_like(sup2)
tresh = 0.01
pts1[sup1 > tresh] = 1
pts2[sup2 > tresh] = 1
pts1 *= mask1
pts2 *= mask2

plt.subplot(3,2,5), lab3.imshow(pts1, cmap='gray')
plt.subplot(3,2,6), lab3.imshow(pts2, cmap='gray')

royrc1 = np.nonzero(pts1)[::-1]
royrc2 = np.nonzero(pts2)[::-1]
rois1 = np.array(lab3.cut_out_rois(img1, *royrc1, 21))
rois2 = np.array(lab3.cut_out_rois(img2, *royrc2, 21))

corr = np.array([np.sum(np.square(r1 - rois2), axis=(1,2)) for r1 in rois1])

vals, ri, ci = lab3.joint_min(corr)
vrc = np.array(list(zip(vals, ri, ci)))

maxi = -1
best_F = None

_pl = np.array([royrc1[0][ri.astype(int)], royrc1[1][ri.astype(int)]])
_pr = np.array([royrc2[0][ci.astype(int)], royrc2[1][ci.astype(int)]])

bpl = None
bpr = None

print("LENGTHS VALS", len(vals))

for iter in range(10000):
    pick = vrc[np.random.choice(np.arange(len(vrc)), 8)]

    pl = np.array([royrc1[0][pick[:,1].astype(int)], royrc1[1][pick[:,1].astype(int)]])
    pr = np.array([royrc2[0][pick[:,2].astype(int)], royrc2[1][pick[:,2].astype(int)]])

    F = lab3.fmatrix_stls(pl, pr)

    inliers = np.sum(np.linalg.norm(lab3.fmatrix_residuals(F, _pl, _pr), axis=0) < 20)
    if inliers > maxi:
        maxi = inliers
        best_F = F
        bpl = pl
        bpr = pr

print("Max inliers %.2f" % (maxi / len(vals)))

plt.figure(2)
plt.subplot(1,2,1), lab3.plot_eplines(best_F, _pr, img1.shape)
plt.subplot(1,2,2), lab3.plot_eplines(best_F, _pl, img2.shape)

lab3.show_corresp(img1, img2, bpl, bpr)

plt.show()
