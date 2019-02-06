import numpy as np
import matplotlib.image as mpimg

def make_circle(x,y,radius, image_sz = 256):
    xx,yy = np.meshgrid(np.arange(image_sz),np.arange(image_sz))
    circle = (xx-x)**2 + (yy-y)**2 < radius**2

    return circle

def get_cameraman():
    return mpimg.imread('/courses/TSBB15/images/cameraman.tif')
