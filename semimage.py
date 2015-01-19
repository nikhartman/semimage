#some stuff to help plot and save colorful SEM images

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.optimize import leastsq
import scipy.ndimage as ndi
from PIL import Image

def tif_to_np(filename):
    """ import tif and convert to numpy array """
    im = Image.open(filename)
    return np.array(im)

def plane_fit(image_array, guess = [0,0,0]):
    """ fits the image to a plane. returns the original array - best fit plane """
    x,y = np.indices(image_array.shape) #grid indices
    xy = np.array([x.ravel(), y.ravel()])
    f = lambda xy, (a, b, c): a*xy[0] + b*xy[1] +c
    A = leastsq(lambda params: image_array.ravel() - f(xy, params), guess)
    imfit = f(xy, A[0])
    return image_array - imfit.reshape(image_array.shape)

def elipse(x, y, a, b):
    """ cut off function for low pass filter. """
    x_0 = a/0.5
    y_0 = b/0.5
    return (x/x_0)**2 + (y/y_0)**2

def hartman_low_pass_filter(image_array):
    """ low pass filter meant to exclude only the absolute highest frequencies.
        changing the elipse shape determines the cutoff frequencies """
    s = image_array.shape
    imfft = np.fft.fftn(image_array, axes = (0,1))
    for i in range(s[0]):
        for j in range(s[1]):
            if elipse(i, j, s[0], s[1]) > 1:
                imfft[i, j] = 0.0
    return np.absolute(np.fft.ifftn(imfft, axes = (0,1)))

def gauss_filter(image_array):
    """ gaussian filter """
    return ndi.filters.gaussian_filter(image_array, 1.0)

def histeq(image_array,nbr_bins=256):
    """ Rescales image data to give equal weight to all intensity ranges.
        Works very well, amplifies background noise quite a bit. Might be
        best paired with a low pass filter to remove all the high frequency
        crap that gets amplified."""

    #get image histogram
    imhist,bins = np.histogram(image_array.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    image_array_eq = np.interp(image_array.flatten(),bins[:-1],cdf)

    return image_array_eq.reshape(image_array.shape)

def normalize(image_array):
    image_array = image_array - image_array.min()
    return (image_array/image_array.max())*100.0

def color_limits(image_array):
    """ returns limits for color scale of image """
    return image_array.mean()-20.0, image_array.mean()+20.0 #sort of empirically determined

#the next two functions are kind of for test purposes. 
#go to the bottom of the page for the current best case settings

def import_apply_filters(filename, filters = []):
    """ filter original image and return numpy array """
    imarray = tif_to_np(filename)

    for f in filters:
        if f == 'histeq':
            imarray = histeq(imarray)
        if f == 'lpf':
            imarray = hartman_low_pass_filter(imarray)
        if f == 'fit':
            imarray = plane_fit(imarray)
        if f == 'gauss':
            imarray = gauss_filter(imarray)

    return imarray

def image_plot(imarray, cmin, cmax, colormap = cm.Greys_r):
    """ plot image array from SEM """
    
    fdpi = 80
    fsize = tuple([item/fdpi*0.75 for item in imarray.shape[::-1]])
    fig = plt.figure(figsize=fsize, dpi = fdpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax = fig.add_subplot(1,1,1)
    img = ax.imshow(imarray, cmap = colormap, vmin = cmin, vmax = cmax)
    ax.axis('off')    

def edit_color_save(filename, filters = [], colormap = cm.Greys_r):
    """takes a tif images, subtracts a planefit, then saves it in
    color as a .png """

    dir_name = 'edited/'
    if os.path.isdir(dir_name)==False: 
        os.mkdir(dir_name[:-1])
    
    imarray = import_apply_filters(filename, filters = filters)
    
    cmin, cmax = color_limits(imarray)
    
    fdpi = 80
    fsize = tuple([item/fdpi for item in imarray.shape[::-1]])
    fig = plt.figure(figsize=fsize, dpi = fdpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax = fig.add_subplot(1,1,1)
    img = ax.imshow(imarray, cmap = colormap, vmin = cmin, vmax = cmax)
    ax.axis('off')
    new_name = os.path.join(dir_name, filename[:-4]+'_edit.png')
    fig.savefig(new_name, dpi = fdpi)
    fig.clf()
    
def edit_color_show(filename, filters = [], colormap = cm.Greys_r):
    """ displays the image with a standard set of parameters """

    imarray = tif_to_np(filename)

    for f in filters:
        if f == 'histeq':
            imarray = histeq(imarray)
        if f == 'lpf':
            imarray = hartman_low_pass_filter(imarray)
        if f == 'fit':
            imarray = plane_fit(imarray)
    
    cmin, cmax = color_limits(imarray)
    
    fdpi = 80
    fsize = tuple([item/fdpi for item in imarray.shape[::-1]])
    fig = plt.figure(figsize=fsize, dpi = fdpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax = fig.add_subplot(1,1,1)
    img = ax.imshow(imarray, cmap = colormap, vmin = 0, vmax = 325)
    ax.axis('off')
    
#below are some functions to be used quickly when editing files
    
def cnt_image_edit(filename):
    """ these are currently my favorite settings for processing
        nanotube images using this set of functions. """

    dir_name = 'edited/'
    if os.path.isdir(dir_name)==False: 
        os.mkdir(dir_name[:-1])

    imarray = import_apply_filters(filename, filters = ['fit'])
    
    fdpi = 80
    fsize = tuple([item/fdpi for item in imarray.shape[::-1]])
    fig = plt.figure(figsize=fsize, dpi = fdpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax = fig.add_subplot(1,1,1)
    cmin, cmax = color_limits(imarray)
    img = ax.imshow(imarray, cmap = cm.Greys_r, vmin = cmin, vmax = cmax)
    ax.axis('off')
    new_name = os.path.join(dir_name, filename[:-4]+'_edit.png')
    fig.savefig(new_name, dpi = fdpi)
    fig.clf()
    return cmin, cmax

#def device_image_edit(filename):
#    """ similar to above, but my favorite settings for editing device
#        images. """
#
#    imarray = tif_to_np(filename)
