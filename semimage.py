import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import scipy
from scipy.optimize import leastsq
from scipy import ndimage 
from skimage import io, img_as_float

# filters and utility functions for general purpose

def image_to_np(filename):
    return img_as_float(ndimage.imread(filename))

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
    return ndimage.filters.gaussian_filter(image_array, 1.0)

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

def normalize(image_array, max_val = 100.0):
    image_array = image_array - image_array.min()
    return (image_array/image_array.max())*max_val

##### matched filter bank built specifically for carbon nanotube location #####

def sinc(x, k, a):
    """ return a sinc function cutoff at +/-2*pi/a """
    
    iz = int(len(x)/2)
    il = iz-np.ceil(2.0*np.pi/k)
    iu = iz+np.ceil(2.0*np.pi/k)+1
    
    s = np.sin(k*x)/(k*x) # function to start with
    s[iz] = 1 # numpy doesn't know about limits
    s[il:iu] -= s[il:iu].mean()
    s[0:il] = 0; s[iu:] = 0
    return a*s

def build_kernel(k, a, L, N):
    """ N is an odd integer. The kernel will be size NxN with (x,y) = (0,0)
        in the center. """
    x = np.arange(-(N/2),(N/2)+1,1)
    kernel = np.zeros((N,N), dtype = np.float)
    kernel[abs(x)<L/2.0, :] = sinc(x, k, a)
    return kernel

def build_filter_bank(k, a, L, N, R):
    """ builds the nanotube filter bank 
            k -- inverse length for sinc function
            a -- height of sinc function
            L -- nanotube length to search for
            N -- size of filter matrix
            R -- number of rotations """
    
    rotations = np.linspace(0,180,R+1)[:-1]

    fbank = np.zeros((len(rotations),N,N))
    kernel = build_kernel(k, a, L, N)
    for i, r in enumerate(rotations):
        fbank[i] = ndimage.rotate(kernel, r, reshape=False, mode='nearest')
    return fbank

def apply_filter(im, fbank, threshold):
    """ apply the bank of filters to a given image """

    result = np.zeros(im.shape)
    for i, f in enumerate(fbank):
        imfilt = ndimage.convolve(im, fbank[i], mode='nearest')
        result += imfilt>threshold
    return result

class MatchedFilter(object):

    def __init__(self, k, a, L, N, R, threshold):
        self.threshold = threshold
        self.bank = build_filter_bank(k, a, L, N, R)
    def __call__(self, im):
        return apply_filter(im, self.bank, self.threshold)

###### function to quickly edit nanotube images using matched filter bank #####
    
def cnt_image_edit(filename):
    """ these are currently my favorite settings for processing
        nanotube images using this set of functions. """

    # some stuff to handle files or filelists
    if type(filename)==type(''):
        filename = [filename]
    elif type(filename)==type([]):
        pass
    else:
        print "Enter an string or list of strings"

    # use the first entry to create a new directory
    current_path = os.path.dirname(filename[0])
    new_path = os.path.join(current_path, 'edited')
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    
    params = [1.75, 10.0, 16.0, 25, 15, 3.6] # k, a, L, N, R, threshold
    mfilter = MatchedFilter(*params)
    for f in filename:
        print 'working on file: {0}'.format(f)
        im = image_to_np(f)
        im_filtered = ndimage.filters.median_filter(mfilter(im), size=(3,3))
        fnew = os.path.join(new_path,os.path.basename(f)[:-4]+'.png')
        scipy.misc.toimage(-im_filtered, cmin=-6, cmax=0).save(fnew)
