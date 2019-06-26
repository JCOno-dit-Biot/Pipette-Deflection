#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:30:10 2019

@author: jean-christophe
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.ndimage
import skimage.measure
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats.distributions import  t






def ROI_def(img):

    '''
    Parameters
    ----------
    img: arrays
       image to analyze and find edges of a droplet
    angle : int
       rotation angle for the axis of the droplet to be horizontal
    ret : int
        value of the threshold (B&W processing)
    pipetteDiameter: int
        diameter of the pipette in pixels
        
    Returns
    -------
    h : array
        returns the profile of the droplet
        
    Note
    ----
    Crops the image to define ROI
    Need to click 2 points on image prompt (top left corner and bottom right)
    
    '''
    
    plt.figure('Pick top left and bottom right corner')
    plt.imshow(img,cmap='gray'); 
    
    x = (np.floor(plt.ginput(2)))
    x=x.astype(int)

    return x

def findEdge(img,angle,ret, pipetteDiameter):
    
    '''
    Parameters
    ----------
    img: arrays
       image to analyze and find edges of a droplet
    angle : int
       rotation angle for the axis of the droplet to be horizontal
    ret : int
        value of the threshold (B&W processing)
    pipetteDiameter: int
        diameter of the pipette in pixels
        
    Returns
    -------
    h : array
        returns the profile of the droplet
    '''
    # pre-processing of the image ##
    #################################
    
    imgRot=scipy.ndimage.rotate(img,angle,cval=1)
    
    ret,th = cv2.threshold(imgRot,ret,255,cv2.THRESH_BINARY)
 

    #ret,th = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    im_floodfill = th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img_patch= scipy.ndimage.morphology.binary_fill_holes(im_floodfill_inv)
    img_int=img_patch.astype(int)
    
    label_image = skimage.measure.label(img_int)
    for region in skimage.measure.regionprops(label_image):
    # Everywhere, skip small areas
        if region.area < 1000:
           coord=region.coords
           img_int[coord[:,0],coord[:,1]]=0
    
    upEdge=[]
    lowEdge=[]
    h=[]

    for k in range(img_int.shape[1]):
        A=np.argwhere(img_int[1:,k]-img_int[:-1,k])
        if A.any():
            upEdge.append(np.min(A))
            lowEdge.append(np.max(A))
            h.append((lowEdge[-1]-upEdge[-1]))
    
    h=np.asarray(h)
    h=h[h>=pipetteDiameter]/2
    return h

def ccorrf(x, y, unbiased=True, demean=True, sym = True):
    ''' crosscoorrelation function for 1D

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators is n-k, otherwise n
    sym : boolean
        if True, the outpur is symmetrical (lag in both positive and negative)
    Returns
    -------
    ccorrf : array
       cross-correlation array

    Notes
    -----
    This uses np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.
    '''
    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n)
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
        
    if sym:
        return (np.correlate(xo, yo, 'full') / (d*(np.std(x) * np.std(y))))
    else:
        return (np.correlate(xo, yo, 'full') / (d*(np.std(x) * np.std(y))))[n - 1:]   
        



def pippos(RefPCD, PCD,ttest='False'):
    
    ''' crosscoorrelation function for 1D

    Parameters
    ----------
    RefPCD, PCD: arrays
       intensity profile along the cross-correlation line that includes the pipette
    ttest : boolean (Default False)
       if True, runs t-test on the gaussian fit to calculate confidence intervals
    
    Returns
    -------
    ccorrf : array
       cross-correlation array

    Notes
    -----
    d:
        position of the pipette in pix
    tval: 
        if ttest=True, tval returns the results of the t-test
    '''
    
    #performs the cross-correlation. ccf from statsmodels uses np.correlate
    corr=ccorrf(np.array(RefPCD), np.array(PCD),'full')
   
    
    maxCVcoordinates= np.argmax(corr)
    pp=5 # number of points in the fit, 5 to the left, 5 to the right
    
    ## fitting the gaussian to find the position of the pipette
    xdata=np.arange(maxCVcoordinates-pp,maxCVcoordinates+pp,1)
    
    ydata=corr[xdata]
    ## initial parameters of the fit
    mean = sum(xdata*ydata)/len(xdata)               
    sigma = sum(ydata*(xdata-mean)**2)/len(xdata)
    #fit to a gaussian curve
    popt, pcov = curve_fit(gauss_function, xdata,ydata, p0 = [1,maxCVcoordinates, pp])
    d=popt[1]
    
    if ttest ==1:
        # Calculate the 95 confidence interval using t-Student test if needed
        alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
        n = len(ydata)    # number of data points
        m = len(popt) # number of parameters
        dof = max(0, n - m) # number of degrees of freedom
        # student-t value for the dof and confidence level
        tval = t.ppf(1.0-alpha/2., dof)
    else:
        tval='N/A'
        
        
    return d, tval

def gauss_function(x, a, x0, sigma):
    '''Gaussian function needed to fit the cross-correlation function
    --> leads to subpixel resolution
    '''
    return a*np.exp(-(x-x0)**2/(sigma**2))