#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:09:52 2019

@author: jean-christophe
"""

import deflection
from matplotlib import pyplot as plt
from tkinter import filedialog 
import tkinter as tk
import tkinter.messagebox
import os
import numpy as np
import cv2
import math
from scipy import stats

def data_selector():
  root = tk.Tk()
  root.withdraw()
  root.update()
  file_path = filedialog.askopenfilename(title = 'Select one of the calibration folder') #asks which file you want to analyze and records the filepath and name to be used in the code
  root.destroy()

  directory = os.path.split(file_path)[0] #directory/path of the file you clicked
  times_step = []
  fileNames = []
  for files in [f for f in os.listdir(directory) if f.endswith('.tif')]:#picks the files of csv format
      fileNames.append(files) 
  numFiles = len(fileNames)
  
  for k in fileNames:		#this sorts by time (picture #)
     t = k[k.index('.tif')-3:k.index('.tif')] #depends on the number of digits
     times_step.append(int(t))
     sort_index = np.argsort(np.asarray(times_step))
  fileNames = np.asarray(fileNames)[sort_index]
  return directory, fileNames #returns the directory (string) and the list of individual files

plt.close('all')
plt.ioff()
###############################################################################
'''This section must be completed before running the program. Some parameters
such as angle, ret, Diam should be adjusted in order to get optimal results depending on
the set of data under consideration'''

## Enter microscope calibration here
pix2micron=1.490

## Angle of rotation for the droplet axis to be horizontal
angle=15

## Threshold value to use to go from greyscale to B&W
ret=55

## Enter first picture to consider for calibration
firstPic=1

## Enter last picture to consider for calibration
lastPic=80

##Enter diameter of the pipette in pixels (measured with ImageJ for example)
Diam= 20

###############################################################################

##Browse folder where the images are saved##
directory, fileNames = data_selector()

plt.figure('Please select an horizontal line (2 points) for the 1D cross-correlation')
A=cv2.imread(directory+'/'+fileNames[1],cv2.IMREAD_GRAYSCALE)
F=cv2.imread(directory+'/'+fileNames[-1],cv2.IMREAD_GRAYSCALE)
plt.imshow((A+F).astype('float')/2,cmap='gray')

##select an horizontal line that intersects with the pipette (2points)##
points = np.asarray(plt.ginput(2))
points=points.astype(int)
plt.close('all')

B = A.astype('float')

y=points[0,1]
line=np.arange(points[0,0],points[1,0],1)

PCD = (B[y,line])
RefPCD = PCD #defines the reference profile to find initial position of the pipette

#find initial position of the pipettte
d,tval = deflection.pippos(PCD,RefPCD,0)
xinit=d*pix2micron

defl=[0]
for k in fileNames:
    A=cv2.imread(directory+'/'+k,cv2.IMREAD_GRAYSCALE)
    B = A.astype('float')
    PCD = (B[y,line])
    d,tval = deflection.pippos(RefPCD,PCD,0)
    defl=np.append(defl,d*pix2micron-xinit)
    
plt.figure()
plt.plot(defl)
plt.ylabel(r'Deflection [$\mu$m]')
plt.xlabel(r'Frame')
plt.show()

np.save(directory+'/deflection',defl)
plt.savefig(directory+'/deflection')

##############################################################################
''' In this section, the code is measuring the volume of the droplet at the tip
of the pipette. Assuming the liquid is water, the force applied on the pipette can be 
calculated.

Then F is plotted against deflection, the spring constant of the pipette is given 
by the slope of this straight line.'''


img = cv2.imread(directory+'/'+fileNames[firstPic-1],0)
imgf=cv2.imread(directory+'/'+fileNames[lastPic],0)
x=deflection.ROI_def((img+imgf)/2)

fig=plt.figure()
ax=fig.gca()
V=[]

for i, k in enumerate(fileNames[firstPic-1:lastPic]):
    img = cv2.imread(directory+'/'+k,0)
    img=img[x[0,1]:x[1,1],x[0,0]:x[1,0]]
    edge=np.asarray(deflection.findEdge(img,15,60,Diam))*pix2micron*1E-6
    xx=np.arange(0,len(edge),1)*pix2micron*1E-6
    ax.plot(xx,edge)
    plt.axis('equal')
    V.append(math.pi*np.trapz(edge**2,xx)-len(edge)*pix2micron*1E-6*math.pi*(Diam*pix2micron*1E-6/2)**2)
    
fig.savefig(directory+'/Cap_profiles.eps')

F = (1000*np.array(V)*9.8)*1E9

defl=np.load(directory+'/deflection.npy')

plt.figure()
axdefl=plt.gca()
axdefl.plot(defl[firstPic-1:lastPic],F,'r*',label='_nolegend_')
plt.xlabel(r'Deflection [$\mu$m]')
plt.ylabel(r'F [nN]')

k, intercept, r_value, p_value, std_err = stats.linregress(defl[firstPic-1:lastPic],F)
xfit=np.arange(np.min(defl[firstPic-1:lastPic]),np.max(defl[firstPic-1:lastPic]),.1)
plt.plot(xfit,k*xfit+intercept,'k--')
plt.legend([r'k= ' + str(round(k,3)) + ' nN/$\mu$m'])

plt.savefig(directory+'/Calibration_curve.eps')
np.save(directory+'/deflection',defl,F)


plt.show()
