import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, amp, cen, wid, bg):
                return amp * np.exp(-(x-cen)**2 /(2*wid**2))+bg

def fit_gaussian_rocking_curve(th, i0, p0):
    p,cov = curve_fit(gaussian,th, i0 ,p0)
    th0 = p[1]
    xplot = np.linspace(p[1]-0.005, p[1]+0.005, 10000)
    yplot = gaussian(xplot,*p)
    return th0,xplot,yplot

def getROI(shape,center,rmin=345, rmax = 385):
    #for center, index 0 is y and index 1 is x
    x, y = np.indices((shape[0],shape[1]))
    r = np.hypot(x-center[1],y-center[0])
    mask = x*0
    mask[(r>rmin)&(r<rmax)] = 1
    return mask