import numpy as np
import config
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def makeFittingData(x, z, pdel=3):
    for i in np.arange(pdel):
        z[z.argmax()] = 0
    ind = np.where(z>0)[0]
    x_pick = x[ind]
    z_pick = z[ind]
    return x_pick, z_pick

def gaussian(x, amp, cen, wid, bg):
                return amp * np.exp(-(x-cen)**2 /(2*wid**2)) + bg

def three_gaussians(x, amp, cen, wid, l_amp, ll_cen, rl_cen, l_wid, bg):
    """ 
    Triple gaussian with a large center and two lobes on either shoulder.
    The width and height of the lobes is fixed to be the same.
    ll : left lobe
    rl : right lobe
    """
    return gaussian(x, amp, cen, wid, bg) + gaussian(x, l_amp, ll_cen, l_wid, bg) + gaussian(x, l_amp, rl_cen, l_wid, bg)

def detect_side_lobes(y_data, threshold_ratio=0.1, prominence=0.025):
    """
    Check if there are smaller peaks that are significant but not as
    high as the main peak.  Returns a true/false.  
    """
    try:
        peaks, _ = find_peaks(y_data, prominence=prominence)
        peak_amplitudes = y_data[peaks]
        max_amplitude = np.max(peak_amplitudes)
        lobes = np.any((peak_amplitudes < max_amplitude) & (peak_amplitudes > threshold_ratio*max_amplitude))
    except ValueError:
        lobes = False
    return lobes

def fit_autocorrelation(a, n=20, pdel=3, dist=10, lobe_wid=3, auto_wid=3, bg=1, thres=0.1, prominence=0.025):
    """
    Looks for side lobes.  If it sees side lobes then it fits three gaussians, otherwise, it only fits a single gaussian.
    dist : distance between lobes and center (pixels)
    """
    x = np.arange(2*n) - n
    aroi = a.copy()
    x, z = makeFittingData(x, aroi, pdel=pdel)
    has_side_lobes = detect_side_lobes(aroi, threshold_ratio=thres, prominence=prominence)

    if has_side_lobes:
        # fit with three gaussians
        p0 = np.array([1., 0., auto_wid, 0.4, -dist, dist, lobe_wid, bg])
        bounds = ((0, -1, 2, 0.0, -dist-5, dist-5, 2, 0.0), (1, 1, 12, 1, -dist+5, dist+5, 12, bg+0.25))
        px, covx = curve_fit(three_gaussians, x, z, p0, bounds=bounds)
        xplot = np.linspace(0, n*2, 600) - n
        zplot = three_gaussians(xplot, *px)
    else:
        # fit with one gaussian
        p0 = np.array([1., 0., auto_wid, bg])
        px, covx = curve_fit(gaussian, x, z , p0)
        xplot = np.linspace(0, n*2, 600) - n
        zplot = gaussian(xplot, *px)
    return x, z, xplot, zplot, has_side_lobes, px

def fit_gaussian_curveFit(a, n=20, pdel=3):
    """
    No longer used.
    """
    x = np.arange(2*n) - n
    aroi = a.copy()
    x, z = makeFittingData(x, aroi, pdel=pdel)
    p0 = np.array([1., 0., 3., 1.])
    px, covx = curve_fit(gaussian, x, z, p0)
    xplot = np.linspace(0, n*2, 600) - n
    zplot = gaussian(xplot, *px)
    x = np.arange(2*n)-n
    aroi = a.copy()   
    x,z = makeFittingData(x, aroi, pdel = pdel)
    return x, z, xplot, zplot

def get_center_portion(autocorr, l1, l2):
    shape = autocorr.shape
    center = [shape[0]//2,shape[1]//2]
    return autocorr[center[0]-l1//2:center[0]+l1//2,center[1]-l2//2:center[1]+l2//2]

def getAverageAutocorr(autocorrs, mask, correlation_shape):
    l1 = correlation_shape[0]
    l2 = correlation_shape[1]
    if mask.max() > 1:
        for j,autocorr in enumerate(autocorrs):
            if j == 0:
                a = get_center_portion(autocorr, l1, l2)
            else:
                a += get_center_portion(autocorr, l1, l2)
        a /= len(autocorrs)
    else:
        a = get_center_portion(autocorrs, l1, l2)
    return np.ascontiguousarray(a)

def safe_fit(a):
    """
    Wraps fit_autocorrelation in a try:except and manages the returned
    output by returning empty arrs if the fit fails.
    """
    success = 1
    try:
        x, z, xplot, zplot, lobes, popt = fit_autocorrelation(a,
                                                              config.AUTOCORR_ROI_NN,
                                                              pdel=config.DELTA_PIXELS,
                                                              dist=config.LOBE_DISTANCE,
                                                              lobe_wid=config.LOBE_WIDTH,
                                                              auto_wid=config.AUTOCORR_WIDTH,
                                                              bg=config.BG_LEVEL,
                                                              thres=config.LOBE_THRESHOLD,
                                                              prominence=config.LOBE_PROMINENCE)
    except RuntimeError:
        print("[Curve fit failed.  Returning empty fit for this update]")
        # return data, but with empty fit results
        x = np.arange(2*config.AUTOCORR_ROI_NN) - config.AUTOCORR_ROI_NN
        aroi = a.copy()   
        x,z = makeFittingData(x, aroi, pdel = pdel)
        zzplot = np.empty(600)
        zzplot[:] = np.nan
        xxplot = np.linspace(0, config.AUTOCORR_ROI_NN*2, 600) - config.AUTOCORR_ROI_NN
        popt = np.empty(8)
        popt[:] = np.nan
        xplot, zplot, lobes, popt = xxplot, zzplot, 0, popt
        success = 0
    return x, z, xplot, zplot, lobes, popt, success
