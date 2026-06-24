import psana
import numpy as np
import config
from mpidata import mpidata
from crosscor import *
from fit_correlation import *
from mpi4py import MPI
from utilities import getROI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# user parameters
thres = config.IMG_THRESHOLD # for thresholding the image
updaterate = config.EVENTS_PER_UPDATE # how often we push to master, in events

# to be changed, roi, full camera is 704*768
mask = np.zeros((704, 768))
center = config.DET_CENTER
ROI = getROI(mask.shape, center, rmin=config.DET_ROI_RMIN, rmax=config.DET_ROI_RMAX)

# you can get multiple ROIs and you can get it any shape
mask[ROI>0] = 1

# Epix general mask
mask[:8, :] = 0
mask[-8:, :] = 0
mask[:, :8] = 0
mask[:, -8:] = 0
mask[:, 290:400] = 0
mask[280:400, :] = 0
#mask[352-10:352+10, :] = 0
#mask[:, 384-10:384+10] = 0
#mask[:, 327:515] = 0

cc = crosscor(mask.shape, mask, normalization = 'symavg')
i0thres = config.I0_THRESHOLD
correlation_shape = (30, 30)
# N = 1000 # how many data points to display (-- is this used anywhere??)
nn = config.AUTOCORR_ROI_NN


def runclient(args):
    psana.setOption('psana.calib-dir', config.CALIB_DIR)
    print('starting client...%s'%args.exprun)

    ds = psana.DataSource(args.exprun)
    det = psana.Detector(args.areaDetName)

    sum_img = None
    autocorrsum = None
    crosscorsum = None
    i0sum = None
    i0sum_crosscor = None
    i0_pre = None
    img_pre = None
    neventsInBatch = 0
    neventsInRank = 0

    for run in ds.runs():
        for nevent, evt in enumerate(ds.events()):
            #if nevent%(size-1) == rank-1:
            img = det.calib(evt)
            if img is None: continue
            img[img < thres] = 0.

            #i0s filter, i0 is the scattering intensity in ROI
            i0 = img[mask == 1].mean()
            if i0 < i0thres: continue

            # get autocorrelations
            autocorr = getAverageAutocorr(cc(img.copy()),
                                          mask,
                                          correlation_shape)

            if img_pre is not None:
                crosscorr = getAverageAutocorr(cc(img.copy(), img_pre.copy()),
                                               mask,
                                               correlation_shape)
                if crosscorsum is None:
                    i0sum_crosscor = i0*i0_pre
                    crosscorsum = crosscorr.copy()*i0*i0_pre
                else:
                    i0sum_crosscor += i0*i0_pre
                    crosscorsum += crosscorr.copy()*i0*i0_pre

            if sum_img is None:
                sum_img = img.copy()
                autocorrsum = autocorr*i0**2
                i0sum = i0**2
            else:
                sum_img += img
                autocorrsum += autocorr*i0**2  
                i0sum += i0**2
           
            shape = autocorr.shape


            # fit curves
            xh, zh, xploth, zploth, hlobes, hpx, success_h = safe_fit(autocorr[shape[0]//2, shape[1]//2-nn:shape[1]//2+nn])
            xv, zv, xplotv, zplotv, vlobes, vpx, success_v = safe_fit(autocorr[shape[0]//2-nn:shape[0]//2+nn,shape[1]//2])

            if success_v:
                lobeHeight = vpx[3]
            else:
                lobeHeight = np.nan

            neventsInBatch += 1
            neventsInRank += 1
            img_pre = img.copy()
            i0_pre = i0.copy()
            #if ((nevent != 0) & ((neventsInRank)%updaterate == 0)): 
            
            if i0sum_crosscor is not None:
                senddata=mpidata()
                if rank==1:
                    senddata.addarray('sumimg', np.ascontiguousarray(sum_img))
                    senddata.addarray('autocorrsum', autocorrsum)

                    senddata.addarray('mask', mask)
                senddata.addarray('lobeHeight', lobeHeight)
                senddata.addarray('i0sum', i0sum)
                senddata.addarray('i0sum_crosscor', i0sum_crosscor)
                #senddata.addarray('crosscorsum', crosscorsum)
                senddata.small.nevents = neventsInBatch
                senddata.send()
            sum_img = None
            crosscorrsum = None
            neventsInBatch = 0
        md = mpidata()
        md.endrun()
