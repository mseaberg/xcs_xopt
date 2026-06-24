import sys
#from Talbot_functions_crop import *
from lcls_beamline_toolbox.xraybeamline2d.util import Util
from beam import *
from psana import *
import numpy as np
from mpidata import mpidata 
import h5py
import scipy.ndimage.interpolation as interpolate
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas
from pitch2 import *
import h5py
import psana_utility
import zmq
#from numpyClientServer import *
import epics
from psmon import publish
from psmon.plots import Image,XYPlot

#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

    


def runclient(args,pars,comm,rank,size):

    sh_mem = args.live
    #expName = args.experiment
    expName = pars['exp_name']
    hutchName = args.hutch.lower()
    runNum = args.run
    pars['run'] = runNum
    detName = pars['detName']
    thresh = pars['thresh']
    ipm_threshold = pars['ipm_threshold']


    expName = expName
    update = pars['update_events']
    runString = 'exp=%s:run=%s:smd' % (expName, runNum)
    #runString = runNum
    #runString += ':dir=/reg/d/ffb/%s/%s/xtc:live' % (hutchName,expName)
    #print(runString)

    roi = pars['roi']
    xmin = roi[0]
    xmax = roi[1]
    ymin = roi[2]
    ymax = roi[3]


    # miscellaneous parameters
    dx = pars['pixel']

    N = ymax-ymin
    M = xmax-xmin

    x = np.linspace(xmin, xmax, M)*dx
    y = np.linspace(ymin, ymax, N)*dx

    calibDir = '/sdf/data/lcls/ds/%s/%s/calib' % (hutchName, expName)
    #calibDir = '/cds/home/opr/cxiopr/experiments/%s/calib' % (expName)

    ds = []
    if sh_mem:
        
        setOption('psana.calib-dir', calibDir)
        ds = DataSource('shmem=psana.0:stop=no')
        
    else:
        ds = DataSource(runString)

    det0 = Detector(detName)
    ipm4 = Detector('XCS-SB1-BMMON')

    nevents = np.empty(0)

    # initialize instance of the mpidata class for communication with the master process
#    md = mpidata()

    # initialize i1 (which resets after each update) depending on the rank of the process
    i1 = int((rank-1)*update/(size-1))


    md = mpidata()

    
    i0 = -1

    nevents = np.empty(0)

    # event loop
    for nevent,evt in enumerate(ds.events()):

        
        # check if we've reached the event limit        
        if nevent == args.noe : break
        #if nevent%(size-1)!=rank-1: continue # different ranks look at different events

        #if det0.image(evt) is None: continue
        # increment counter
        i1 += 1


        i0 += 1
        nevents = np.append(nevents,nevent) 

        #print(evt.keys())

        # send mpi data object to master when desired
        if i1 == update:

            md=mpidata()


            i1 = 0
            #print(i1)
            if det0.image(evt) is None: continue
            if ipm4.get(evt) is None: continue


            print(nevent)

            # select image ROI
            #img0 = np.copy(det0.image(evt)[ymin:ymax,xmin:xmax])
            print('getting image')
            img0 = det0.image(evt)
            norm = ipm4.get(evt).TotalIntensity()
            #img0[img0<20] = 0
            #img0 -= 20
            #img0[img0<20] = 0
            # check if the image needs to be rotated

            img1 = np.copy(img0[ymin:ymax,xmin:xmax])
            #img1[img1<20] = 0
            img1 -= thresh
            img1[img1<0] = 0

            #if np.sum(img1)<1e5:continue

            N,M = np.shape(img1)
            #print(N)
            #print(M)
            # get scan pv

            #### find peaks ####
            #print(np.shape(scan))

            lineout_x = Util.get_horizontal_lineout(img1)
            lineout_y = Util.get_vertical_lineout(img1)

            thresh2 = 100
            # disregard the centroid calculation if intensity is below the threshold
            if np.sum(img1)>thresh2:
                cx, wx = gaussian_stats(x, lineout_x)
                cy, wy = gaussian_stats(y, lineout_y)
            else:
                cx = np.array(np.nan)
                cy = np.array(np.nan)
                wx = np.array(np.nan)
                wy = np.array(np.nan)

            intensity = np.sum(img1)/norm
            #intensity = np.mean(img1)
            # require ipm4 to be larger than some number (default 50)
            if norm<ipm_threshold:
                #pass
                intensity = np.array(np.nan)

            print('image processed')

            #img0 = img0/np.max(img0)
            md.addarray('cx',cx)
            md.addarray('cy',cy)
            md.addarray('wx',wx)
            md.addarray('wy',wy)
            # normalize by ipm4
            md.addarray('intensity',intensity)
            md.addarray('nevents',nevents[-1])
            if rank==1:
                md.addarray('img',img0[ymin:ymax,xmin:xmax])
            md.small.event = nevent
           
            md.send()

            nevents = np.empty(0)
           # 
    md.endrun()


def runmaster(nClients,args,pars,comm,rank,size):

    print('running')

    #servername = args.server

    #context = zmq.Context()
    #socket1 = context.socket(zmq.PUB)
    #socket1.connect("tcp://"+servername+":12301")

    # get ROI info
    roi = pars['roi']
    xmin = roi[0]
    xmax = roi[1]
    ymin = roi[2]
    ymax = roi[3]

    # initialize arrays
    N = ymax-ymin
    M = xmax-xmin
    dx = pars['pixel']*1e6


    x = np.linspace(xmin,xmax,M)
    y = np.linspace(ymin,ymax,N)
    x,y = np.meshgrid(x,y)

    N_event = 120

    dataDict = {}
    dataDict['nevents'] = np.ones(N_event)*-1
    dataDict['intensity'] = np.zeros(N_event)
    dataDict['cx'] = np.zeros(N_event)
    dataDict['cy'] = np.zeros(N_event)
    dataDict['wx'] = np.zeros(N_event)
    dataDict['wy'] = np.zeros(N_event)

    nevent = -1

    cx_PV = epics.PV('XCS:USER:SND:X_CENTROID')
    cy_PV = epics.PV('XCS:USER:SND:Y_CENTROID')
    wx_PV = epics.PV('XCS:USER:SND:X_WIDTH')
    wy_PV = epics.PV('XCS:USER:SND:Y_WIDTH')
    intensity_PV = epics.PV('XCS:USER:SND:INTENSITY')

    numEvents = 0

    while nClients > 0:
        # Remove client if the run ended
        md = mpidata()
        rank1 = md.recv()
        #print(rank1)
        if md.small.endrun:
            nClients -= 1
        else:

            #nevents = np.append(nevents,md.nevents)
            dataDict['nevents'] = update(md.nevents,dataDict['nevents']) 
            dataDict['intensity'] = update(md.intensity,dataDict['intensity'])
            dataDict['cx'] = update(md.cx,dataDict['cx'])
            dataDict['cy'] = update(md.cy,dataDict['cy'])
            dataDict['wx'] = update(md.wx,dataDict['wx'])
            dataDict['wy'] = update(md.wy,dataDict['wy'])

            if md.nevents>nevent:
                nevent = md.nevents
            



            #if rank1==1:
            #    numEvents += 1
            #    #if np.mod(numEvents,4)==0:
            #    circle = np.abs(x-
            #    imPlot = Image(numEvents,"image",md.img)
            #    publish.send("test_image",imPlot)
            #    print('sending image')

            if True:
                
                mask = dataDict['nevents']>=0
                #mask = np.logical_and(mask,dataDict['intensity']>1e6)

                eventMask = dataDict['nevents'][mask]

                order = np.argsort(eventMask)
                eventMask = eventMask[order]

                intensity = dataDict['intensity'][mask]
                cx = dataDict['cx'][mask]
                cy = dataDict['cy'][mask]
                wx = dataDict['wx'][mask]
                wy = dataDict['wy'][mask]


                #cx = np.average(cx,weights=intensity)
                #cy = np.average(cy,weights=intensity)
                #wx = np.average(wx,weights=intensity)
                #wy = np.average(wy,weights=intensity)
                #intensity = np.mean(intensity)
                cx = cx[-1]
                cy = cy[-1]
                wx = wx[-1]
                wy = wy[-1]
                intensity = intensity[-1]

                print('{}: {}'.format(rank1,cx))
                print(intensity)
                cx_PV.put(cx*1e6)
                cy_PV.put(cy*1e6)
                wx_PV.put(wx*1e6)
                wy_PV.put(wy*1e6)
                intensity_PV.put(intensity)

                if rank1==1:
                    numEvents += 1
                    if np.mod(numEvents,4)==0:

                        w_eff = np.sqrt(wx*wy)*1e6
                        outer_rad = (x*dx-cx*1e6)**2 + (y*dx-cy*1e6)**2<(2*w_eff)**2
                        inner_rad = (x*dx-cx*1e6)**2 + (y*dx-cy*1e6)**2>(2*w_eff-3)**2
                        circle = np.logical_and(inner_rad,outer_rad).astype(float)*500
                        #circle *= 500
                        imPlot = Image(numEvents,"image",md.img+(circle),aspect_ratio=1)
                        #imPlot = Image(numEvents,"image",md.img)
                        publish.send("test_image",imPlot)
                        print('sending image')
    
                #else:
                #    socket1.send_pyobj(send_dict)
                #socket1.send_string('data', zmq.SNDMORE)
                #socket1.send_pyobj(send_dict)
                print("sent data")
                #numpysocket.startClient(servername,12301,send_dict1) 

def gaussian_stats(x_data, y_data, thresh=0.1):

    # normalize input (and subtract any offset)
    y_norm = Util.normalize_trace(y_data)
    # threshold input
    y_data_thresh = Util.threshold_array(y_norm, thresh)

    # calculate centroid
    cx = np.sum(y_data_thresh * x_data) / np.sum(y_data_thresh)

    # calculate second moment
    sx = np.sqrt(np.sum(y_data_thresh * (x_data - cx) ** 2) / np.sum(y_data_thresh))
    fwx_guess = sx * 2.355

    guess = [cx, sx]

    try:
        mask = y_data_thresh > 0
        px, pcovx = optimize.curve_fit(Util.fit_gaussian, x_data[mask], y_norm[mask],p0=guess)
        sx = px[1]
        cx = px[0]
    except:
        print('Fit failed. Using second moment for width.')

    return cx, sx

               

def update(newValue,currentArray):

    if len(np.shape(currentArray))>1:
        currentArray = np.roll(currentArray,-1,axis=0)
        currentArray[-1,:] = newValue
    else:
        currentArray = np.roll(currentArray,-1)
        currentArray[-1] = newValue
    return currentArray


def running_average(arr, window):
    out = pandas.Series(arr).rolling(window, min_periods=1).mean().values
    return out
