import psana
import numpy as np
from lcls_beamline_toolbox.xraybeamline2d.util import Util
import argparse
import wfs_utils

parser = argparse.ArgumentParser()
parser.add_argument("-b","--hutch", help="hutch name",type=str)
parser.add_argument("-e","--experiment", help="experiment number",type=str)
parser.add_argument("-r","--run", help="run number from DAQ")
parser.add_argument("-n","--noe",help="number of events, all events=0",default=-1, type=int)
parser.add_argument("-c","--config",help="config file name",default='wfs',type=str)
parser.add_argument("-s","--server",help="server name",type=str)
parser.add_argument("-l","--live",help="run live",type=bool,default=False)
args = parser.parse_args()

pars = wfs_utils.parse_wfs_config_gui(args.config)


runNum = args.run

pars['run'] = runNum
detName = pars['detName']
thresh = pars['thresh']
ipm_threshold = pars['ipm_threshold']

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


ds = psana.MPIDataSource('exp=xcsx1015123:run={}:smd'.format(runNum))

det0 = psana.Detector(detName)
ipm4 = psana.Detector('XCS-SB1-BMMON')

sml_dir = '/sdf/data/lcls/ds/xcs/xcsx1015123/hdf5/smalldata/seaberg/'
filename = sml_dir + 'run{:04d}.h5'.format(runNum)
smldata = ds.small_data(filename,gather_interval=100)


# initialize instance of the mpidata class for communication with the master process
#    md = mpidata()

# initialize i1 (which resets after each update) depending on the rank of the process

i0 = -1

nevents = np.empty(0)

# event loop
for nevent,evt in enumerate(ds.events()):

    
    # check if we've reached the event limit        
    if nevent == args.noe : break
    #if nevent%(size-1)!=rank-1: continue # different ranks look at different events

    #if det0.image(evt) is None: continue
    # increment counter


    if det0.image(evt) is None: continue
    if ipm4.get(evt) is None: continue

    # select image ROI
    #img0 = np.copy(det0.image(evt)[ymin:ymax,xmin:xmax])
    img0 = det0.image(evt)
    norm = ipm4.get(evt).TotalIntensity()

    img1 = np.copy(img0[ymin:ymax,xmin:xmax])
    #img1[img1<20] = 0
    img1 -= thresh
    img1[img1<0] = 0

    #if np.sum(img1)<1e5:continue

    N,M = np.shape(img1)

    lineout_x = Util.get_horizontal_lineout(img1)
    lineout_y = Util.get_vertical_lineout(img1)

    thresh2 = 100
    # disregard the centroid calculation if intensity is below the threshold
    cx, wx = gaussian_stats(x, lineout_x)
    cy, wy = gaussian_stats(y, lineout_y)

    intensity = np.sum(img1)/norm
    #intensity = np.mean(img1)
    # require ipm4 to be larger than some number (default 50)
    smldata.event(cx=cx,cy=cy,wx=wx,wy=wy,intensity=intensity,
            ipm4=norm)

    #img0 = img0/np.max(img0)
       # 
smldata.save()


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

         
