# service parameters
#-------------------------------------------------
UPDATE_RATE = 1 # plot-push frequency, measured in "client updates"
EVENTS_PER_UPDATE = 50 # events per update from client.  This is the smallest averaging window. 
RECENT_UPDATE_RATE = 4 # # how many client updates to use for "recent autocorrsum", which is averaged over RECENT_UPDATE_RATE*EVENTS_PER_UPDATE events.
RECENT_BUFFER_LEN = 30 # length of the buffer that remembers the recent autocorrelations.
PLOT_ASPECT = 1 # value for publish.plot_opts.aspect 

# autocorrelation parameters
#-------------------------------------------------
IMG_THRESHOLD = 4 # pixel value to threshold image (KeV)
#IMG_THRESHOLD = 70 # for testing (xcslx6920)
I0_THRESHOLD = 0.00001 # detector image sum threshold value (KeV)
#I0_THRESHOLD = 1 # for testing (xcslx6920)
LOBE_PROMINENCE = 0.015 # peak prominence threshold for detecting peaks as candidate side-lobes
LOBE_THRESHOLD = 0.1 # Minimum ratio of lobe to peak amplitude for detecting side-lobes
LOBE_DISTANCE = 8 # best guess for fitting side lobe distance from center gaussian (in pixels) 
LOBE_WIDTH = 3 # best guess for fitting the gaussian width of the side lobe peaks
AUTOCORR_WIDTH = 3 # best guess for fitting the gaussian width of the autocorrelation peak
BG_LEVEL = 1 # best guess for fitting level of autocorrelation background
AUTOCORR_ROI_NN = 15 # 2 nn is the size of the autocorr ROI
DELTA_PIXELS = 3 # number of pixels at the center of autocorrelation to remove 

# detector parameters
#-------------------------------------------------
CALIB_DIR = '/sdf/data/lcls/ds/xcs/xcsx1016723/calib'
#CALIB_DIR = '/reg/d/psdm/xcs/xcslx6920/calib' # for testing (xcslx6920)
DET_ROI_RMIN = 392
DET_ROI_RMAX = 440
#DET_ROI_RMIN = 
#DET_ROI_RMIN = 190 # for testing (xcslx6920)
#DET_ROI_RMAX = 220 # for testing (xcslx6920)
#DET_CENTER = [306, 338]
DET_CENTER = [412, 355]
#DET_CENTER = [200,200]
#DET_CENTER = [380, 260] # for testing (xcslx6920)
