from psmon import publish
from psmon.plots import XYPlot, Image, MultiPlot
import numpy as np
import collections
import time
import config
import epics
from mpidata import mpidata 
from fit_correlation import *

# PARAMETERS ARE SET IN `config.py` 
#---------------------------------------------
# service parameters
publish.plot_opts.aspect = config.PLOT_ASPECT
updaterate = config.UPDATE_RATE
recent_updaterate = config.RECENT_UPDATE_RATE
eventsper = config.EVENTS_PER_UPDATE
deque_len = config.RECENT_BUFFER_LEN

# autocorrelation parameters
prom = config.LOBE_PROMINENCE
thres = config.LOBE_THRESHOLD
lobe_prom = config.LOBE_PROMINENCE
lobe_thres = config.LOBE_THRESHOLD
lobe_dist = config.LOBE_DISTANCE
lobe_wid = config.LOBE_WIDTH
autocorr_wid = config.AUTOCORR_WIDTH
bg = config.BG_LEVEL
nn = config.AUTOCORR_ROI_NN
pdel = config.DELTA_PIXELS
#---------------------------------------------

def runmaster(nClients):
    while (1):
        print('**** New Run ****')
        nClientsInRun = nClients
        myplotter = Plotter()
        cy_PV = epics.PV('XCS:USER:SND:Y_CENTROID')
        while nClientsInRun > 0:
            md = mpidata()
            rank1 = md.recv()
            # Check if clients have completed runs
            if md.small.endrun:
                nClientsInRun -= 1
            else:
                cy_PV.put(md.lobeHeight)
                if rank1==1:
                    myplotter.update(md) # send data from clients to plotter

class Plotter:
    def __init__(self):
        self.nupdate = 0
        self.recent_nupdate = 0
        self.recent_nevents = 0
        self._recent_nevents = 0
        self.lastnevents = 0
        self.mask = None
        self.sumimg = None
        self.autocorrsum = None
        self.recent_autocorrsum = None
        self.recent_i0sum = None
        self.recent_avgautocorr = None
        self.deque = collections.deque(maxlen = deque_len)
        self.lasttime = None
        self.success_h = 0
        self.success_v = 0
        self.success_rh = 0
        self.success_rv = 0

    def update(self,md):
        self.nupdate += 1

        if self.autocorrsum is None:
            self.sumimg = md.sumimg
            self.mask = md.mask
            self.i0sum = np.float(md.i0sum)
            self.autocorrsum = md.autocorrsum
            self.recent_autocorrsum = np.copy(md.autocorrsum)
            self.recent_i0sum = np.copy(np.float(md.i0sum))
            self.nevents = md.small.nevents
           # self.recent_avgautocorr = np.zeros_like(self.recent_autocorrsum)
        else:
            # Summate aggregate values for averaging
            self.sumimg += md.sumimg         
            self.i0sum += md.i0sum
            self.autocorrsum += md.autocorrsum
            self.recent_autocorrsum += md.autocorrsum
            self.nevents += md.small.nevents
            self.recent_i0sum += md.i0sum
#         self.norm = md.norm
#         self.avgautocorr = self.autocorrsum/self.i0sum/self.norm

        self.avgautocorr = self.autocorrsum/self.i0sum
        shape = self.avgautocorr.shape

        # refresh the recent autocorrelation plot
        if self.nupdate%recent_updaterate == 0: # so that it runs on the first update and every `recent_updaterate` after 
            deque_dict = {}
            self.recent_avgautocorr = self.recent_autocorrsum/self.recent_i0sum
            if self.success_rv and self.success_rh:
                inst_contrast = np.mean([self.rzplotv.max()-self.rzplotv.min(), self.rzploth.max()-self.rzploth.min()])
                recent_autocorr_width_v = self.rvpx[2]
                recent_autocorr_width_h = self.rhpx[2]
            else:
                inst_contrast = np.nan
                recent_autocorr_width_v = np.nan
                recent_autocorr_width_h = np.nan
            if self.rvlobes and self.success_rv:
                lobeHeight = self.rvpx[3]
            else:
                lobeHeight = np.nan
            deque_dict = { # "instantaneous" snapshot of the average autocorrelation.  Goes into buffer with length `deque_len`. 
                'rxh' : self.rxh,
                'rxv' : self.rxv,
                'rzh' : self.rzh,
                'rzv' : self.rzv,
                'rxploth' : self.rxploth,
                'rxplotv' : self.rxplotv,
                'rzploth' : self.rzploth,
                'rzplotv' : self.rzplotv,
                'update no.' : self.recent_nupdate,
                'update time' : np.nan, # not yet implement (would be nice to plot time instead of update no.)
                'lobe height' : lobeHeight,
                'inst. contrast' : inst_contrast,
                'autocorr vertical width' : recent_autocorr_width_v,
                'autocorr horizontal width' : recent_autocorr_width_h
            }
            self.deque.append(deque_dict) 
            self.recent_autocorrsum = np.copy(md.autocorrsum) # reset aggregate values to latest
            self.recent_i0sum = np.copy(np.float(md.i0sum)) # reset aggregate values to latest
            self.recent_nevents = self._recent_nevents
            self._recent_nevents = 0 # reset recent update counter
            self.recent_nupdate += 1 
        
        # fit curves
        self.xh, self.zh, self.xploth, self.zploth, self.hlobes, self.hpx, self.success_h = safe_fit(self.avgautocorr[shape[0]//2, shape[1]//2-nn:shape[1]//2+nn])
        self.xv, self.zv, self.xplotv, self.zplotv, self.vlobes, self.vpx, self.success_v = safe_fit(self.avgautocorr[shape[0]//2-nn:shape[0]//2+nn,shape[1]//2])

        if self.recent_avgautocorr is None: # Should only happen before `recent_updaterate` updates have passed.
            self.recent_avgautocorr = np.copy(self.recent_autocorrsum/self.recent_i0sum)

        self.rxh, self.rzh, self.rxploth, self.rzploth, self.rhlobes, self.rhpx, self.success_rh = safe_fit(self.recent_avgautocorr[shape[0]//2,
                                                                                                                                    shape[1]//2-nn:shape[1]//2+nn])
        self.rxv, self.rzv, self.rxplotv, self.rzplotv, self.rvlobes, self.rvpx, self.success_rv = safe_fit(self.recent_avgautocorr[shape[0]//2-nn:shape[0]//2+nn,
                                                                                                                                    shape[1]//2])
            
        # remove delta fn in middle of autocorrelation
        self.avgautocorr[shape[0]//2-1:shape[0]//2+2, shape[1]//2] = np.nan
        self.avgautocorr[shape[0]//2, shape[1]//2-1:shape[1]//2+2] = np.nan
        self.recent_avgautocorr[shape[0]//2-1:shape[0]//2+2, shape[1]//2] = np.nan
        self.recent_avgautocorr[shape[0]//2, shape[1]//2-1:shape[1]//2+2] = np.nan
 
        # Print update summary and create plots
        if self.nupdate%updaterate == 0:
            print('Client update:', self.nupdate)
            print('Master received total events:', self.nevents)
            if self.lasttime is not None:
                print('Rate:', np.around((self.nevents-self.lastnevents)/(time.time()-self.lasttime),3), 'events/sec')
            print('-'*40)
 
            self.lasttime = time.time()
            self._recent_nevents += self.nevents - self.lastnevents
            self.lastnevents = self.nevents
            
            # assemble recent contrast lineplot 
            recent_contrasts = np.zeros(deque_len)
            recent_updates = np.zeros(deque_len)
            recent_lobeHeight = np.zeros(deque_len)
            recent_autoCorrWidthVertical = np.zeros(deque_len)
            recent_autoCorrWidthHorizontal = np.zeros(deque_len)
            for i in range(len(self.deque)):
                recent_contrasts[i] = self.deque[i]['inst. contrast']
                recent_lobeHeight[i] = self.deque[i]['lobe height']
                recent_autoCorrWidthVertical[i] = self.deque[i]['autocorr vertical width']
                recent_autoCorrWidthHorizontal[i] = self.deque[i]['autocorr horizontal width']
                recent_updates[i] = self.deque[i]['update no.']
            for x in [recent_contrasts,
                      recent_autoCorrWidthVertical,
                      recent_autoCorrWidthHorizontal,
                      recent_lobeHeight,
                      recent_updates]:
                x[1:][recent_updates[1:]==0] = np.nan
            recent_updates += np.ones_like(recent_updates)

            # create plots
            avgautocorr_plot = Image(self.nupdate,
                                     "total avg autocor",
                                     self.avgautocorr[shape[0]//2-nn:shape[0]//2+nn, shape[1]//2-nn:shape[1]//2+nn],
                                     xlabel = 'horizontal (pixel)',
                                     ylabel = 'vertical (pixel)',
                                     aspect_ratio = 1)
            avgimage_plot = Image(self.nupdate,
                                     "total avg image",
                                     self.sumimg*(1+2*self.mask),
                                     xlabel = 'horizontal (pixel)',
                                     ylabel = 'vertical (pixel)',
                                     aspect_ratio = 1)
            autocorr_lineplot = XYPlot(self.nupdate,
                                       "total avg autocorr line plot",
                                       [self.xh, self.xv, self.xploth, self.xplotv],
                                       [self.zh, self.zv, self.zploth, self.zplotv],
                                       formats = ['bo', 'ro', 'b-', 'r-'],
                                       leg_label = ['hor', 'ver', 'hor_fit', 'ver_fit'],
                                       xlabel = 'distance (pixel)',
                                       ylabel = 'correlation')
            recent_autocorr_plot = Image(self.recent_nupdate,
                                         "recent avg autocor: last {} events".format(self.recent_nevents),
                                         self.recent_avgautocorr[shape[0]//2-nn:shape[0]//2+nn, shape[1]//2-nn:shape[1]//2+nn],
                                         xlabel = 'horizontal (pixel)',
                                         ylabel = 'vertical (pixel)',
                                         aspect_ratio = 1)
            recent_autocorr_lineplot = XYPlot(self.recent_nupdate,
                                              "recent avg autocorr line plot: last {} events".format(self.recent_nevents),
                                              [self.rxh, self.rxv, self.rxploth, self.rxplotv],
                                              [self.rzh, self.rzv, self.rzploth, self.rzplotv],
                                              formats = ['bo', 'ro', 'b-', 'r-'],
                                              leg_label = ['hor', 'ver', 'hor_fit', 'ver_fit'],
                                              xlabel = 'distance (pixel)',
                                              ylabel = 'correlation')
            recent_contrast_lineplot = XYPlot(self.recent_nupdate,
                                              "recent contrast line plot: last {} events".format(self.recent_nevents),
                                              [recent_updates, recent_updates],
                                              [recent_contrasts, recent_contrasts], #idk how else to get marker and line
                                              formats = ['s', 'r-'],
                                              xlabel = 'updates',
                                              ylabel = 'relative contrast')
            recent_lobeHeight_lineplot = XYPlot(self.recent_nupdate,
                                              "recent lobe height line plot: last {} events".format(self.recent_nevents),
                                              [recent_updates, recent_updates],
                                              [recent_lobeHeight, recent_lobeHeight], #idk how else to get marker and line
                                              formats = ['s', 'r-'],
                                              xlabel = 'updates',
                                              ylabel = 'lobe height')
            recent_autoCorrWidth_lineplot = XYPlot(self.recent_nupdate,
                                              "recent autocorrelation width line plot: last {} events".format(self.recent_nevents),
                                                [recent_updates, recent_updates, recent_updates, recent_updates],
                                                [recent_autoCorrWidthVertical,
                                                 recent_autoCorrWidthVertical,
                                                 recent_autoCorrWidthHorizontal,
                                                 recent_autoCorrWidthHorizontal], #idk how else to get marker and line
                                                   formats = ['s', 'r-', 's', 'b-'],
                                                   xlabel = 'updates',
                                                   ylabel = 'Autocorrelation Width')

            autocorr_plot = MultiPlot(self.nupdate, "autocorr plot", ncols=4)
            autocorr_plot.add(avgautocorr_plot)
            autocorr_plot.add(recent_autocorr_plot)
            autocorr_plot.add(avgimage_plot)
            autocorr_plot.add(recent_contrast_lineplot)
            autocorr_plot.add(autocorr_lineplot)
            autocorr_plot.add(recent_autocorr_lineplot)
            autocorr_plot.add(recent_lobeHeight_lineplot)
            autocorr_plot.add(recent_autoCorrWidth_lineplot)

            publish.send('AVGAUTOCORR', autocorr_plot)
