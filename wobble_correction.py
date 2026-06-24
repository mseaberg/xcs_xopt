import pandas as pd
import epics
from ophyd.signal import EpicsSignalRO
import numpy as np
import time

# load lookup table
# this will have 6 columns, TBD number of rows depending on the L motion resolution we choose
# t1_L | t1_th2_offset | t1_chi2_offset | t4_L | t4_th2_offset | t4_chi2_offset
lut = pd.read_csv('wobble_lookup.csv')
t1_L = EpicsSignalRO('XCS:SND:T1:L.RBV', auto_monitor=True)
t4_L = EpicsSignalRO('XCS:SND:T4:L.RBV', auto_monitor=True)
t1_th2_offset = epics.PV('XCS:USER:SND:T1_TH2_OFFSET')
t1_chi2_offset = epics.PV('XCS:USER:SND:T1_CHI2_OFFSET')
t4_th2_offset = epics.PV('XCS:USER:SND:T4_TH2_OFFSET')
t4_chi2_offset = epics.PV('XCS:USER:SND:T4_CHI2_OFFSET')

t1_L_lut = lut['t1_L']
t4_L_lut = lut['t4_L']
t1_th2_lut = lut['t1_th2']
t1_chi2_lut = lut['t1_chi2']
t4_th2_lut = lut['t4_th2']
t4_chi2_lut = lut['t4_chi2']


while True:
    t1_th2 = np.interp(t1_L.value, t1_L_lut, t1_th2_lut)
    t1_chi2 = np.interp(t1_L.value, t1_L_lut, t1_chi2_lut)
    t4_th2 = np.interp(t4_L.value, t4_L_lut, t4_th2_lut)
    t4_chi2 = np.interp(t4_L.value, t4_L_lut, t4_chi2_lut)
    t1_th2_offset.put(t1_th2)
    t1_chi2_offset.put(t1_chi2)
    t4_th2_offset.put(t4_th2)
    t4_chi2_offset.put(t4_chi2)
    time.sleep(.1)
#   check t1_L and t4_L positions, set offsets to PVs based on lookup table
#   update at ~1Hz (sleep or similar)