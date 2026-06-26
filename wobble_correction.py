import pandas as pd
import epics
from ophyd.signal import EpicsSignalRO
import numpy as np
import time
import signal
import sys

# load lookup table
# this will have 6 columns, TBD number of rows depending on the L motion resolution we choose
# t1_L | t1_th2_offset | t1_chi2_offset | t4_L | t4_th2_offset | t4_chi2_offset
lut = pd.read_csv('wobble_lookup.csv', encoding='utf-8-sig')
t1_L = EpicsSignalRO('XCS:SND:T1:L.RBV')
t4_L = EpicsSignalRO('XCS:SND:T4:L.RBV')
t1_th2_offset = epics.PV('XCS:USER:SND:T1_TH2_OFFSET')
t1_chi2_offset = epics.PV('XCS:USER:SND:T1_CHI2_OFFSET')
t4_th2_offset = epics.PV('XCS:USER:SND:T4_TH2_OFFSET')
t4_chi2_offset = epics.PV('XCS:USER:SND:T4_CHI2_OFFSET')

offset_pvs = [t1_th2_offset, t1_chi2_offset, t4_th2_offset, t4_chi2_offset]

t1_L_lut = lut['t1_L']
t4_L_lut = lut['t4_L']
t1_th2_lut = lut['t1_th2']
t1_chi2_lut = lut['t1_chi2']
t4_th2_lut = lut['t4_th2']
t4_chi2_lut = lut['t4_chi2']


def shutdown(signum, frame):
    print('\nShutting down — zeroing offsets...')
    for pv in offset_pvs:
        pv.put(0)
    print('Done.')
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# startup confirmation
print('Wobble correction starting')
print(f'  LUT range: t1_L [{t1_L_lut.iloc[0]:.3f}, {t1_L_lut.iloc[-1]:.3f}], '
      f't4_L [{t4_L_lut.iloc[0]:.3f}, {t4_L_lut.iloc[-1]:.3f}]')
print(f'  t1_L = {t1_L.get():.4f}, t4_L = {t4_L.get():.4f}')
print('  Offset PVs:', [pv.pvname for pv in offset_pvs])
print('Running (Ctrl+C to stop)...')

while True:
    try:
        t1_L_curr = t1_L.get()
        t4_L_curr = t4_L.get()
        if t1_L_curr is None or t4_L_curr is None:
            print('WARNING: got None from L readback, skipping')
            time.sleep(.1)
            continue
        t1_th2 = np.interp(t1_L_curr, t1_L_lut, t1_th2_lut)
        t1_chi2 = np.interp(t1_L_curr, t1_L_lut, t1_chi2_lut)
        t4_th2 = np.interp(t4_L_curr, t4_L_lut, t4_th2_lut)
        t4_chi2 = np.interp(t4_L_curr, t4_L_lut, t4_chi2_lut)
        t1_th2_offset.put(t1_th2)
        t1_chi2_offset.put(t1_chi2)
        t4_th2_offset.put(t4_th2)
        t4_chi2_offset.put(t4_chi2)
    except Exception as e:
        print(f'WARNING: {e}')
    time.sleep(.1)
