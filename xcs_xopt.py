import numpy as np
import scipy.optimize as optimize
import json
import sys
import time
from datetime import datetime
import os
import socket
import logging
from ophyd import EpicsSignal

from pcdsdevices.device_types import Newport, IMS
from pcdsdevices.sim import FastMotor
#from xcs.db import RE, bpp, bps, seq, daq
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.linalg import hadamard

from bluesky.plans import list_scan, scan
from databroker import Broker

from ophyd.device import Device
from ophyd import EpicsMotor
from ophyd.status import StatusBase, SubscriptionStatus
from threading import RLock
from ophyd.status import DeviceStatus
from ophyd.signal import EpicsSignalRO, Signal
from ophyd.device import Component as Cpt
from pcdsdevices.signal import AvgSignal
from hxrsnd.sndsystem import SplitAndDelay

from xopt import Xopt, Evaluator
import torch
import gpytorch
import botorch
from xopt import VOCS
from xopt.generators.bayesian import ExpectedImprovementGenerator, UpperConfidenceBoundGenerator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
import pandas as pd

#### would be good to generalize this
savepath = '/cds/home/opr/xcsopr/experiments/xcsx1015123/optimization_data/'

#### Was messing around with trying to capture two signals in order to output a normalized signal
#### Not currently being used although these are being initialized.
class AvgNormSignal(AvgSignal):
    def __init__(self, signal, norm_signal, averages, duration, *, name, parent=None, **kwargs):
        super().__init__(signal, averages, duration, name=name, parent=parent, **kwargs)
        if isinstance(signal, str):
            signal = getattr(parent, signal)
        self.raw_sig = signal
        self.norm_sig = norm_signal
        self._lock = RLock()
        self.averages = averages
        self.duration = duration
        self.raw_sig.subscribe(self._update_avg)
        self.norm_sig.subscribe(self._update_norm)

    def reset_buffer(self) -> None:
        with self._lock:
            self.index = 0
            self.norm_index = 0
            self.values = np.empty(self._avg)
            self.norm_values = np.empty(self._avg)
            self.values.fill(np.nan)
            self.norm_values.fill(np.nan)

    def _update_norm(self, *args, value: float, **lwargs) -> None:
        with self._lock:
            self.norm_values[self.norm_index] = value
            self.norm_index = (self.norm_index + 1) % len(self.norm_values)

    def _update_avg(self, *args, value: float, **kwargs) -> None:
        with self._lock:
            self.values[self.index] = value
            self.index = (self.index + 1) % len(self.values)
            self.put(np.nanmean(self.values)/np.nanmean(self.norm_values))


class User():
    
    
    def __init__(self):

        # various diode signals relevant to split and delay
        dd = EpicsSignalRO('XCS:SND:DIO:AMPL_12')
        t1_dh = EpicsSignalRO('XCS:SND:DIO:AMPL_11')
        t4_dh = EpicsSignalRO('XCS:SND:DIO:AMPL_15')
        do = EpicsSignalRO('XCS:SND:DIO:AMPL_14')
        di = EpicsSignalRO('XCS:SND:DIO:AMPL_10')
        dci = EpicsSignalRO('XCS:SND:DIO:AMPL_13')
        dcc = EpicsSignalRO('XCS:SND:DIO:AMPL_8')
        dco = EpicsSignalRO('XCS:SND:DIO:AMPL_9')
        ipm4 = EpicsSignalRO('XCS:SB1:BMMON:SUM')
        ipm5 = EpicsSignalRO('XCS:SB2:BMMON:SUM')

        #### was trying to set up a separate RE from hutch-python to do scan and fit,
        #### but will restrict this to the GUI now
        # self.db = Broker.named('temp')
        #RE.subscribe(self.db.insert)
        self.snd = SplitAndDelay('XCS:SND', name='snd')
        self.fast_motor1 = FastMotor(name='fast_motor1')
        self.fast_motor2 = FastMotor(name='fast_motor2')

        #### AvgSignals that are useful for SND alignment
        # start with 120 samples in 1 second duration for signals
        self.dd_signal = AvgSignal(dd,120,1,name='dd_signal')
        self.t1_dh_signal = AvgSignal(t1_dh,120,1,name='t1_dh_signal')
        self.t4_dh_signal = AvgSignal(t4_dh,120,1,name='t4_dh_signal')
        self.do_signal = AvgSignal(do,120,1,name='do_signal')
        self.di_signal = AvgSignal(di,120,1,name='di_signal')
        self.dci_signal = AvgSignal(dci,120,1,name='dci_signal')
        self.dcc_signal = AvgSignal(dcc,120,1,name='dcc_signal')
        self.dco_signal = AvgSignal(dco,120,1,name='dco_signal')
        self.cx_signal = AvgSignal(EpicsSignalRO('XCS:USER:SND:X_CENTROID'),120,1,name='cx_signal')
        self.cy_signal = AvgSignal(EpicsSignalRO('XCS:USER:SND:Y_CENTROID'),120,1,name='cy_signal')
        self.wx_signal = AvgSignal(EpicsSignalRO('XCS:USER:SND:X_WIDTH'),120,1,name='wx_signal')
        self.wy_signal = AvgSignal(EpicsSignalRO('XCS:USER:SND:Y_WIDTH'),120,1,name='wy_signal')
        self.intensity_signal = AvgSignal(EpicsSignalRO('XCS:USER:SND:INTENSITY'),120,1,name='intensity_signal')
        self.ipm4_signal = AvgSignal(ipm4,120,1,name='ipm4_signal')
        self.ipm5_signal = AvgSignal(ipm5,120,1,name='ipm5_signal')

        #### Not using the AvgNormSignal, that I don't think was quite working...
        # self.t1_dh_norm = AvgNormSignal(t1_dh,ipm4,120,1,name='t1_dh_norm')
        # self.dd_norm = AvgNormSignal(dd,t1_dh,120,1,name='dd_norm')
        # self.ipm5_norm = AvgNormSignal(ipm5,ipm4,120,1,name='ipm5_norm')
        # self.dcc_norm = AvgNormSignal(dcc,dci,120,1,name='dcc_norm')
        # self.t4_dh_norm = AvgNormSignal(t4_dh, dd, 120,1,name='t4_dh_norm')
        # self.dco_norm = AvgNormSignal(dco,dcc,120,1,name='dco_norm')

        #### SnD motors
        self.snd_t1_th1 = EpicsMotor('XCS:SND:T1:TH1',name='snd_t1_th1')
        self.snd_t1_th2 = EpicsMotor('XCS:SND:T1:TH2',name='snd_t1_th2')
        self.snd_t2_th = EpicsMotor('XCS:SND:T2:TH',name='snd_t2_th')
        self.snd_t3_th = EpicsMotor('XCS:SND:T3:TH',name='snd_t3_th')
        self.snd_t4_th1 = EpicsMotor('XCS:SND:T4:TH1',name='snd_t4_th1')
        self.snd_t4_th2 = EpicsMotor('XCS:SND:T4:TH2',name='snd_t4_th2')
        self.snd_t1_chi1 = EpicsMotor('XCS:SND:T1:CHI1',name='snd_t1_chi1')
        self.snd_t1_chi2 = EpicsMotor('XCS:SND:T1:CHI2',name='snd_t1_chi2')
        self.snd_t4_chi1 = EpicsMotor('XCS:SND:T4:CHI1',name='snd_t4_chi1')
        self.snd_t4_chi2 = EpicsMotor('XCS:SND:T4:CHI2',name='snd_t4_chi2')

        # start up with simulated motors
        self.input_list = [self.fast_motor1,self.fast_motor2]
        self.name_list = [motor.name for motor in self.input_list]

        #### set up Xopt parameters
        self.start_pos = {}
        self.pos_range = {}
        self.motor_dict = {}
        for num, name in enumerate(self.name_list):
            self.start_pos[name] = 0.0
            self.pos_range[name] = 50e-6*180/np.pi
            self.motor_dict[name] = self.input_list[num]


        self.vocs = None
        self.X = None
        self.generator = None
        self.evaluator = None
        self.num = 0

        self.detailed_output = {}
        self.detailed_output['BPE'] = np.array([])
        self.detailed_output['Intensity'] = np.array([])

        self.x_target = 0.0
        self.y_target = 0.0
        self.n_init = 0
        self.intensity_scale = 1e-4

    def set_motors(self, motion_range=50e-6, sim=False):
        """
        Set inputs to actual SnD motors
        Parameters
        ----------
        motion_range: float
            Xopt motion range for angular motions (radians)
        sim: bool
            If True, use simulated motors, otherwise use real motors

        Returns
        -------

        """
        if sim:
            self.input_list = [self.fast_motor1,self.fast_motor2]
        else:
            self.input_list = [self.snd.t1.th1,self.snd.t1.chi1,self.snd.t1.th2,
                self.snd.t1.chi2,self.snd.t4.th2,self.snd.t4.chi2,
                self.snd.t4.th1,self.snd.t4.chi1]


        self.name_list = [motor.name for motor in self.input_list]

        self.start_pos = {}
        self.pos_range = {}
        self.motor_dict = {}
        for num, name in enumerate(self.name_list):
            self.start_pos[name] = 0.0
            self.pos_range[name] = motion_range*180/np.pi
            self.motor_dict[name] = self.input_list[num]


    def set_target(self):
        # function to call when setting the centroid target based on cc branch
        # position
        signal_list = [self.cx_signal,self.cy_signal]
        status_list = []
        print('sending triggers') 
        for signal in signal_list:
            status = signal.trigger()
            status_list.append(status)

        done_reading = False
        print('averaging...')
        while not done_reading:
            time.sleep(0.1)
            done_reading = all([status.done for status in status_list])

        self.x_target = self.cx_signal.get()
        self.y_target = self.cy_signal.get()

    def initialize_turbo(self,pos_range=None,n_init=64, scale=1):
        #start_vars = [motor.wm() for motor in self.input_list]
        #start_values = dict(zip(start_names,start_vars))

        self.n_init = n_init
        self.set_vocs(pos_range)
        self.num = 0
        self.intensity_scale = scale

        self.evaluator = Evaluator(function=self.eval_function)
        self.generator = ExpectedImprovementGenerator(vocs=self.vocs,
                turbo_controller="optimize")
        self.X = Xopt(evaluator=self.evaluator, generator=self.generator, vocs=self.vocs)

    def set_vocs(self, motion_range=None,low=0,high=1):

        for motor in self.input_list:
            self.start_pos[motor.name] = motor.wm()


        if motion_range is not None:
            for name in self.name_list:
                self.pos_range[name] = motion_range
            #self.pos_range = pos_range
        else:
            for name in self.name_list:
                self.pos_range[name] = 50e-6*180/np.pi

        low = low
        high = high
        vocs_variables = {}
        #for key in start_values.keys():
        #    vocs_variables[key] = [low, high]

        for name in self.name_list:
            #low = self.start_pos[name] - self.pos_range[name]/2
            #high = self.start_pos[name] + self.pos_range[name]/2
            vocs_variables[name] = [low, high]
        self.vocs = VOCS(variables=vocs_variables, objectives={"f": "MINIMIZE"})

    def eval_function_transformed(self, input_dict: dict) -> dict:
        """
        This will need to be tested. This is based on the work published with Aashwin
        Parameters
        ----------
        input_dict

        Returns
        -------

        """
        #keys = [f"x{i}" for i in range(1, 9)]
        keys = self.name_list
        v = np.array([input_dict[key] for key in keys])
        H8 = hadamard(8)
        R = H8 / np.sqrt(8)
        out = R @ v + 0.5
        Xinp = np.expand_dims(out, axis=0)
        data = self.get_snd_outputs(Xinp)

        if np.isnan(data['cx']):
            bpe = np.nan
        else:

            bpe = np.sqrt((data['cx']-self.x_target)**2+(data['cy']-self.y_target)**2)
        self.detailed_output['BPE'][self.num] = bpe
        self.detailed_output['Intensity'][self.num] = data['intensity']
        self.num += 1
        print('BPE: {}'.format(bpe))
        print('Intensity: {}'.format(data['intensity']))
        #result = np.log(bpe) - np.log(data['intensity']*100)
        #result = -np.log(data['intensity']*100)
        # we used this one first
        #result = np.log(bpe)
        # next objective function
        #result = -np.log(data['intensity'])
        #if np.isnan(bpe):
        #    result = np.log(500)-np.log(data['intensity'])
        #else:
        #    result = np.log(bpe) - np.log(data['intensity'])
        ##result = np.log(bpe) - self.intensity_scale * np.log(data['intensity'])
        #if np.isnan(result):
        #    result = np.nan
        result = -self.intensity_scale * data['intensity'] + bpe/350

        return {"f": result}



    def initialize_BO(self, pos_range=None, n_init=64, scale=1e-4):

        self.n_init = n_init
        self.num = 0
        self.intensity_scale = scale
        self.set_vocs(pos_range)
        self.evaluator = Evaluator(function=self.eval_function)
        #self.generator = ExpectedImprovementGenerator(vocs=self.vocs)
        self.generator = UpperConfidenceBoundGenerator(vocs=self.vocs)
        self.X = Xopt(evaluator=self.evaluator, generator=self.generator, vocs=self.vocs)

    def initialize_BO_transformed(self, pos_range=None, n_init=64, scale=1e-4):

        self.n_init = n_init
        self.num = 0
        self.intensity_scale = scale
        self.set_vocs(pos_range,low=-0.5,high=0.5)
        self.evaluator = Evaluator(function=self.eval_function_transformed)
        #self.generator = ExpectedImprovementGenerator(vocs=self.vocs)
        self.generator = UpperConfidenceBoundGenerator(vocs=self.vocs)
        self.X = Xopt(evaluator=self.evaluator, generator=self.generator, vocs=self.vocs)


    def run_BO(self,num_iter=150,seed=42):
        self.detailed_output['BPE'] = np.zeros(self.n_init+num_iter)
        self.detailed_output['Intensity'] = np.zeros(self.n_init+num_iter)

        sampler = qmc.LatinHypercube(d=len(self.name_list),seed=seed)
        xs = sampler.random(n=self.n_init)
        init_dict = {}
        for num, name in enumerate(self.name_list):
            init_dict[name] = xs[:,num]
        init_samples = pd.DataFrame(init_dict)
        self.X.evaluate_data(init_samples)

        #self.X.random_evaluate(self.n_init)
        for i in range(num_iter):
            print(i)
            self.X.step()
            self.X.generator.beta += 0.2
            print('beta={}'.format(self.X.generator.beta))
        y1 = self.X.generator.data['f']
        y1_maxs = np.maximum.accumulate(y1)

        return self.X.data

    def run_turbo(self,num_iter=150):
        self.detailed_output['BPE'] = np.zeros(self.n_init+num_iter)
        self.detailed_output['Intensity'] = np.zeros(self.n_init+num_iter)

        # random sampling
        sampler = qmc.LatinHypercube(d=len(self.name_list))
        xs = sampler.random(n=self.n_init)
        init_dict = {}
        for num, name in enumerate(self.name_list):
            init_dict[name] = xs[:,num]
        init_samples = pd.DataFrame(init_dict)
        self.X.evaluate_data(init_samples)

        print('done with random sampling')
        self.X.generator.train_model()
        self.X.generator.turbo_controller.update_state(self.X.generator.data)
        self.X.generator.turbo_controller.get_trust_region(self.X.generator.model)

        for i in range(num_iter):
            #if i % 10 == 0:
            print(f"Step: {i+1}")
            model = self.X.generator.train_model()
            trust_region = self.X.generator.turbo_controller.get_trust_region(self.generator.model).squeeze()
            scale_factor = self.X.generator.turbo_controller.length
            region_width = trust_region[1] - trust_region[0]
            best_value = self.X.generator.turbo_controller.best_value
            n_successes = self.X.generator.turbo_controller.success_counter
            n_failures = self.X.generator.turbo_controller.failure_counter
            acq = self.X.generator.get_acquisition(model)
            self.X.step()

        return self.X.data

    def eval_function(self,input_dict: dict) -> dict:
        data = self.get_snd_outputs(input_dict) 

        # beam position error

        if np.isnan(data['cx']):
            bpe = np.nan
        else:

            bpe = np.sqrt((data['cx']-self.x_target)**2+(data['cy']-self.y_target)**2)
        self.detailed_output['BPE'][self.num] = bpe
        self.detailed_output['Intensity'][self.num] = data['intensity']
        self.num += 1
        print('BPE: {}'.format(bpe))
        print('Intensity: {}'.format(data['intensity']))
        #result = np.log(bpe) - np.log(data['intensity']*100)
        #result = -np.log(data['intensity']*100)
        # we used this one first
        #result = np.log(bpe)
        # next objective function
        #result = -np.log(data['intensity'])
        #if np.isnan(bpe):
        #    result = np.log(500)-np.log(data['intensity'])
        #else:
        #    result = np.log(bpe) - np.log(data['intensity'])
        ##result = np.log(bpe) - self.intensity_scale * np.log(data['intensity'])
        #if np.isnan(result):
        #    result = np.nan
        result = -self.intensity_scale * data['intensity'] + bpe/350

        return {"f": result}


    def get_snd_outputs(self, inputs):
        # move to inputs
        #motor_list = [self.fast_motor]
        #intensity_list = [self.t1_dh_signal, self.dd_signal, self.t4_dh_signal, self.do_signal]
        #        self.intensity_signal]
        #intensity_list = [self.t1_dh_signal, self.dd_signal, self.t4_dh_signal]
        intensity_list = [self.do_signal]

        centroid_list = [self.cx_signal,self.cy_signal,self.wx_signal,self.wy_signal]
        signal_list = intensity_list + centroid_list
        #signal_list = intensity_list + centroid_list
        #other_signals = [self.ipm4_signal,self.di_signal,self.intensity_signal]
        #signal_list = signal_list + other_signals
        status_list = []

        # calculate where to move motors based on range and starting position
        #new_positions = (inputs*self.pos_range - self.pos_range/2) + self.start_pos

        #for num, motor in enumerate(motor_list):
        #    status = motor.move(new_positions[num])
        #    status_list.append(status)
        for key, value in inputs.items():
            new_pos = (value*self.pos_range[key] - self.pos_range[key]/2) + self.start_pos[key]
            time.sleep(.1)
            status = self.motor_dict[key].move(new_pos)
            status_list.append(status)
        done_moving = False
        while not done_moving:
            time.sleep(0.1)
            done_moving = all([status.done for status in status_list])

        status_list = []
        print('sending triggers') 
        for signal in signal_list:
            status = signal.trigger()
            status_list.append(status)

        done_reading = False
        print('averaging...')
        while not done_reading:
            time.sleep(0.1)
            done_reading = all([status.done for status in status_list])

        signal_sum = 0
        for signal in intensity_list:
            temp = signal.get()
            if temp<=0:
                temp = 1e-3
            signal_sum += temp
        #norm = self.di_signal.get()
        norm = self.ipm4_signal.get()
        if norm <= 1000:
            self.get_snd_outputs(inputs)
            #norm = 1
        #signal_sum /= norm
        # the following is pre-normalized
        ### it looks like this was expecting to be divided by the normalization signal, but never happened,
        ### so seems like an error. It may be because we were using pink beam last time. Need to revisit.
        signal_sum += self.intensity_signal.get()*self.ipm4_signal.get()/50

        if self.intensity_signal.get()*self.ipm4_signal.get()/50 < 1000:
            cx = np.nan
            cy = np.nan
            wx = np.nan
            wy = np.nan
        else:
            cx = np.nanmean(self.cx_signal.values)
            cy = np.nanmean(self.cy_signal.values)
            wx = np.nanmean(self.wx_signal.values)
            wy = np.nanmean(self.wy_signal.values)


        # might need to use the following instead of "get"
        #signal_sum = np.nanmean(self.intensity_signal.values)
        #if not np.isnan(zyla_intensity):
        #    signal_sum += zyla_intensity
        #cx = self.cx_signal.get()
        #cy = self.cy_signal.get()
        #wx = self.wx_signal.get()
        #wy = self.wy_signal.get()
        #cx = np.nanmean(self.cx_signal.values)
        #cy = np.nanmean(self.cy_signal.values)
        #wx = np.nanmean(self.wx_signal.values)
        #wy = np.nanmean(self.wy_signal.values)

        out_dict = {}
        out_dict['intensity'] = signal_sum
        out_dict['cx'] = cx
        out_dict['cy'] = cy
        out_dict['wx'] = wx
        out_dict['wy'] = wy

        return out_dict
    def get_optimum_details(self,plot=True, move_to_optimum=True):
          """
          Takes Xopt object, finds the sample with the minimum and prints the settings 
          of this sample, the beam position error and the inensity at this setting.
          """
          timestamp = str(int(datetime.now().timestamp())) 
          filename = "optimize_output_{}".format(timestamp)
          
          min_idx = np.argmin(self.X.generator.data["f"])
          min_val = self.X.generator.data["f"][min_idx]
          #X_min = [X.generator.data["x1"][min_idx],
          #        X.generator.data["x2"][min_idx],
          #          X.generator.data["x3"][min_idx],
          #        X.generator.data["x4"][min_idx],
          #        X.generator.data["x5"][min_idx],
          #        X.generator.data["x6"][min_idx],
          #          X.generator.data["x7"][min_idx],
          #          X.generator.data["x8"][min_idx]
          #        ]
          X_min = self.X.generator.data.iloc[min_idx]
          inputs = np.array(X_min)
          inputs = inputs[np.newaxis,:]
          #outs = get_snd_outputs_detailed(inputs)
          if move_to_optimum:
            print('moving to optimum')
            outs = self.get_snd_outputs(inputs)
          bpe = self.detailed_output['BPE']
          bpe_out = bpe[min_idx]
          intensity = self.detailed_output['Intensity']
          intensity_out = intensity[min_idx]

          self.X.generator.data.to_csv(savepath+filename+'.csv',index=False)
          np.savez(savepath+filename+'.npz',bpe=bpe,intensity=intensity)


          print("Optimum Inputs: ", X_min)
          #print("BPE: ", outs[0][0].item())
          #print("Intensity:", outs[0][1].item())
          print("BPE: ", bpe_out)
          print("Intensity:", intensity_out)



          if plot:
            y1 = self.X.generator.data["f"]
            y1_mins = np.minimum.accumulate(y1)

            idx = np.arange(len(y1_mins))
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 7))
            ax0.plot(idx, y1_mins,'k')
            ax0.grid()
            ax0.set_xlabel("Sample Number")
            ax0.set_ylabel("Objective")
            ax1.plot(idx[-50:], y1_mins[-50:],'k')
            ax1.grid()
            ax1.set_ylabel("Objective");
            #plt.savefig('objective.pdf', bbox_inches='tight')
            plt.show()

    def move_to_start(self):
        print('moving to start')
        for key in self.start_pos.keys():
            self.motor_dict[key].umv(self.start_pos[key])

    #### This should happen in the GUI
    def dscan_and_fit(self, signal, motor, start, stop, num, move_to_peak=False, norm_signal=None):

        # set up and run the scan
        curr_pos = motor.wm()
        scan_start = start + curr_pos
        scan_stop = stop + curr_pos
        if norm_signal is not None:
            signals = [signal, norm_signal]
        else:
            signals = [signal]
        RE(scan(signals,motor, scan_start, scan_stop, num=num))

        # get the data for the fit
        header = self.db[-1]
        df = header.table()
        if norm_signal is not None:
            sig = df[signal.name]
            norm = df[norm_signal.name]
            sigma = 1/np.abs(norm)*np.sqrt(np.abs(sig) + np.abs(sig)**2/np.abs(norm))
            plot_signal = df[signal.name]/df[norm_signal.name]
            signal_label = '{}/{}'.format(signal.name,norm_signal.name)

        else:
            sig = df[signal.name]
            sigma = np.sqrt(np.abs(sig))
            plot_signal = df[signal.name]
            signal_label = signal.name
       
        #cx,sx = Util.gaussian_stats(df[motor.name],plot_signal,sigma=sigma)
        cx,sx = Util.gaussian_stats(df[motor.name],plot_signal)
        fit = Util.fit_gaussian(df[motor.name],cx,sx)

        plt.figure()
        plt.plot(df[motor.name],Util.normalize_trace(plot_signal),label='data')
        plt.plot(df[motor.name],fit,label='fit')
        plt.legend()
        plt.xlabel(motor.name)
        plt.ylabel(signal_label)

        print(cx)

        if move_to_peak:
            print('moving to peak')
            motor.mv(cx)
        else:
            print('moving to starting position')
            motor.mv(curr_pos)

       
class Util:

    @staticmethod
    def fit_gaussian(x, x0, w):
        """
        Method for fitting to a Gaussian function. This method is a parameter to Scipy's optimize.curve_fit routine.
        :param x: array_like
            Copied from Scipy docs: "The independent variable where the data is measured. Should usually be an
            M-length sequence or an (k,M)-shaped array for functions with k predictors, but can actually be any
            object." Units are meters.
        :param x0: float
            Initial guess for beam center (m).
        :param w: float
            Initial guess for gaussian sigma (m).
        :return: array_like with same shape as x
            Function evaluated at all points in x.
        """
        # just return an array evaluating the Gaussian function based on input parameters.
        if w == 0:
            return np.zeros_like(x)
        else:
            return np.exp(-((x - x0) ** 2 / (2 * w ** 2)))

    @staticmethod
    def gaussian_stats(x_data, y_data, thresh=0.1,sigma=None):

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
            px, pcovx = optimize.curve_fit(Util.fit_gaussian, x_data[mask], y_norm[mask],p0=guess,sigma=sigma)
            sx = px[1]
        except:
            print('Fit failed. Using second moment for width.')

        return cx, sx

    @staticmethod
    def normalize_trace(y_data):

        norm_data = (y_data - np.min(y_data))/(np.max(y_data) - np.min(y_data))

        return norm_data

    @staticmethod
    def threshold_array(array_in, frac):
        """Method for thresholding an array, useful for calculating center of mass
        :param array_in: array-like
            can be any shape array
        :param frac: float
            threshold fraction of image maximum
        :return array_out: array-like
            thresholded array, same shape as array_in
        """

        # make sure the image is not complex
        array_out = np.abs(array_in)

        # subtract minimum/background
        array_out -= np.min(array_out)

        # get thresholding level
        thresh = np.max(array_out) * frac
        # subtract threshold level
        array_out = array_out - thresh
        # set anything below threshold (now 0) to zero
        array_out[array_out < 0] = 0

        return array_out


