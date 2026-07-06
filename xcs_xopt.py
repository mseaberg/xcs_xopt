import numpy as np
import scipy.optimize as optimize
import time
from threading import RLock

import matplotlib.pyplot as plt

from ophyd import EpicsMotor
from ophyd.signal import EpicsSignalRO
from pcdsdevices.sim import FastMotor
from pcdsdevices.signal import AvgSignal
from hxrsnd.sndsystem import SplitAndDelay

from bluesky.plans import scan

from snd_optimizer import SnDBackend, SnDOptimizer

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


class HardwareBackend(SnDBackend):
    """SnDBackend that drives the real SnD motors and reads beamline signals.

    The diode / centroid signals live on the `User` object; the backend
    keeps a reference to it for measurement. The normalized to physical input
    conversion is handled by `SnDBackend.to_physical`.
    """

    def __init__(self, user, input_list, motion_range=50e-6):
        names = [motor.name for motor in input_list]
        pos_range = {name: motion_range * 180 / np.pi for name in names}
        start_pos = {name: 0.0 for name in names}
        super().__init__(names, pos_range, start_pos)
        self.user = user
        self.input_list = input_list
        self.motor_dict = dict(zip(names, input_list))

    def refresh_start_pos(self):
        for motor in self.input_list:
            self.start_pos[motor.name] = motor.wm()

    def measure(self, positions):
        user = self.user
        intensity_list = [user.do_signal]
        centroid_list = [user.cx_signal, user.cy_signal, user.wx_signal, user.wy_signal]
        signal_list = intensity_list + centroid_list

        # move motors to the requested physical setpoints
        status_list = []
        for key, value in positions.items():
            time.sleep(.1)
            status = self.motor_dict[key].move(value)
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
            if temp <= 0:
                temp = 1e-3
            signal_sum += temp

        norm = user.ipm4_signal.get()
        if norm <= 1000:
            return self.measure(positions)

        # the following is pre-normalized
        ### it looks like this was expecting to be divided by the normalization signal, but never happened,
        ### so seems like an error. It may be because we were using pink beam last time. Need to revisit.
        signal_sum += user.intensity_signal.get() * user.ipm4_signal.get() / 50

        if user.intensity_signal.get() * user.ipm4_signal.get() / 50 < 1000:
            cx = np.nan
            cy = np.nan
            wx = np.nan
            wy = np.nan
        else:
            cx = np.nanmean(user.cx_signal.values)
            cy = np.nanmean(user.cy_signal.values)
            wx = np.nanmean(user.wx_signal.values)
            wy = np.nanmean(user.wy_signal.values)

        return {
            'intensity': signal_sum,
            'cx': cx,
            'cy': cy,
            'wx': wx,
            'wy': wy,
        }

    def set_target(self):
        # function to call when setting the centroid target based on cc branch position
        user = self.user
        signal_list = [user.cx_signal, user.cy_signal]
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

        return user.cx_signal.get(), user.cy_signal.get()

    def move_to_start(self):
        for key in self.start_pos.keys():
            self.motor_dict[key].umv(self.start_pos[key])


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
        self.backend = HardwareBackend(
            self, [self.fast_motor1, self.fast_motor2], motion_range=50e-6
        )
        self.optimizer = SnDOptimizer(self.backend, savepath=savepath)
        # sim-axis names matching the real-motor order in set_motors; only set
        # when real motors are selected, and used to build the matching prior.
        self._sim_axis_names = None

    def set_motors(self, motion_range=50e-6, sim=False):
        """
        Set inputs to actual SnD motors and rebuild the optimizer backend.

        Parameters
        ----------
        motion_range: float
            Xopt motion range for angular motions (radians)
        sim: bool
            If True, use simulated ophyd motors, otherwise use real motors
        """
        if sim:
            input_list = [self.fast_motor1, self.fast_motor2]
            self._sim_axis_names = None
        else:
            # The real SnD motors, paired position-for-position with the
            # differentiable-sim axis names below. The prior is positional: the
            # i-th optimizer variable must be the i-th sim axis, so these two
            # lists MUST stay in the same order. Note the t4 th/chi ordering
            # differs from snd_prior.DEFAULT_AXIS_NAMES, which is why we hand the
            # prior an explicit axis_names list instead of relying on that default.
            input_list = [self.snd.t1.th1, self.snd.t1.chi1, self.snd.t1.th2,
                self.snd.t1.chi2, self.snd.t4.th2, self.snd.t4.chi2,
                self.snd.t4.th1, self.snd.t4.chi1]
            self._sim_axis_names = ["t1_th1", "t1_chi1", "t1_th2", "t1_chi2",
                "t4_th2", "t4_chi2", "t4_th1", "t4_chi1"]

        self.backend = HardwareBackend(self, input_list, motion_range=motion_range)
        self.optimizer = SnDOptimizer(self.backend, savepath=savepath)

    # -- optimization delegation ------------------------------------------
    # These thin wrappers keep the interactive / GUI API stable while the
    # optimization logic lives in snd_optimizer.SnDOptimizer.
    def set_target(self):
        self.optimizer.set_target()

    def initialize_turbo(self, pos_range=None, n_init=64, scale=1):
        if pos_range is not None:
            self.backend.set_pos_range(pos_range)
        self.optimizer.initialize_turbo(n_init=n_init, scale=scale)

    def initialize_BO(self, pos_range=None, n_init=64, scale=1e-4):
        if pos_range is not None:
            self.backend.set_pos_range(pos_range)
        self.optimizer.initialize_BO(n_init=n_init, scale=scale)

    def initialize_BO_transformed(self, pos_range=None, n_init=64, scale=1e-4):
        if pos_range is not None:
            self.backend.set_pos_range(pos_range)
        self.optimizer.initialize_BO_transformed(n_init=n_init, scale=scale)

    def run_BO(self, num_iter=150, seed=42):
        return self.optimizer.run_BO(num_iter=num_iter, seed=seed)

    def run_turbo(self, num_iter=150):
        return self.optimizer.run_turbo(num_iter=num_iter)

    def get_optimum_details(self, plot=True, move_to_optimum=True):
        self.optimizer.get_optimum_details(plot=plot, move_to_optimum=move_to_optimum)

    def move_to_start(self):
        self.optimizer.move_to_start()

    def enable_prior(self, intensity_scale=1e-4, energy=9500.0,
                     delay=280e-3, detector="do"):
        """Attach the differentiable-sim physics prior to the GP for objective "f".

        Must be called AFTER ``set_target`` (the prior captures the centroid
        target) and BEFORE ``initialize_turbo`` / ``initialize_BO`` (the prior is
        wired in when the generator's model is built). Real SnD motors only.

        ``intensity_scale`` must match the ``scale`` passed to ``initialize_*`` so
        the prior and the measured objective live on the same scale.
        """
        if self._sim_axis_names is None:
            raise RuntimeError(
                "enable_prior requires the real SnD motors; "
                "call set_motors(sim=False) first"
            )
        # Imported lazily: snd_prior pulls in torch + the differentiable sim,
        # which need not be installed for plain hardware runs without a prior.
        from snd_prior import build_snd_sim

        # Match the prior's normalized->physical range to the backend's. start_pos
        # is left to default to the sim's own aligned baseline (NOT the hardware
        # encoder positions): both map normalized 0.5 -> aligned, so the relative
        # search space is shared even though the absolute references differ.
        range_val = next(iter(self.backend.pos_range.values()))
        backend_pos_range = {name: range_val for name in self._sim_axis_names}

        _, prior = build_snd_sim(
            axis_names=self._sim_axis_names,
            backend_pos_range=backend_pos_range,
            backend_start_pos=None,
            energy=energy,
            delay=delay,
            detector=detector,
            x_target=self.optimizer.x_target,
            y_target=self.optimizer.y_target,
            intensity_scale=intensity_scale,
        )
        self.optimizer.set_prior_mean(prior)

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
