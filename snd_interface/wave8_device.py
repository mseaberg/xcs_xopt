from ophyd.device import Device
from ophyd.status import StatusBase, SubscriptionStatus
from threading import RLock
import numpy as np
from ophyd.status import DeviceStatus
from ophyd.signal import EpicsSignalRO, Signal
from ophyd.utils.epics_pvs import data_type, data_shape
import time
from ophyd.device import Component as Cpt
from pcdsdevices.signal import AvgSignal

# consider using pcdsdevices AvgSignal as a template, and over-writing the trigger method to reset the buffer, and also wait until it is filled, then "put" the value

class AvgSignalTriggered(AvgSignal):
    def __init__(self, signal, averages, duration, *, name, parent=None, **kwargs):
        super().__init__(signal, averages, name=name, parent=parent, **kwargs)
        if isinstance(signal, str):
            signal = getattr(parent, signal)
        self.raw_sig = signal
        self._lock = RLock()
        self.averages = averages
        self.duration = duration
        self.raw_sig.subscribe(self._update_avg)

    def trigger(self):
        self.averages = self._avg
        status = StatusBase(settle_time=self.duration)
        status.set_finished()
        return status



class AvgSignalTriggeredOld(Signal):
    """
    Signal that acts as a rolling average of another signal.

    This will subscribe to a signal, and fill an internal buffer with values
    from `SUB_VALUE`. It will update its own value to be the mean of the last n
    accumulated values, up to the buffer size. If we haven't filled this
    buffer, this will still report a mean value composed of all the values
    we've receieved so far.

    Warning: this means that if we only have recieved ONE value, the mean will
    just be the mean of a single value!

    Parameters
    ----------
    signal : Signal
        Any subclass of `ophyd.signal.Signal` that returns a numeric value.
        This signal will be subscribed to be `AvgSignal` to calculate the mean.

    averages : int
        The number of `SUB_VALUE` updates to include in the average. New values
        after this number is reached will begin overriding old values.
    """

    def __init__(self, signal, averages, duration, *, name, parent=None, **kwargs):
        super().__init__(name=name, parent=parent, **kwargs)
        if isinstance(signal, str):
            signal = getattr(parent, signal)
        self.raw_sig = signal
        self._lock = RLock()
        self.averages = averages
        self.duration = duration
        self.raw_sig.subscribe(self._update_avg)

    @property
    def connected(self):
        return self.raw_sig.connected

    @property
    def averages(self):
        """The size of the internal buffer of values to average over."""
        return self._avg

    @averages.setter
    def averages(self, avg):
        """Reinitialize an empty internal buffer of size `avg`."""
        with self._lock:
            self._avg = avg
            self.index = 0
            # Allocate uninitalized array
            self.values = np.empty(avg)
            # Fill with nan
            self.values.fill(np.nan)

    def _update_avg(self, *args, value, **kwargs):
        """Add new value to the buffer, overriding old values if needed."""
        with self._lock:
            self.values[self.index] = value
            self.index = (self.index + 1) % len(self.values)
            # This takes a mean, skipping nan values.
            self.put(np.nanmean(self.values))

#    def trigger(self):
#        def check_index(*, old_value, value, **kwargs):
#            return self.index>0
#        self.averages = self._avg
#        status = SubscriptionStatus(self.raw_sig, check_index,settle_time=self.duration,timeout=5)
#        return status

    def trigger(self):
        self.averages = self._avg
        status = StatusBase(settle_time=self.duration)
        status.set_finished()
        return status




class DiodeDevice(Device):
    def __init__(self, channel, num_samples, **kwargs):
        super().__init__(**kwargs)
        self.diode = Cpt(channel, num_samples, name='diode', kind='hinted')

class Diode(Device):
    
    def __init__(self, channel, num_samples, **kwargs):
        
        super().__init__(**kwargs)
        self.channel = EpicsSignalRO(channel)
        self.num_samples = num_samples
        self.data = {}
        self.value = 0.0

    def stage(self):
        
        t_0 = time.time()
        self.data = {self.name: {'value': 0.0, 'timestamp': t_0}}

        return [self]

    def trigger(self):
        t_0 = time.time()
        sum = 0
        for i in range(self.num_samples):
            sum += self.channel.get()
            time.sleep(.0083)

        t_end = time.time()
        self.value = sum/self.num_samples

        self.data[self.name] = {'value': self.value, 'timestamp': t_end}

        return DeviceStatus(self, done=True, success=True)

    def read(self):
        return self.data

    def get(self):
        return self.data[self.name]['value']

    def describe(self):
        data = {self.name: {'source':'PV', 'dtype':data_type(self.value),'shape':data_shape(self.value), 'units':'s','precision':3}}
        return data

    def unstage(self):
        return [self]
