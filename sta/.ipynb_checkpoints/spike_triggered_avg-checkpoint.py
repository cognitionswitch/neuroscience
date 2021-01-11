import numpy as np
import pandas as pd

class STA:
    """Estimates a spike-triggered average stimulus from event data.
    
    The STA class inputs 1) an event list and 2) a list of stimulus values 
    presumed to influence the generation of those events. The STA will compute 
    simple averages of those stimulus values time-locked to event occurances 
    over a user specified window duration. The STA class can also generate 
    averages of stimulus values over window durations for the occurance of pairwise 
    events with or without adjacency requirements between events. Finally, the 
    STA class can generate averages of stimulus values over window durations 
    time-locked to the occurance of an arbitrary event pattern.   
    
    Arguments
    ------------
    event_ls: list or 1D numpy array 
        - list of binary or boolean elements signifying sequential position of event of interest
    stim_ls: list or 1D numpy array
        - list of stimulus values with a presumed causal relationship with the events contained in event_ls
    samp_pd: float {t | t > 0}
        - sampling period of events in event_ls and stimulus values in stim_ls; use same units as kernel_pd
    kernel_pd: float {t | t > 0}
        - kernel period representing the duration of the look-back window from the events 
        contained in event_ls; use same units as samp_pd 
    
    Attributes
    ------------
    rho: numpy.array
        - list of binary or boolean elements signifying sequential position of event of interest
    stim: numpy.array
        - list of stimulus values with a presumed causal relationship with the events contained in event_ls
    samp_pd: float 
        - sampling period of events in event_ls and stimulus values in stim_ls; use same units as kernel_pd
    samp_rt: float 1/samp_pd 
        - sampling rate of the events in event_ls and stimulus values in stim_ls 
    kernel_pd: float
        - kernel period representing the duration of the look-back window from the events contained 
        in event_ls; use same units as samp_pd
    kernel_wd: int int(kernel_pd/samp_pd)
        - width of the kernel window
    self.sig_fig: int
        - significant figures used to generate a time index for the events in event_ls and stimulus 
        values in stim_ls
    sta_sr: pandas.series
        - simple spike triggered average; value is set to None until instance methods are executed
    
    Methods
    ------------
    simple_sta: 
        - simple spike-triggered average based on single event occurance in event_ls
    pairwise_sta:
        - spike-triggered average defined by pairwise event occurances in event_ls
    multi_sta:
        - spike-triggered average defined by user defined event pattern (with arbitrary number 
        of events and arbitrary intervals between these events) in event_ls 
    """
    
    def __init__(self, event_ls:list, stim_ls:list, samp_pd:float=0.01, kernel_pd:float=0.10):
        
        self.rho = np.array(event_ls) if type(event_ls) == list else event_ls
        self.stim = np.array(stim_ls) if type(stim_ls) == list else stim_ls
        self.samp_pd = samp_pd
        self.samp_rt = 1/self.samp_pd
        self.kernel_pd = kernel_pd
        self.kernel_wd = int(self.kernel_pd/self.samp_pd)
        self.sig_fig = 0 if len(str(self.samp_pd).split('.')) < 2 else len(str(self.samp_pd).split('.')[-1])
        self.sta_sr = None
        
    def simple_sta(self, event_ls:list=None):
        """Estimates a simple spike-triggered average stimulus from a record of events data.
        
        Arguments
        ------------
        event_ls: list (default: None)
            - if the STA().simple_sta() method call includes event_ls, a simple spike-triggered average 
            will be computed using this provided event_ls and the stim_ls provided with the STA() 
            constructor; otherwise the simple spike-triggered average will be based on the 
            event_ls provided with the STA() constructor. 
        
        Returns
        ------------
        sta_sr: pandas.Series
            - a simple average of stimulus values time-locked to the occurance of events in event_ls
        """
        if event_ls is not None:
            assert sum(event_ls) >= 1, 'event list must contain at least one event occurance'
        
        # create df of kernel-window lookback periods
        spike_idx = self.rho.nonzero()[0] if event_ls is None else event_ls.nonzero()[0]
        start_idx = np.maximum(0, spike_idx-self.kernel_wd)
        spike_ls = [self.stim[start:spike] for start, spike in zip(start_idx, spike_idx)]
        
        stim_df = pd.DataFrame(spike_ls)
        stim_df.columns = [-round(x*self.samp_pd, self.sig_fig) for x in range(self.kernel_wd, 0, -1)]
        sta_sr = stim_df.mean(axis='rows')
        
        if event_ls is None:
            self.sta_sr = sta_sr
        
        return sta_sr
    
    def pairwise_sta(self, min_period:int=1, max_period:int=1, adjacent:bool=True):
        """Estimates lists of pairwise spike-triggered average stimulus from a record of single events.
        
        STA.pairwise_sta() generates lists of spike triggered average stimulus based on a user defined range of 
        periods between event pairs. Each period represents an interval between two events. The occurance of
        this interval defines the event used to time-lock the look-back windows for a spike-triggered average. 
        
        Arguments
        ------------
        min_period: int
            - the minimum number of periods to use for the inter-event interval
        max_period: int
            - the maximum number of periods to use for the inter-event interval
        adjacent: bool (default=True)
            - if True, the interval between two events must not contain occurances of intermediate events 
        
        Returns
        ------------
        ord1_sta_df: pandas.DataFrame
            - a DataFrame with columns corresponding to an inter-event duration and rows corrsponding to the 1st-order
            estimated spike-triggered average of stimulus values time-locked to the occurance of inter-event durations 
            indexed by the column names. The 1st-order spike-triggered averages represent simple spike-triggered 
            averages time-locked to events defined by inter-event durations. 
        ord2_sta_df: pandas.DataFrame
            - a DataFrame with columns corresponding to an inter-event duration and rows corrsponding to the 2nd-order
            estimated spike-triggered average of stimulus values time-locked to the occurance of inter-event durations 
            indexed by the column names. The 2nd-order spike-triggered averages represent spike-triggered 
            averages time-locked to events defined by inter-event durations after subtracting out the 1st-order spike-
            triggered averages.
        """
        
        if self.sta_sr is None:
            self.simple_sta()
        
        # range of intervals between event pairs
        isi_rg = range(min_period, max_period+1)
        
        if adjacent:
            # STA for spike pairs: events must be adjacent (no intermediate events)
            trig_arr_ls = list(
                map(lambda isi: 
                    (pd.DataFrame(self.rho[::-1])
                     .rolling(window=isi+1, min_periods=isi+1)
                     .apply(lambda x: (x==([1]+[0]*(isi-1)+[1])).all(), raw=True)
                     .fillna(0))
                    .values[::-1],
                    isi_rg
                   )
            )
        else:            
            # generate paired event sequences based on event pairs with fixed interval durations
            trig_arr_ls = [np.append(self.rho[t:], np.zeros(t))*self.rho for t in isi_rg]
            
        # generate indices of event pairings for paried event sequences
        trig_idx_ls = [trig_arr.nonzero()[0] for trig_arr in trig_arr_ls]
        
        start_idx_ls = [np.maximum(0, trig_idx-self.kernel_wd) for trig_idx in trig_idx_ls]
        
        # generate lists of sta for each paried event duration 
        sr_idx = [-round(x*self.samp_pd, self.sig_fig) for x in range(self.kernel_wd, 0, -1)]
        sta0_sr_ls = []
        for start_idx, trig_idx in zip(start_idx_ls, trig_idx_ls):
            stim0_ls = [self.stim[start:trig] for start, trig in zip(start_idx, trig_idx)]
            stim0_df = pd.DataFrame(stim0_ls, columns=sr_idx).T
            sta0_sr_ls.append(stim0_df.mean(axis='columns'))
            
        ord1_sta_df = pd.concat(sta0_sr_ls, axis=1)
        ord1_sta_df.columns = [round(self.samp_pd*d, self.sig_fig) for d in isi_rg]
        # compute the component of the signal due to simple STA for each spike in the pair and sum them
        sta_sum_df = pd.DataFrame([self.sta_sr + self.sta_sr.shift(-t, fill_value=0)  for t in isi_rg], 
                                  index=ord1_sta_df.columns, columns=ord1_sta_df.index).T
        # subtract away the simple STA components to get the 2nd order stimulus STA 
        ord2_sta_df = ord1_sta_df.subtract(sta_sum_df)
        
        return ord1_sta_df, ord2_sta_df
    
    def multi_sta(self, interval_ls:list):
        """Estimates columns of n-wise spike-triggered average stimulus from a record of single events provided to constructor.
        
        STA().multi_sta() generates columns of spike triggered average stimuli based on a user defined sequence of 
        periods between event occurances. The periods in the sequence represent intervals between corresponding 
        subsequent events. The occurance of these intervals between events defines an event used to time-lock 
        the look-back windows for a spike-triggered average. 
        
        STA().multi_sta() computes a sequence of spike-triggered averages with orders corresponding to the 
        cumulative sequences in the period sequence. For example, the first column will always return
        a simple spike-triggered average from simple events; the second column will return a spike-triggered
        average on events defined by the first interval of interval_ls; the third column will return a
        spike-triggered average on events defined by the first interval followed by the second interval of 
        interval_ls; etc. 
        
        This method does not currently support an optional constraint that events must be adjacent 
        (ie., intermediate events between intervals may occur).
        
        Arguments
        ------------
        interval_ls: list
            - list of integers signifying intervals between occurances in list of input events.
        
        Returns
        ------------
        trig_df: pandas.DataFrame
            - a DataFrame with columns corresponding to an multi-event durations and rows corrsponding to the 1st-order
            estimated spike-triggered average of stimulus values time-locked to the occurance of multi-event durations 
            accruing with the values of interval_ls. The 1st-order spike-triggered averages represent 
            simple spike-triggered averages time-locked to events defined by inter-event durations. 
        """
        
        if self.sta_sr is None:
            self.simple_sta()
        
        trig_arr_ls = [self.sta_sr]
        intvl_du_ls = [0]
        for i in range(len(interval_ls)):
            # generate paired event sequences based on event pairs with fixed interval durations
            trig_arr = self._event_pair_recurse(interval_ls[0:i+1])
            trig_arr_ls.append(self.simple_sta(trig_arr))
            intvl_du_ls.append(intvl_du_ls[i]+interval_ls[i])                        
        
        trig_df = pd.DataFrame(trig_arr_ls, index=[self.samp_pd*d for d in intvl_du_ls]).T
        
        return trig_df
    
    def _event_pair_recurse(self, intvl_ls:list):
        """Constructs event list (triggers) with arbitrary inter-event durations from a list of input events.

        Recursive function that constructs a higher-order event list from list of input events with arbitrary 
        inter-event durations between occurances of input event list.

        Args
        ------------
        event_ls: list
            - input event list
        intvl_ls: list (t | t > 0)
            - list of integers signifying intervals between occurances in list of input events.

        Returns
        ------------
        trig-arr: numpy.array (1D)
            - list of events constructed from original list of input events
        """
        if len(intvl_ls) == 0:
            trig_arr = self.rho
        else:
            t = sum(intvl_ls) 
            trig_arr = np.append(self.rho[t:], np.zeros(t))*self._event_pair_recurse(intvl_ls[:-1])

        return trig_arr