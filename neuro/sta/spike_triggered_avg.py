import numpy as np
import pandas as pd

class STA:
    """Estimates spike-triggered average stimulus from event data.
    
    The STA class requires as inputs 1) a discrete-valued event sequence and 
    2) a sequence of stimulus values presumed to influence the generation of 
    those events. An STA instance computes simple averages of those stimulus 
    values time-locked to event occurances over a user specified window duration. 
    STA instances can also generate averages of stimulus values over window 
    durations time-locked to pairwise events with or without adjacency 
    requirements between events. Finally, STA instances can generate averages 
    of stimulus values over window durations time-locked any arbitrary event 
    pattern.  
    
    Arguments
    ------------
    event_list: list or 1D numpy array 
        - list of binary or boolean elements signifying sequential position of event of interest.
    stim_list: list or 1D numpy array
        - list of stimulus values with a presumed causal relationship with the events contained in `event_list`.
    samp_pd: float {t | t > 0}
        - sampling period of events in `event_list` and stimulus values in `stim_list`; use same units as `kernel_pd`.
    kernel_pd: float {t | t > 0}
        - kernel period representing the duration of the look-back window from the events. 
        contained in `event_list`; use same units as `samp_pd`.
    
    Attributes
    ------------
    rho: numpy.array
        - list of binary or boolean elements signifying sequential position of event of interest.
    stim: numpy.array
        - list of stimulus values with a presumed causal relationship with the events contained in `event_list`.
    samp_pd: float 
        - sampling period of events in `event_list` and stimulus values in `stim_list`; 
        use same units as `kernel_pd.`
    samp_rt: float  
        - equal to 1/`samp_pd`. Sampling rate of the events in `event_list` and stimulus values in `stim_list`. 
    kernel_pd: float
        - kernel period representing the duration of the look-back window from the events contained 
        in `event_list`; use same units as `samp_pd`.
    kernel_wd: int 
        - width of the kernel window; equal to int(`kernel_pd`/`samp_pd`).
    self.sig_fig: int
        - significant figures used to generate a time index for the events in `event_list` and stimulus 
        values in `stim_list`.
    sta_sr: pandas.series
        - simple spike triggered average; value is set to None until instance methods are executed.
    
    Methods
    ------------
    simple_sta: 
        - simple spike-triggered average based on single event occurance in `event_list`.
    pairwise_sta:
        - spike-triggered average defined by pairwise event occurances in `event_list`.
    multi_sta:
        - spike-triggered average defined by user defined event pattern (with arbitrary number 
        of events and arbitrary intervals between these events) in `event_list`.
    """
    
    def __init__(self, event_list:list, stim_list:list, samp_pd:float=0.01, kernel_pd:float=0.10):
        
        self.rho = np.array(event_list) if isinstance(event_list, list) else event_list
        self.stim = np.array(stim_list) if isinstance(stim_list, list) else stim_list
        self.samp_pd = samp_pd
        self.samp_rt = 1/self.samp_pd
        self.kernel_pd = kernel_pd
        self.kernel_wd = int(self.kernel_pd/self.samp_pd)
        self.sig_fig = 0 if len(str(self.samp_pd).split('.')) < 2 else len(str(self.samp_pd).split('.')[-1])
        self.sta_sr = None
        
    def simple_sta(self, event_list:list=None):
        """Estimates a spike-triggered average stimulus from a record of event data.
        
        Arguments
        ------------
        event_ls: list (default: None)
            - if the STA().simple_sta() method call includes event_ls, a simple spike-triggered average 
            will be computed using this provided event_ls and the stim_ls provided with the STA() 
            constructor; otherwise the simple spike-triggered average will be based on the 
            event_ls provided with the STA() constructor. 
        
        Returns
        ------------
        sta_sr: numpy.array
            - a simple average of stimulus values time-locked to the occurance of events in event_ls
        """
        
        if event_list is not None:
            assert 1 in event_list, 'event list must contain at least one event occurance'
        
        # create df of kernel-window lookback periods
        spike_idx = self.rho.nonzero()[0] if event_list is None else event_list.nonzero()[0]
        start_idx = np.maximum(0, spike_idx-self.kernel_wd)
        sta_sr = self._sta(start_idx, spike_idx)
        
        if event_list is None:
            self.sta_sr = sta_sr
        
        return sta_sr
    
    def _sta(self, start_index, spike_index):
        spike_list = []
        mask_list = []
        for start_id, spike_id in zip(start_index, spike_index):
            pre_obs_prd = self.kernel_wd - spike_id # for lookback window preceding observation period 
            pre_obsv_win = [] if pre_obs_prd <= 0 else [0]*pre_obs_prd # backfill lookback w 0s 
            stim_lookback =  pre_obsv_win + list(self.stim[start_id:spike_id])
            spike_list.append(stim_lookback)

        stimulus_array = np.array(spike_list, dtype=float, ndmin=2).T
        
        return np.dot(stimulus_array, self.rho[spike_index])/sum(self.rho[spike_index])
    
    def pairwise_sta(self, min_period:int=1, max_period:int=None, adjacent:bool=True):
        """Estimate pairwise spike-triggered average stimuli.
        
        STA.pairwise_sta() generates list of spike triggered average (STA) stimuli based 
        on a user defined range of periods between event pairs. Each period represents an
        interval between two events. The occurance of this interval between events 
        defines trigger used to time-lock the look-back windows for an STA. 
        
        Arguments
        ------------
        min_period: int
            - the minimum number of periods to use for the inter-event interval
        max_period: int
            - the maximum number of periods to use for the inter-event interval
        adjacent: bool (default=True)
            - if True, the criteria for an STA trigger requires that the inter-event 
            interval between two events must not contain occurances of intermediate events. 
        
        Returns
        ------------
        ord1_sta_df: pandas.DataFrame
            - a DataFrame with columns corresponding to an inter-event duration and rows 
            corrsponding to the 1st-order estimated STA of stimulus values time-locked to 
            the occurance of inter-event durations indexed by the column names.  
        ord2_sta_df: pandas.DataFrame
            - a DataFrame with columns corresponding to an inter-event duration, and rows 
            corrsponding to the 2nd-order STA of stimulus values time-locked to the 
            inter-event durations of corresponding column names. Specifically, 2nd-order 
            STAs represent stimulus averages time-locked to inter-event durations after 
            subtracting out the sum of two 1st-order STAs seperated by inter-event intervals 
            within min_period and max_period, inclusive.
        """
        
        max_period = min_period if max_period is None else max_period
        
        if self.sta_sr is None:
            self.simple_sta()
        
        # range of intervals between event pairs
        isi_rg = range(min_period, max_period+1)
        
        if adjacent:
            # STA for spike pairs: events must be adjacent (no intermediate events)
            trig_arr_ls = list(
                map(lambda isi: 
                    (pd.DataFrame(self.rho[::-1])
                     .rolling(window = isi + 1, min_periods = isi + 1)
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
            sta0_sr_ls.append(self._sta(start_idx, trig_idx))
            
        ord1_sta_df = pd.DataFrame(sta0_sr_ls, columns=sr_idx).T
        ord1_sta_df.columns = [round(self.samp_pd*d, self.sig_fig) for d in isi_rg]
        
        # compute the component of the signal due to simple STA for each spike in the pair and sum them
        sta_sum_df = pd.DataFrame([self.sta_sr + self._shift(self.sta_sr, -t, fill_value=0)  for t in isi_rg], 
                                  index=ord1_sta_df.columns, 
                                  columns=ord1_sta_df.index).T
        
        # subtract away the simple STA components to get the 2nd order stimulus STA 
        ord2_sta_df = ord1_sta_df.subtract(sta_sum_df)
        
        return ord1_sta_df, ord2_sta_df
    
    def _shift(self, xs, n, fill_value):
        """Shift input list up or down with backfill.
        
        Helper function to quickly shift an input list up or down.
        """
        # https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
        e = np.empty_like(xs)
        if n >= 0:
            e[:n], e[n:] = fill_value, xs[:-n]
        else:
            e[n:], e[:n] = fill_value, xs[-n:]
        return e
    
    def multi_sta(self, interval_list:list, return_high_order:bool=False):
        """Estimate spike-triggered average triggered by arbitrary input sequence of events.
        
        STA().multi_sta() generates columns of spike triggered average (STA) from an input 
        sequence `interval_list` representing periods between event occurances. The 
        intervals between events defines the triggers to time-lock the look-back windows for 
        the STA. 
        
        Outputs are columns of STAs corresponding to the cumulative intervals 
        `interval_list`. For example, the first column always returns a simple STA triggered
        by simple events; the second column returns the STA triggered when response events 
        are separated by the first entry of `interval_list`; the third column returns an STA 
        triggered when three response events are separated by the first and second entries 
        of `interval_list`; etc. 
        
        STA.multi_sta() always returns first order STA. Higher order STA are also returned 
        when `return_high_order` = True. High order STA are the difference between  
        multi-event STA and the sum of lower order STAs triggered by the decomposed event 
        sequence. 
        
        For example, a trigger composed of two events separated by a lag generates a 
        multi-event STA. This trigger is decomposable into two single events separated by a 
        lag. The difference between the multi-event STA and the simple sum of two simple 
        STAs generated by a single trigger (separated by a lag) represents a 2nd order STA. 
        
        For `return_high_order` = True, an input sequence of N intervals returns a dataframe
        with N columns. The first column is always the simple STA; the nth column represents
        the nth order STA. 
        
        STA.multi_sta() does not currently support an optional constraint that events must be 
        adjacent (ie., intermediate events between intervals may occur).
        
        Arguments
        ------------
        interval_list: list
            - list of integers signifying lags between event occurances.
            
        return_high_order: bool, default is False
            - returna dataframe of high order STAs with contributions to response sequences 
            given in `interal_list` that remain after removing lower order STAs. 
        
        Returns
        ------------
        n_event_df: pandas.DataFrame
            - DataFrame with columns corresponding to successive intervals between events and 
            rows corrsponding to the 1st-order STA time-locked to the occurance of multi-events 
            separated by intervals indicated by corresponding columns. The 1st-order STA 
            represent simple STAs time-locked to events defined by inter-event durations of 
            `interval_list`. 
            
        nth_order_sta: pandas.DataFrame
            - Dataframe with the nth column corresponding to the n-1 order STA. 
        """
        
        if self.sta_sr is None:
            self.simple_sta()
        
        trig_arr_ls = [self.sta_sr]
        intvl_du_ls = [0]
        for i in range(len(interval_list)):
            # generate paired event sequences based on event pairs with fixed interval durations
            trig_arr = self._event_pair_recurse(interval_list[0:i+1])
            trig_arr_ls.append(self.simple_sta(trig_arr))
            intvl_du_ls.append(intvl_du_ls[i]+interval_list[i])                        
        
        n_event_df = pd.DataFrame(trig_arr_ls, index=[self.samp_pd*d for d in intvl_du_ls]).T
        n_event_df.index = [-round(x*self.samp_pd, 3) for x in range(self.kernel_wd, 0, -1)]
        
        if return_high_order:
        # generate nth order contribution to STA
            cum_int = [0] + [sum(interval_list[:i+1]) for i in range(len(interval_list))]
            cumsum_sta_shift = {}
            nth_order_sta = pd.DataFrame()
            for i, s in enumerate(cum_int):
                cumsum_sta_shift[i] = n_event_df.shift(-s, fill_value=0).cumsum(axis=1)
                precapture_sta =  0 if i < 1 else cumsum_sta_shift[0].iloc[:, i - 1]
                for r in range(0, i):
                    precapture_sta += cumsum_sta_shift[i-r].iloc[:, r]

                nth_order_sta.loc[:, str(cum_int[i] * self.samp_pd)] \
                = n_event_df.iloc[:, i] - precapture_sta
                
            return n_event_df, nth_order_sta
        
        else:
            return n_event_df
    
    def _event_pair_recurse(self, interval_list:list):
        """Constructs event list (triggers) with arbitrary inter-event durations from a list of input events.

        Recursive function that constructs a higher-order event list from list of input events with arbitrary 
        inter-event durations between occurances of input event list.

        Arguments
        ------------
        interval_list: list (t | t > 0)
            - list of integers signifying intervals between occurances in list of input events.

        Returns
        ------------
        trig_arr: numpy.array (1D)
            - list of multi-event triggers constructed from original list of simple input events.
        """
        
        if len(interval_list) == 0:
            trig_arr = self.rho
        else:
            t = sum(interval_list) 
            trig_arr = np.append(self.rho[t:], np.zeros(t))*self._event_pair_recurse(interval_list[:-1])

        return trig_arr