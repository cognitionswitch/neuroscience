import numpy as np
import pandas as pd
import pytest
import os
import sys


sys.path.append(os.path.join(os.getcwd(), 'neuro/sta'))
from spike_triggered_avg import STA

class TestSTA():
    
    valid_data = {'event_list': np.array([1, 0, 0, 1, 0]), 
                       'stim_list': [-0.4, -6, 0, 3.0, 2/3], 
                       'samp_pd': 0.01, 
                       'kernel_pd': 0.2}
    # type errors
    def type_vals(valid_types):
        
        invalid_primary = [('event_list', {1, 0, 0, 1, 0}), # not sequence
                           ('stim_list', 0.4), # not sequence
                           ('samp_pd', 'b'), # not numeric 
                           ('kernel_pd', 'a')] # not numeric
        
        kw, type_err, ids = [], [], []
        for (k, v) in invalid_primary:
            kw.append(k)
            type_err.append(tuple(v0 if k0 != k else v for k0, v0 in valid_types.items()))
            ids.append('function arg type error: ' + k)
        
        # container elements        
        invalid_elements = {'event_list': [1, 0, 0, -1.0, 0], # contains float 
                            'stim_list': [-0.4, -6, 0, 3.0, 'a']} # contains str
        
        for k, v in invalid_elements.items():
            type_err.append(tuple(v0 if k0 != k else v for k0, v0 in valid_types.items()))
            ids.append('container element type error: ' + k)
        
        return {'argnames':tuple(kw), 'argvalues':type_err, 'ids':ids}

    @pytest.mark.parametrize(**type_vals(valid_data))
    def test_types(self, event_list, stim_list, samp_pd, kernel_pd):
        with pytest.raises(TypeError):
            STA(event_list, stim_list, samp_pd, kernel_pd)
            
            
    # value errors
    def value_vals(valid_vals):
        
        invalid_vals = {'event_list': [1, 0, 0, -1, 0], # negative elements
                        'stim_list': [-0.4, -6, 0, 3.0, 2/3, 10], # unequal length 
                        'samp_pd': 0, # non-positive
                        'kernel_pd': -4} # non-positive
        
        kw, val_err, ids = [], [], []
        for k, v in invalid_vals.items():
            kw.append(k)
            val_err.append(tuple(v0 if k0 != k else v for k0, v0 in valid_vals.items()))
            ids.append('value errors: ' + k)
        return {'argnames':tuple(kw), 'argvalues':val_err, 'ids':ids}

    @pytest.mark.parametrize(**value_vals(valid_data))
    def test_values(self, event_list, stim_list, samp_pd, kernel_pd):
        with pytest.raises(ValueError):
            STA(event_list, stim_list, samp_pd, kernel_pd)