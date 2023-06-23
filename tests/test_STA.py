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
        
        kw = ('event_list', 'stim_list', 'samp_pd', 'kernel_pd')
        
        invalid_primary = [('event_list', {1, 0, 0, 1, 0}, 'input not a sequence'), # not sequence
                           ('stim_list', 0.4, 'input not a sequence'), # not sequence
                           ('samp_pd', 'b', 'input not numeric'), # not numeric 
                           ('kernel_pd', 'a', 'input not numeric'),
                           ('event_list', [1, 0, 0, -1.0, 0], 'container elements include float'), 
                           ('stim_list', [-0.4, -6, 0, 3.0, 'a'], 'container elements include str')]
        
        type_err, ids = [], []
        for (k, v, i) in invalid_primary:
            type_err.append(tuple(v0 if k0 != k else v for k0, v0 in valid_types.items()))
            ids.append(k + ' type error: ' + i)
        
        return {'argnames':kw, 'argvalues':type_err, 'ids':ids}

    @pytest.mark.parametrize(**type_vals(valid_data))
    def test_types(self, event_list, stim_list, samp_pd, kernel_pd):
        with pytest.raises(TypeError):
            STA(event_list, stim_list, samp_pd, kernel_pd)
            
            
    # value errors
    def value_vals(valid_vals):
        
        kw = ('event_list', 'stim_list', 'samp_pd', 'kernel_pd')
        
        invalid_vals = [('event_list', [1, 0, 0, -1, 0], 'negative elements'),
                        ('stim_list', [-0.4, -6, 0, 3.0, 2/3, 10], 'unequal length'),  
                        ('samp_pd', 0, 'non-positive'),
                        ('kernel_pd', -4, 'non-positive')]
        
        val_err, ids = [], []
        for (k, v, i) in invalid_vals:
            val_err.append(tuple(v0 if k0 != k else v for k0, v0 in valid_vals.items()))
            ids.append(k + ' value error: ' + i)
            
        return {'argnames':kw, 'argvalues':val_err, 'ids':ids}

    @pytest.mark.parametrize(**value_vals(valid_data))
    def test_values(self, event_list, stim_list, samp_pd, kernel_pd):
        with pytest.raises(ValueError):
            STA(event_list, stim_list, samp_pd, kernel_pd)