#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.integrate import odeint

class Neuron:
    
    def __init__(self, pos:tuple=None, radius:float=0.25, valence:bool=True, potential:float=0.0,  
                 theta:float=0.5, alpha:float=0.1, rho:float=0.0001, epsilon:float=0.6, beta:float=0.1):
        
        assert self._validate_params(alpha, beta, epsilon, theta, rho)[0],\
        'alpha, beta, epsilon, theta, rho must be of type "float"'
        assert self._validate_params(alpha, beta, epsilon, theta, rho)[1],\
        'slope parameters alpha, beta must non-zero'
        assert self._validate_attrib(pos, radius, valence, potential)[0],\
        'position must be tuple of ints or floats'
        assert self._validate_attrib(pos, radius, valence, potential)[1],\
        'radius must be float or int and greater than 0'
        assert self._validate_attrib(pos, radius, valence, potential)[2],\
        'valence must be boolean, 1, or -1'
        assert self._validate_attrib(pos, radius, valence, potential)[3],\
        'potential must be float and less than 1'
        
        self.position = np.random.multivariate_normal([0,0], np.eye(2)) if pos is None else pos
        self.radius = radius
        self.valence = 1 if valence else -1
        self.potential = potential
        
        self.theta = theta
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        self.beta = beta
        
    def __repr__(self):
        return (
            ('Neuron(\n pos={pos}, radius={rad},\n' +
             ' valence={val}, potential={pot},\n' +
             ' theta={theta}, alpha={alpha},\n' +
             ' rho={rho}, epsilon={eps}, beta={beta} \n)')
            .format(pos=self.position, rad=self.radius, 
                    val=self.valence, pot=self.potential, 
                    theta=self.theta, alpha=self.alpha, 
                    rho=self.rho, eps=self.epsilon, beta=self.beta)
        )
        
    def _validate_params(self, alpha, beta, epsilon, theta, rho)->list:
        float_chk = all([type(p) in [float, int] for p in [alpha, beta, epsilon, theta, rho]])
        slope_chk = all([p != 0 for p in [alpha, beta]])
        return float_chk, slope_chk
    
    def _validate_attrib(self, pos, radius, valence, potential)->list:
        if pos is not None:
            pos_chk =\
            (type(pos) == tuple) & all([type(p) in [float, int] for p in pos]) & (len(pos) == 2)
        else: pos_chk = True
        rad_chk = (type(radius) in [float, int]) & (radius > 0)
        valence_chk = isinstance(valence, bool) | valence in [1, -1]   
        potential_chk = (type(potential) == float) & (potential < 1)
        return pos_chk, rad_chk, valence_chk, potential_chk        

    def firing_rt(self)->float:
        return 1/(1 + np.exp((self.theta-self.potential)/self.alpha))
    
    def _outgrowth_fn(self, F:float)->float:
        return 1 - 2/(1 + np.exp((self.epsilon-F)/self.beta))
#     def _outgrowth_fn(self)->float:
#         return self.rho*(1 - 2/(1 + np.exp((self.epsilon-self.firing_rt())/self.beta)))


class NeuriteOutgrowth:
    
    def __init__(self, neuron_ls:list=None, H:float=0.1):
        assert self._validate_input(neuron_ls), 'neuron_ls must contain elements of type "Neuron"'

        self.neuron_ls = neuron_ls if neuron_ls is not None else [Neuron(), Neuron()] 
        self.distance_mat = self._distance_mat_fn()
        self.H = H
        
    def _validate_input(self, neuron_ls):
        return all([type(n) == Neuron for n in list(neuron_ls)]) if neuron_ls is not None else True
    
    def _distance_mat_fn(self)->np.array:
        n_cn = len(self.neuron_ls)
        dist_mat = np.zeros((n_cn,n_cn))
        for i, n in enumerate(self.neuron_ls):
            dist_mat[i,i:] =\
            [np.linalg.norm(np.subtract(n.position, m.position)) for m in self.neuron_ls[i:]]
            
        dist_mat = dist_mat + dist_mat.T
        return dist_mat
    
    def _overlap(self)->np.array:
        # for demonstration, see https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6
        
        # 1) compute d_mat:
        d0_mat = np.zeros(self.distance_mat.shape) # initialize
        # - each neuron is represented by a row; each columns reflects other neurons in ensemble
        # - entries represent distances from neuron position where chord joining intersecting 
        # circumferences cuts radius from neuron 
        for i, n in enumerate(self.neuron_ls):
            d0_mat[i,i:] = [(n.radius**2 - m.radius**2 + self.distance_mat[i,i+j]**2)/(2*self.distance_mat[i,i+j])
                              if (i+j != i) & (n.radius + m.radius > self.distance_mat[i,i+j]) else 0
                              for j, m in enumerate(self.neuron_ls[i:])
                           ]
        d_mat = np.tril(self.distance_mat) - np.tril(d0_mat.T) + np.triu(d0_mat)
        
        # 2) compute area_mat
        area0_mat = np.zeros(self.distance_mat.shape)
        for i, n in enumerate(self.neuron_ls):
            for j, m in enumerate(self.neuron_ls[i:]):
                if (i+j == i) | (self.distance_mat[i,i+j] <= m.radius - n.radius):
                    area0_mat[i,i+j] = np.pi*n.radius**2
                elif self.distance_mat[i,i+j] <= n.radius - m.radius:
                    area0_mat[i,i+j] = np.pi*m.radius**2
                elif (self.distance_mat[i,i+j] >= n.radius + m.radius):
                    area0_mat[i,i+j] = 0
                else: 
                    d2 = self.distance_mat[i,i+j] - d_mat[i,i+j] 
                    area0_mat[i,i+j] = (
                        n.radius**2 * np.arccos(d_mat[i,i+j]/n.radius) -
                        d_mat[i,i+j] * np.sqrt(n.radius**2 - d_mat[i,i+j]**2) +
                        m.radius**2 * np.arccos(d2/m.radius) -
                        d2 * np.sqrt(m.radius**2 - d2**2)
                    )
        area_mat = np.tril(area0_mat.T,-1) + area0_mat 
        return area_mat
    
    def wt(self)->np.array:
        return 1*self._overlap()
    
    def OutgrowthSys(self, t:range=range(0,1000), init_state:list=None)->dict:
        
        if init_state is None:
            init_state = [n.potential for n in self.neuron_ls] + [n.radius for n in self.neuron_ls]
        else:
            assert len(init_state) == len(self.neuron_ls)*2,\
            'Number of initial state params (# potentials + # radii) must be double number of neurons'
            
#         cnct = np.asarray([np.tril(self._overlap(),-1).sum()])
        cnct = np.asarray([self._overlap().sum()/2])
            
        state = odeint(self._OutgrowthSys, init_state, t)        
        
        for _t in range(1,len(t)):
            for i, n in enumerate(self.neuron_ls):
                n.radius = state[_t, -len(self.neuron_ls)+i]
                cnct0 = np.asarray([np.tril(self._overlap(),-1).sum()])
            cnct = np.concatenate((cnct, cnct0))

        avg_potential_arr = state[:,:len(self.neuron_ls)].mean(1)
        state_keys = [item for ls in [['potential_'+str(j), 'radius_'+str(j)] 
                                      for j in range(len(self.neuron_ls))] for item in ls]
        state_dc = {str(k):state[:,i] for i, k in enumerate(sorted(state_keys))}
        derived_dc = dict(t=np.array(t), connectivity=cnct, avg_potential=avg_potential_arr)
        return {**derived_dc, **state_dc}
    
    def _OutgrowthSys(self, state0, t):
        potential_ls = state0[:int(len(state0)/2)]
        radius_ls = state0[int(len(state0)/2):]
        state_tup_ls = [(p, r) for p, r in zip(potential_ls, radius_ls)]
        for n, s in zip(self.neuron_ls, state_tup_ls): 
            n.potential, n.radius = s[0], s[1]

        X = np.array([n.potential for n in self.neuron_ls])
        firing_rt_vc = np.array([n.firing_rt() for n in self.neuron_ls]) # firing_rt
        valence_gt0 = np.array([n.valence>0 for n in self.neuron_ls])*1   
        valence_lt0 = np.array([n.valence<0 for n in self.neuron_ls])*1
        
        # dX/dT vector for excitatory neurons (eq 5.1)
        dX_exc = -valence_gt0*X +\
        (valence_gt0*(1-X))*self.wt()@(valence_gt0*firing_rt_vc) -\
        (valence_gt0*(self.H+X))*self.wt()@(valence_lt0*firing_rt_vc)
        # dX/dT vector for inhibitory neurons (eq 5.2)
        dX_inh = -valence_lt0*X +\
        (valence_lt0*(1-X))*self.wt()@(valence_gt0*firing_rt_vc) -\
        (valence_lt0*(self.H+X))*self.wt()@(valence_lt0*firing_rt_vc)
        
        dX = dX_exc + dX_inh
        
        # compute outgrowth function G = f(potential)
        outgrowth_vc = np.array([n._outgrowth_fn(f) for n, f in zip(self.neuron_ls, firing_rt_vc)])
        # dR/dT vector
        dR = [n.rho*g for n, g in zip(self.neuron_ls, outgrowth_vc)]

        return list(dX) + list(dR)