#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.integrate import odeint

class Neuron:
    """Neuron class with attributes for determining growth and retraction of neuritic 
    field based on activation levels against a homeostatic threshold. 
    The Rewiring Brain, Ooyen & Butz-Ostendorf (2017). 
    
    A Neuron instance alters the radial extent of its neuritic field as a function 
    of its firing rate relative to a threshold parameter (epsilon). The firing rate
    of a Neuron instance is itself a function of its membrane potential.
    
    Parameters
    ------------
    pos: tuple {(x, y) | -Inf < x, y < Inf }
        - position of Neuron in 2D space. If no values are provided, pos is randomly 
        sampled from a 2D Gaussian with mean [0,0] and an identity covariance matrix.   
    radius : float {r | 0 < r}, default=0.25
        - initial radial extent of the neuritic field
    valence: bool {True | False} or {l | -1}, default=1
        - parameter determining whether Neuron is excitatory (True or 1) or inhibitory (False or -1) 
    potential: float {p | p <= 1}, default=0.0
        - normalized Neuron membrane potential with max of 1 
    theta: float {t | t <= 1}, default=0.5
        - threshold parameter controlling Neuron membrane potential value for which firing rate method
        evaluates to 0.5
    alpha: float {a | 0 < a}, default=0.1
        - scaling parameter controlling steepness of firing rate method   
    rho: float {r | 0 < r}, default=0.0001
        - scaling parameter controlling rate of change of neuritic field growth function
    epsilon: float {e | 0 < e < 1}, default=0.6
        - threshold parameter reflecting Neuron homeostatic value for which growth function method
        evaluates to 0
    beta: float {b | 0 < b}, default=0.1
        - scaling parameter controlling steepness of growth function method
        
    Attributes
    ------------
    position: tuple {(x, y) | -Inf < x, y < Inf }
        - Position of neuron in 2D space
    radius: float {r | 0 < r}, default=0.25
        - radius of neuritic field
    valence: 1 or -1
        - excitatory (1) or inhibitory (-1) valence of Neuron 
    potential: float {r | 0 < r}
        - normalized Neuron membrane potential (max of 1)
        
    theta: float {t | t <= 1}
        - firing rate parameter: Neuron membrane potential value for which firing rate method
        evaluates to 0.5
    alpha: float {a | 0 < a}
        - firing rate parameter: scaling parameter controlling steepness of firing rate method   
    rho: float {r | 0 < r}
        - outgrowth function parameter: scaling parameter controlling rate of change of neuritic 
        field growth function
    epsilon: float {e | 0 < e < 1}
        - outgrowth function parameter: Neuron homeostatic value for which growth function method
        evaluates to 0
    beta: float {b | 0 < b}, default=0.1
        - outgrowth function parameter: scaling parameter controlling steepness of growth function method 
    """
    
    def __init__(self, position:tuple=None, 
                 radius:float=0.25, 
                 valence:bool=True, 
                 potential:float=0.0,  
                 theta:float=0.5, 
                 alpha:float=0.1, 
                 rho:float=0.0001, 
                 epsilon:float=0.6, 
                 beta:float=0.1):
        
        assert self._validate_params(alpha, beta, epsilon, theta, rho)[0],\
        'alpha, beta, epsilon, theta, rho must be of type "float"'
        assert self._validate_params(alpha, beta, epsilon, theta, rho)[1],\
        'slope parameters alpha, beta must non-zero'
        assert self._validate_attrib(position, radius, valence, potential)[0],\
        'position must be tuple of ints or floats'
        assert self._validate_attrib(position, radius, valence, potential)[1],\
        'radius must be float or int and greater than 0'
        assert self._validate_attrib(position, radius, valence, potential)[2],\
        'valence must be boolean, 1, or -1'
        assert self._validate_attrib(position, radius, valence, potential)[3],\
        'potential must be float and less than 1'
        
        self.position = np.random.multivariate_normal([0,0], np.eye(2)) if position is None else position
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
            (f'Neuron(\n pos = {self.position},\n' + 
             f' radius = {self.radius},\n' +
             f' valence = {self.valence},\n' +
             f' potential = {self.potential},\n' +
             f' theta = {self.theta},\n' + 
             f' alpha = {self.alpha},\n' +
             f' rho = {self.rho},\n' + 
             f' epsilon = {self.epsilon},\n' + 
             f' beta = {self.beta} \n)')
        )
        
    def _validate_params(self, alpha, beta, epsilon, theta, rho)->list:
        """Range and type checks for firing rate and growth rate functions"""
        float_chk = all([type(p) in [float, int] for p in [alpha, beta, epsilon, theta, rho]])
        slope_chk = all([p > 0 for p in [alpha, beta]])
        return float_chk, slope_chk
    
    def _validate_attrib(self, position, radius, valence, potential)->list:
        """Range and type checks for position, radius, valence, and potential parameters"""
        if position is not None:
            pos_chk = (
                (type(position) == tuple) & 
                all([type(p) in [float, int] for p in position]) & 
                (len(position) == 2)
            )
        else: pos_chk = True
        rad_chk = (type(radius) in [float, int]) & (radius > 0)
        valence_chk = isinstance(valence, bool) | valence in [1, -1]   
        potential_chk = (type(potential) == float) & (potential < 1)
        return pos_chk, rad_chk, valence_chk, potential_chk        

    def firing_rt(self)->float:
        """Firing rate as a function of Neuron membrane potential Neuron().potential"""
        return 1/(1 + np.exp((self.theta-self.potential)/self.alpha))
    
    def _outgrowth_fn(self, F:float)->float:
        """Outgrowth as a function of Neuron firing rate Neuron().firing_rt()"""
        return 1 - 2/(1 + np.exp((self.epsilon-F)/self.beta))


class NeuriteOutgrowthNetwork:
    """Ensemble of interconnected Neurons that can influence each other's activation levels
    and indirectly influence the extent of each other's neuritic field radii.
    The Rewiring Brain, Ooyen & Butz-Ostendorf, (2017).
    
    An OutgrowthSystem instance is an ensemble of Neurons able to dynamically influence and 
    alter each other's membrane potential and neuritic field. 
    
    Parameters
    ------------
    neuron_ls: list {[N1, N2, ..., Nn] | Ni = Neuron()}, default=[Neuron(), Neuron()]
        - a list of elements of type Neuron()
    H: float {h | h < 1}, default=0.1
        - H represents ratio of inhibitory to excitatory saturation potential, effectively placing
        a lower bound on Neuron().potential during dynamic simulation. This parameter is only 
        relevant when the OutgrowthSystem contains inhibitory neurons.
        
    Attributes
    ------------
    neuron_ls: list {[N1, N2, ..., Nn] | Ni = Neuron()}
        - a list of elements of type Neuron()
    distance_mat: numpy.array
        - a 2D numpy.array with entry {r,d} representing the distance from Neuron() r to Neuron() d
    H: float {h | h < 1}
        - ratio of inhibitory to excitatory saturation potential; only relevant when the 
        OutgrowthSystem contains inhibitory neurons.
    """
    
    def __init__(self, neuron_ls:list=None, H:float=0.1):
        assert self._validate_input(neuron_ls), 'neuron_ls must contain elements of type "Neuron"'

        self.neuron_ls = neuron_ls if neuron_ls is not None else [Neuron(), Neuron()] 
        self.distance_mat = self._distance_mat_fn()
        self.H = H
        
    def __len__(self) -> int:
        return len(self.neuron_ls)
    
    def __getitem__(self, position:int):
        return self.neuron_ls[position]
        
    def _validate_input(self, neuron_ls):
        """Type check for elements of neuron_ls"""
        return all([type(n) == Neuron for n in list(neuron_ls)]) if neuron_ls is not None else True
    
    def _distance_mat_fn(self)->np.array:
        """Computes distances between Neuron() elements of neuron_ls"""
        n_cn = len(self.neuron_ls)
        dist_mat = np.zeros((n_cn,n_cn))
        for i, n in enumerate(self.neuron_ls):
            dist_mat[i,i:] =\
            [np.linalg.norm(np.subtract(n.position, m.position)) for m in self.neuron_ls[i:]]
            
        dist_mat = dist_mat + dist_mat.T
        return dist_mat
    
    def _overlap(self)->np.array:
        """Computes area of overlap between Neuron() elements of neuron_ls"""
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
        """Computes synaptic weight between Neuron() elements of neuron_ls"""
        return 1*self._overlap()
    
    def OutgrowthSys(self, t:range=range(0,1000), init_state:list=None)->dict:
        """Simulates dynamic activation and neuritic outgrowth of Neuron() elements of neuron_ls
        
        Parameters
        ------------
        t: iterable, default=range(0,1000)
            - time sequence over which dynamic simulation will run
        init_state: list, default=[n.potential for n in self.neuron_ls] + [n.radius for n in self.neuron_ls]
            - list of length 2*len(neuron_ls) with elements 1, ..., len(neuron_ls) containing Neuron().potential
            values of Neuron() elements of neuron_ls; and elements len(neuron_ls)+1, ..., 2*len(neuron_ls) containing 
            Neuron().radius values of elements of neuron_ls
            
        Returns 
        ------------
        {**derived_dc, **state_dc}: dict
            - dictionary contains following keys:value pairs: 
            t:numpy.array(t), 
            connectivity:[sequence of connectivity metric], 
            avg_potential:[sequence of average_potential],  
            potential_0:[sequence of Neuron().potential for first element of neuron_ls], ...
            radius_0:[sequence of Neuron().radius for first element of neuron_ls], ...
        """
        
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
        """Dynamic system of equations governing changes to Neuron().potential and 
        Neuron().radius for elements of neuron_ls"""
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