# Neurite Outgrowth Model Implementation

The Neurite Outgrowth implementation includes two class definitions to emulate the neurite outgrowth models described in *The Rewiring Brain* (Ooyen & Butz-Ostendorf, 2017). This document shows code, simulations, and analytics that highlight and replicate the behavior of the model as outlined in that textbook.

The Neurite Outgrowth model describes a dynamic system of equations to emulate the behavior of neurite growth and retraction from a collection of neurons oriented within a 2D spatial configuration. Neurite growth and retraction is activity-dependent so that high activity beyond a threshold leads to retraction of neuritic processes while activity below this threshold leads to growth of neuritic processes. A neuron's activity depends in turn on the overlap of its neuritic field with the neuritic fields of other neurons in the network.

Neurons in the network may have excitatory or inhibitory valence, so that neuritic field contact may generate higher or lower activity depending on the valence of the neuron with which neuritic fields overlap. For a detailed description of the model, see *The Rewiring Brain*, chapter 5. 

### Class Definitions

The Neurite Outgrowth model implementation relies on two class definitions. The first is the Neuron class that represents the relevant attributes of individual neurons in the network. The second is the NeuriteOutgrowthNetwork class that represents more global attributes relevant to the entire network of neurons.

## Demo and usage

In addition to standard Python packages, the class definitions require the numpy package and `odeint` function from the scipy.integrate library. For analytics, we will use the pandas library; and for visualization, we use matplotlib and seaborn libraries.


```python
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from neurite_outgrowth import Neuron, NeuriteOutgrowthNetwork 
```

#### Neuron Class 

The Neuron class includes attributes and methods with sufficient detail to describe the properties and behaviors of individuals neurons in the network. These attributes include:
1. Position (pos: tuple): the 2D coordinate position of the neuron.
2. Radius (radius: float): inital value of the radius of the neuron's neuritic field.
3. Valence (valence: {-1,1} or {False, True}): valence indicating an excitatory (1 or True) or an inhibitory (-1 or False) neuron.
4. Membrane potential (potential: float): initial value of the neuron's membrane potential.
5. Membrane potential threshold (theta: float): threshold parameter determining the membrane potential required to bring a neuron's firing rate to half of its maximum value. 
6. Steepness of firing rate function (alpha: float): parameter governing how quickly a neuron's firing rate increases as a function of its membrane potential.
7. Scaling of the neuritic field growth function (rho: float): parameter governing the time scale for the growth of a neuron's neuritic field relative to the change in it's firing rate.
8. Neuritic field growth threshold (epsilon: float): firing rate threshold parameter controlling whether a neuron will grow or retract its neuritic field.
9. Steepness of the neuritic field growth function (beta: float): a scaling parameter controlling steepness of a neuron's neuritic field growth function as a function of its firing rate.

And the Neuron public methods include:
1. firing_rt(): firing rate as a function of Neuron membrane potential Neuron().potential

#### NeuriteOutgrowthNetwork Class

The NeuriteOutgrowthNetwork class includes attributes that characterize the entire network:
1. A list of neurons in a network instance (neuron_ls: list[Neuron,]): this will default to a simple network of two neurons if no list is provided.
2. The distances between each neuron in the network instance (distance_mat: np.matrix): this matrix is extracted from the properties of each of the neurons passed in neuron_ls.
3. The lower bound on the inhibitory membrane potential: (H: float).

Public methods for the NeuriteOutgrowthNetwork class include:
1. wt(): currently this method simply passes through a matrix with elements representing areas of overlap between neuron instances in a network instance; n the future, this function may include additional functionality related to differential weighting between excitatory or inhibitory neurons and their targets.
2. OutgrowthSys(): simulates dynamic activation and neuritic outgrowth of Neuron() instances of neuron_ls.

#### Generating a list of Neurons

After importing the Neuron class, we generate a list of Neurons. All attributes of Neuron instances take default values unless specifically assigned during the creation of instances. Here we assign five Neuron instances a position in 2D coordinates and a neuritic field radius.  


```python
neuron_ls = [Neuron((0.5,0.5), radius=0.5), 
             Neuron((0.5,-0.5), radius=0.5), 
             Neuron((-0.5,-0.5), radius=0.5),
             Neuron((-0.5,0.5), radius=0.5),
             Neuron((0,0), radius=0.5)
            ]
```


```python
neuron_ls
```




    [Neuron(
      pos=(0.5, 0.5), radius=0.5,
      valence=1, potential=0.0,
      theta=0.5, alpha=0.1,
      rho=0.0001, epsilon=0.6, beta=0.1 
     ), Neuron(
      pos=(0.5, -0.5), radius=0.5,
      valence=1, potential=0.0,
      theta=0.5, alpha=0.1,
      rho=0.0001, epsilon=0.6, beta=0.1 
     ), Neuron(
      pos=(-0.5, -0.5), radius=0.5,
      valence=1, potential=0.0,
      theta=0.5, alpha=0.1,
      rho=0.0001, epsilon=0.6, beta=0.1 
     ), Neuron(
      pos=(-0.5, 0.5), radius=0.5,
      valence=1, potential=0.0,
      theta=0.5, alpha=0.1,
      rho=0.0001, epsilon=0.6, beta=0.1 
     ), Neuron(
      pos=(0, 0), radius=0.5,
      valence=1, potential=0.0,
      theta=0.5, alpha=0.1,
      rho=0.0001, epsilon=0.6, beta=0.1 
     )]



#### Populate neural ensemble

We can pass a list of Neuron instances to the NeuriteOutgrowthNetwork constructor to generate a NeuriteOutgrowthNetwork instance `overshoot_sys`.


```python
overshoot_sys = NeuriteOutgrowthNetwork(neuron_ls)
```

The internal method `overshoot_sys._overlap()` generates a matrix with elements showing the areas of overlap between each of the five neurons in our network instance. 


```python
overshoot_sys._overlap()
```




    array([[0.78539816, 0.        , 0.        , 0.        , 0.14269908],
           [0.        , 0.78539816, 0.        , 0.        , 0.14269908],
           [0.        , 0.        , 0.78539816, 0.        , 0.14269908],
           [0.        , 0.        , 0.        , 0.78539816, 0.14269908],
           [0.14269908, 0.14269908, 0.14269908, 0.14269908, 0.78539816]])



#### Simulate dynamics of neurite outgrowth/retraction

The OutgrowthSys() method takes in an array of time steps over which the simulated network is dynamically updated. The output contains a dictionary record of the membrane potential and neuritic field radii dynamics of the individual Neuron instances; average connectivity of the network; average membrane potential; and the corresponding time step for each record.


```python
# init_state = [n.potential for n in nsys.neuron_ls] + [n.radius for n in nsys.neuron_ls]
# out_dc = nsys.OutgrowthSys(t, init_state)
t = np.arange(0, 15000)
out_dc = overshoot_sys.OutgrowthSys(t)
```

#### Organize data and plot

Below, we can plot some characteristics of the network as a whole. Shown below are the aggregate network connectivity and the average membrane potential of the network over the time steps of the simulation.  


```python
state_wd_df = pd.DataFrame.from_dict(out_dc)
```


```python
_=(
    state_wd_df
    .loc[lambda df: df.t.gt(0),['t', 'connectivity', 'avg_potential']]
    .melt(id_vars='t', var_name='series', value_name='values')
    .pipe(sns.FacetGrid, row='series', aspect=3, sharey=False)
    .map(sns.lineplot, 't', 'values')
)
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_27_0.png)



```python
_=(
    state_wd_df
    .loc[lambda df: df.t.gt(0),['t', 'connectivity', 'avg_potential']]
    .assign(branch = lambda df: np.where(df.t<4000, 0, np.where(df.t>8000, 1, 2)))
    .melt(id_vars=['connectivity', 'branch'], var_name='series', value_name='abscissa')
    .pipe(sns.FacetGrid, col='series', hue='branch', aspect=2, sharex=False, sharey=True)
    .map(sns.lineplot, 'abscissa', 'connectivity')
)
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_28_0.png)


We can also reshape the data to show the membrane potential and neuritic field radii for each of the Neuron instances over the time steps of the simulation. 


```python
state_df = (
    state_wd_df.loc[:,[c for c in state_wd_df.columns if c not in ['connectivity', 'avg_potential']]]
    .pipe(pd.wide_to_long, stubnames=['potential', 'radius'], i='t', j='neuron idx', sep='_')
    .reset_index()
    .melt(id_vars=['t', 'neuron idx'], var_name='series', value_name='value')
)
```


```python
outgrowth_sim_pl=(
    state_df
    .pipe(sns.FacetGrid, col='series', hue='neuron idx', aspect=2, sharey=False)
    .map(sns.lineplot, 't', 'value')
)
_=outgrowth_sim_pl.add_legend()
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_31_0.png)


Depending on the initial conditions of the network, other network behaviors can be observed. Below, we show initial conditions of a network instance that lead to run-away connectivity, and other initial conditions that show network oscillations. 

##### Run-away connectivity


```python
mix_neuron_ls = [Neuron((0.5,0.5), radius=0.5), 
             Neuron((0.5,-0.5), radius=0.5), 
             Neuron((-0.5,-0.5), radius=0.5),
             Neuron((-0.5,0.5), radius=0.5),
             Neuron((0,0), radius=0.5, valence=False)
            ]
```


```python
runawaysys = NeuriteOutgrowthNetwork(mix_neuron_ls)
```


```python
t = np.arange(0, 25000)
runaway_dc = runawaysys.OutgrowthSys(t)
```


```python
state2_wd_df = pd.DataFrame.from_dict(runaway_dc)
```


```python
_=(
    state2_wd_df
    .loc[lambda df: df.t.gt(0),['t', 'connectivity', 'avg_potential']]
    .melt(id_vars='t', var_name='series', value_name='values')
    .pipe(sns.FacetGrid, row='series', aspect=3, sharey=False)
    .map(sns.lineplot, 't', 'values')
)
```


```python
_=(
    state2_wd_df
    .loc[lambda df: df.t.gt(0),['t', 'connectivity', 'avg_potential']]
    .assign(branch = lambda df: 
            np.where(df.t.le(5450), 0, 
                     np.where(df.t.le(7900), 1, 
                              np.where(df.t.le(11150), 2, 
                                       np.where(df.t.le(14350), 3, 4)))))
    .melt(id_vars=['connectivity', 'branch'], var_name='series', value_name='abscissa')
    .pipe(sns.FacetGrid, col='series', hue='branch', aspect=2, sharex=False, sharey=True)
    .map(sns.lineplot, 'abscissa', 'connectivity')
)
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_39_0.png)



```python
state2_df = (
    state2_wd_df.loc[:,[c for c in state2_wd_df.columns if c not in ['connectivity', 'avg_potential']]]
    .pipe(pd.wide_to_long, stubnames=['potential', 'radius'], i='t', j='neuron idx', sep='_')
    .reset_index()
    .melt(id_vars=['t', 'neuron idx'], var_name='series', value_name='value')
)
```


```python
outgrowth_sim_pl=(
    state2_df
    .pipe(sns.FacetGrid, row='series', hue='neuron idx', height=4, aspect=3.5, sharey=False)
    .map(sns.lineplot, 't', 'value')
)
_=outgrowth_sim_pl.add_legend()
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_41_0.png)



```python
_=(
    state2_df
    .pipe(sns.FacetGrid, row='neuron idx', col='series', aspect=2, sharey=False)
    .map(sns.lineplot, 't', 'value')
)
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_42_0.png)


##### Oscillations


```python
osc_neuron_ls = [Neuron((0.5,0.5), radius=0.5, epsilon=0.5), 
             Neuron((0.5,-0.5), radius=0.5, epsilon=0.6), 
             Neuron((-0.5,-0.5), radius=0.5, epsilon=0.7),
             Neuron((-0.5,0.5), radius=0.5, epsilon=0.8),
             Neuron((0,0), radius=0.5, epsilon=0.4)
            ]
```


```python
osc_sys = NeuriteOutgrowthNetwork(osc_neuron_ls)
```


```python
t = np.arange(0, 25000)
osc_dc = osc_sys.OutgrowthSys(t)
```


```python
state3_wd_df = pd.DataFrame.from_dict(osc_dc)
```


```python
_=(
    state3_wd_df
    .loc[lambda df: df.t.gt(0),['t', 'connectivity', 'avg_potential']]
    .melt(id_vars='t', var_name='series', value_name='values')
    .pipe(sns.FacetGrid, row='series', aspect=3, sharey=False)
    .map(sns.lineplot, 't', 'values')
)
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_48_0.png)



```python
_=(
    state3_wd_df
    .loc[lambda df: df.t.gt(0),['t', 'connectivity', 'avg_potential']]
    .assign(branch = lambda df: 
            np.where(df.t.le(4025), 0, 
                     np.where(df.t.le(7700), 1, 
                              np.where(df.t.le(10475), 2, 
                                       np.where(df.t.le(13425), 3, 
                                                np.where(df.t.le(14050), 4, 
                                                        np.where(df.t.le(14825), 5, 6)))))))
    .melt(id_vars=['connectivity', 'branch'], var_name='series', value_name='abscissa')
    .pipe(sns.FacetGrid, col='series', hue='branch', aspect=2, sharex=False, sharey=True)
    .map(sns.lineplot, 'abscissa', 'connectivity')
)
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_49_0.png)



```python
state3_df = (
    state3_wd_df.loc[:,[c for c in state3_wd_df.columns if c not in ['connectivity', 'avg_potential']]]
    .pipe(pd.wide_to_long, stubnames=['potential', 'radius'], i='t', j='neuron idx', sep='_')
    .reset_index()
    .melt(id_vars=['t', 'neuron idx'], var_name='series', value_name='value')
)
```


```python
outgrowth_sim_pl=(
    state3_df
    .pipe(sns.FacetGrid, row='series', hue='neuron idx', height=4, aspect=3.5, sharey=False)
    .map(sns.lineplot, 't', 'value')
)
_=outgrowth_sim_pl.add_legend()
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_51_0.png)



```python
_=(
    state3_df
    .pipe(sns.FacetGrid, row='neuron idx', col='series', aspect=2, sharey=False)
    .map(sns.lineplot, 't', 'value')
)
```


![png](neurite_outgrowth_demo_files/neurite_outgrowth_demo_52_0.png)

