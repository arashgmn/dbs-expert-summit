#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2

import time

from archs import *
from mass import *
from utils import *

from scipy.fft import fft, rfft, irfft, rfftfreq
from scipy import signal,interpolate
from scipy import optimize

import matplotlib.pyplot as plt

# from pdb import set_trace

# import seaborn as sns
# sns.set()
# sns.set_palette('colorblind',n_colors=9)


# # Short Intro about `archs` and `mass` files
# `archs` includes many different architectures of mass models of different type all having
# - a (signed boolean) adjececncy matrix (1 if excitatort, -1 if inhibitory, and 0 otherwise),
# - a list of the population names. 
# Of our interest is `BasalGanglia_MullerRobinson` that specifies the architecture of the network. The strength of each link must be scaled (**by a non-negative number**) later.
# 
# `mass` contains implementaiton of two classes: 
# - `MassPopulation`: a class representing different masses with their own configuration
# - `MassNetwork`: a class that is mainly used here to simulate the network deterministically (or stochastically).

# # Reproduction of van Albada & Robinson 2009
# Please read the documentaiton of the `MassPopulation` class for clarificaiton on the connectivity. I find it more general to define the afferents inputs as 
# 
# $$\Delta V_{i \to j} = \Delta V_{ij} = N_i N_j p_{ij} g(V_i) K_i $$
# 
# in which
# - $N_i$ is the number of neurons in population `i`
# - $p_{ij}$ is the probability of connection between any neuron from `i` to `j`
# - $g(V_i)$ is the activation function of population `i`, usually a sigmoid
# - $K_i$, the time-integrated voltage change due to a spike from population `i`. (In theory, it can be also a function of the target population `j` too.)
# 
# That is, connectivity is determined by the connection probability, populations' size, and spike strengths. However, many of these coeffecients can be merged together. In my implementation of `MassPopulation`, afferents are modelled as:
# 
# $$\Delta V_{ij} = (N_i K_i) (p_{ij} N_j) g(V_i) := (KN)_{i} W_{ij} g(V_i)$$
# 
# with $KN$ representing the overal strength of outcomming signals from a population, and $W_{ij}$ being the average number of connection each neuron of `i` makes with population `j`. Van Albada & Robinson, factor it differently. In their notation
# 
# $$\nu_{ij} = s_{ij} N_{ij} \sim (K_i) (N_i N_j p_{ij})$$
# 
# with $\nu$ interpreted as the connectivity which they have reported. Thus, to reproduce their work I initiate my masses by setting the `KN` attribute of all instances to 1 and later, scaling my connectivty matrix by what the have reported. Note that, the architecture of their network is already available in `archs.py`.
# 

# In[2]:


tau_d = 1./160  # seconds
tau_r = 1./640 # seconds
r = 3.8e-3  # Volts
gamma = 125. # Hz
phi_n = 10 # Hz
phi_n_std = 2 #Hz

rectify = False 

param_Pyr = {'thr':14e-3, 'KN':1, 'nu_max': 300} # Pyr (e)
param_Int = {'thr':14e-3, 'KN':1, 'nu_max': 300} # Int (i) ++ thr and KN (their Q) is not reported ++
param_D1  = {'thr':19e-3, 'KN':1, 'nu_max': 65 } # D1  (d1)
param_D2  = {'thr':19e-3, 'KN':1, 'nu_max': 65 } # D2  (d2)
param_GPi = {'thr':10e-3, 'KN':1, 'nu_max': 250} # GPi (p1)
param_GPe = {'thr': 9e-3, 'KN':1, 'nu_max': 300} # GPe (p2)
param_STN = {'thr':10e-3, 'KN':1, 'nu_max': 500} # STN (zeta)
param_RET = {'thr':13e-3, 'KN':1, 'nu_max': 500} # RET (r)
param_REL = {'thr':13e-3, 'KN':1, 'nu_max': 300} # REL (s) ++ KN (their Q) isn't reported ++


configs = {}
for pop in ['Pyr','Int','D1','D2','GPi','GPe','STN','RET','REL']:
    config = eval('param_'+pop)
    config['tau_r'] = tau_r
    config['tau_d'] = tau_d
    config['r'] = r
    config['rectify'] = rectify
    
    
    if pop =='REL':
        config['input_ext'] = phi_n * 0.5e-3 
        config['noise'] = phi_n_std* 0.5e-3
    
    if (pop=='Pyr'):
        config['field'] = True
        config['gamma'] = gamma
    
    configs[pop] = config


# In[3]:


# these are index of each population similar to the paper
i_ = 0
e_ = 1
d1_= 2
d2_= 3
p1_= 4
p2_= 5
zeta_ = 6
r_ = 7
s_ = 8

# vanAlbada_Robinson_2009
W =[
    [-1.9,-1.9,   0,   0,    0,   0,   0,  0,   0], # source: Int (i)
    [1.6 , 1.6, 1.0,  .7,    0,   0, 0.1,.15, 0.8], # source: Pyr (e)
    [ 0  , 0  ,-0.3,   0, -0.1,   0,   0,  0,   0], # source: D1  (d1)
    [ 0  , 0  ,   0,-0.3,    0, -.3,   0,  0,   0], # source: D2  (d2)
    [ 0  , 0  ,   0,   0,    0,   0,   0,  0,-.03], # source: GPi (p1)
    [ 0  , 0  ,   0,   0, -.03, -.1,-.04,  0,   0], # source: GPe (p2)
    [ 0  , 0  ,   0,   0,  0.3, 0.3,   0,  0,   0], # source: STN (zeta)
    [ 0  , 0  ,   0,   0,    0,   0,   0,  0, -.4], # source: RET (r)
    [ 0.4, 0.4, 0.1, .05,    0,   0,   0,.03,   0], # source: REL (s)
]

# later on we can load it simply by using the functions in `utils.py`
# W = load_VA_connectivity()


# **NOTE**: Rows represent the source pop and cols the targets. i.e., `W[i,j]` shows the coupling from `i` to `j`. **This is exactly opposite of the convention of the paper!** 

# In[4]:


# Define the architecture and network
arch = BasalGanglia_MullerRobinson()
net = MassNetwork(arch=arch, 
                order=2,         # order of the dynamical system of each mass model. 
                configs=configs, # configuration of each mass model specified above
                W=abs(np.array(W))*1e-3,  # the coupling must be positive and correctly scales to mV.s
                noisy=False,     # We don't want stochastic integration now
                 )


# In[5]:


# For finding the steady state case, let's simulate the network for 3 seconds
init = np.zeros(net.size)
t0 = 0 
dur = 3
dt = 1e-3
out = net.simulate(t0, dur, dt, ic = init,) 


# In[6]:


plot_v_nu(net, net.t, out, print_stats=True)


# Which gives the same results as in the paper. Let us also check the same for different parkinsonian modes. All we need to do is to specify the cases stated in the paper as an argument to the untility function `parkinsonian_maker`. Let us check to "full parkinsonian" mode which corresponds to case "V". You're welcome to check other cases yourself.

# In[7]:


arch = BasalGanglia_MullerRobinson()
W, configs = parkinsonian_maker('V', np.array(W), configs) #full parkinsonian
net = MassNetwork(arch=arch, 
                order=2,         # order of the dynamical system of each mass model. 
                configs=configs, # configuration of each mass model specified above
                W=abs(np.array(W))*1e-3,  # the coupling must be positive and correctly scales to mV.s
                noisy=False,     # We don't want stochastic integration now
                 )


# In[8]:


# For finding the steady state case, let's simulate the network for 3 seconds
init = np.zeros(net.size)
t0 = 0 
dur = 3
dt = 1e-3
out = net.simulate(t0, dur, dt, ic = init,) 
plot_v_nu(net, net.t, out, print_stats=True)


# All the numbers (after two digit rounding) match those reported by van Albada & Robinson 2009. Note that, since everything is stationary, power spectrum won't give us any interesting info. But, check the noisy system by chaning the `noisy` flag to `True`. The following function, plots the PSDs and the time-frequency graphs of all populations. You may want to play a bit with the noise intensity, Or add noise or input on other populations.

# In[9]:


# plot_time_freq(net, net.t, out, slc=0.5, sim_dur=dur)


# # Model response types
# Now that we have a similar (*but not identical*) model, let's see how changing the decay rate of the mass models alters the response of the system. However, before that, let's explore the system more. Our model is non-linear. Thus, it may exhibit interesting behavoirs. So far we have seen that there are at least as set of parameters that the system is attracted to a fixed point. We can see by simple exploration, that the following reponses are also possible:
# 
# - Oscillatory (unimodal, multimodel)
# - Chaotic 
# 
# Let us again define the configuration and save them as default. Later, we edit them to generate theses cases.

# In[10]:


# van Albada
tau_d = 1./160  # seconds
tau_r = 1./640 # seconds

r = 3.8e-3  # Volts
gamma = 125. # Hz
phi_n = 10 # Hz
phi_n_std = 2 #Hz

rectify = False 

param_Pyr = {'thr':14e-3, 'KN':1, 'nu_max': 300} # Pyr (e)
param_Int = {'thr':14e-3, 'KN':1, 'nu_max': 300} # Int (i) 
param_D1  = {'thr':19e-3, 'KN':1, 'nu_max': 65 } # D1  (d1)
param_D2  = {'thr':19e-3, 'KN':1, 'nu_max': 65 } # D2  (d2)
param_GPi = {'thr':10e-3, 'KN':1, 'nu_max': 250} # GPi (p1)
param_GPe = {'thr': 9e-3, 'KN':1, 'nu_max': 300} # GPe (p2)
param_STN = {'thr':10e-3, 'KN':1, 'nu_max': 500} # STN (zeta)
param_RET = {'thr':13e-3, 'KN':1, 'nu_max': 500} # RET (r)
param_REL = {'thr':13e-3, 'KN':1, 'nu_max': 300} # REL (s)


configs = {}
for pop in ['Pyr','Int','D1','D2','GPi','GPe','STN','RET','REL']:
    config = eval('param_'+pop)
    config['tau_r'] = tau_r
    config['tau_d'] = tau_d
    config['r'] = r
    config['rectify'] = rectify
    
    
    if pop =='REL':
        config['input_ext'] = phi_n * 0.5e-3 
        config['noise'] = phi_n_std* 0.5e-3
    
    if (pop=='Pyr'):
        config['field'] = True
        config['gamma'] = gamma
    
    configs[pop] = config

# default configs
configs0 = copy.deepcopy(configs)

# default connectivity
W0 = load_VA_connectivity()

# for later convenience
i_ = 0
e_ = 1
d1_= 2
d2_= 3
p1_= 4
p2_= 5
zeta_ = 6
r_ = 7
s_ = 8


# ## Oscillatory (single mode)
# In this case, the parameters produce a harmonic oscillation of all voltages with the same rate. It is refleted in the time series, and also the power spectrum.

# In[11]:


configs = copy.deepcopy(configs0)
W = W0.copy()


# In[12]:


# change mass configs
tau_d = 1./505  # seconds
tau_r = 1./200 # seconds

configs = copy.deepcopy(configs0)
for conf in configs.values():
    conf['tau_d'] = tau_d
    conf['tau_r'] = tau_r

# change connectivity configs
W[s_, r_]*=0.6
W[s_, i_]*=0.9 # these two alone can induce gamma oscillation
W[s_, e_]*=1.1 # these two alone can induce gamma oscillation


# In[13]:


arch = BasalGanglia_MullerRobinson()
net = MassNetwork(arch=arch, order=2, configs=configs,
                  W=abs(np.array(W))*1e-3, # to convert to mV s
                  noisy=False, 
                 )
init = np.zeros(net.size)
t0 = 0 
dur= 2
dt = 2.5e-3
slc = 0.5
out = net.simulate(t0, dur, dt, ic = init,) 

plot_v_nu(net, net.t, out, )# extra='plt.xlim(1,1.25)')
plt.savefig('single-harmonic.png',dpi=300, bbox_inches='tight')
plot_psd(net, net.t, out, slc=slc)
plt.savefig('single-harmonic-psd.png',dpi=300, bbox_inches='tight')
# find_harmonic(net, net.t, out, how='acf',prominence=3 )


# ## oscillatory response (higher modes)
# In this case, there are more than one oscillation. Note the voltage traces. You can spot slow and fast oscillations easily. Also note that not all oscillations are in-phase.

# In[14]:


configs = copy.deepcopy(configs0)
W = W0.copy()


# In[15]:


# no change in config

# change connectivity configs (same as single-mode)
W[s_, r_]*=0.6
W[s_, i_]*=0.9 # these two alone can induce gamma oscillation
W[s_, e_]*=1.1 # these two alone can induce gamma oscillation


# In[16]:


arch = BasalGanglia_MullerRobinson()
net = MassNetwork(arch=arch, order=2, configs=configs,
                  W=abs(np.array(W))*1e-3, # to convert to mV s
                  noisy=False, 
                 )
init = np.zeros(net.size)
t0 = 0 
dur= 2
dt = 2.5e-3
slc = 0.5
out = net.simulate(t0, dur, dt, ic = init,) 

plot_v_nu(net, net.t, out, )# extra='plt.xlim(1,1.25)')
plt.savefig('multi-harmonic.png',dpi=300, bbox_inches='tight')
plot_psd(net, net.t, out, slc=slc)
plt.savefig('multi-harmonic-psd.png',dpi=300, bbox_inches='tight')
# find_harmonic(net, net.t, out, how='acf',prominence=3 )


# ## Choatic response
# In this case, there exist an attractor in the dynamical system that voltages wonder around, but they never quite reach a steady oscillation pattern. The system is chaotic.

# In[17]:


configs = copy.deepcopy(configs0)
W = W0.copy()


# In[18]:


# change mass config
tau_d = 1./50  # seconds
tau_r = 1./200 # seconds
for conf in configs.values():
    conf['tau_d'] = tau_d
    conf['tau_r'] = tau_r

# change connectivity 
W[p2_, p2_] *= 1e-4
W[s_, r_] *= 1e-4


# In[19]:


arch = BasalGanglia_MullerRobinson()
net = MassNetwork(arch=arch, order=2, configs=configs,
                  W=abs(np.array(W))*1e-3, # to convert to mV s
                  noisy=False, 
                 )
init = np.zeros(net.size)
t0 = 0 
dur= 2
dt = 2.5e-3
slc = 0.5
out = net.simulate(t0, dur, dt, ic = init,) 


plot_v_nu(net, net.t, out, )
plt.savefig('chaotic.png',dpi=300, bbox_inches='tight')
plot_psd(net, net.t, out, slc=slc,)
plt.savefig('chaotic-psd.png',dpi=300, bbox_inches='tight')
# find_harmonic(net, net.t, out, how='acf',prominence=3 )


# # Parameter sweep and Beta power ratio
# We keep all the parameters as in the van Albada & Robinson 2009 except the decay time of the masses. By exploration, we oberved combinations of rise and decay produce completely different responses, some showing a beta peak and some not. Thus, we kept the rise time constant and swept the decay timescale such in the range of 0.001 to 1 seconds, corresponding to $0.64\tau_r$ to $64\tau_r$, respectively.
# 
# For every combination of $\tau_r, \tau_d$ we make two networks, corresponding to the healty and pathological states, and compute their total power in the beta frequency range (12-30 Hz). This ratio is a proxy for labeling the parameters set as *P-conformal* (conforming to parkinsonian beta exchange biomarker) or *non-P-conformal*. Finally we plot this ratio for all structures and see if there are multitude of P-conformal regions. 
# 
# > SPOILER ALERT: There are!

# In[20]:


configs_n = copy.deepcopy(configs0)
W_n = W0.copy() 
W_p, configs_p = parkinsonian_maker('V', np.array(W_n), configs_n) #full parkinsonian


# In[25]:


tau_r = configs0['STN']['tau_r']
tau_ds = np.linspace(.64, 64, 100, endpoint=True)*tau_r

Ntau = len(tau_ds)

outs_n = []
outs_p = []

t0 = 0 
dur= 3
dt = 2.5e-3
for i, tau in enumerate(tau_ds):
    for pop in configs_n.keys():
        configs_n[pop]['tau_d'] = tau
        configs_p[pop]['tau_d'] = tau
    
    arch = BasalGanglia_MullerRobinson()
    net_n = MassNetwork(arch=arch, order=2, configs=configs_n,
                      W=abs(np.array(W_n))*1e-3, noisy=False,)
    net_p = MassNetwork(arch=arch, order=2, configs=configs_p,
                      W=abs(np.array(W_p))*1e-3, noisy=False,)

    start = time.time()
    out_n = net_n.simulate(t0, dur, dt, ic = np.zeros(net_n.size),)
    out_p = net_p.simulate(t0, dur, dt, ic = np.zeros(net_p.size),)
    
    # this function just extracts the voltages traces and ignores the remaining state variable
    outs_n.append(extract_v(net_n, out_n))
    outs_p.append(extract_v(net_p, out_p))
    
    print('i={} finished in {} seconds.\n'.format(i,round(time.time()-start,3)))


# In[26]:


fs = 1./dt
ratios = -1*np.ones((len(tau_ds), len(net_n.pop_names)))
slc = net_n.t>0.5 # drop out the first 0.5 seconds of simulation

for i in range(len(tau_ds)):
    for pop in range(len(net.pop_names)):
        freq, psd_n = signal.periodogram(outs_n[i][pop][slc], fs)
        freq, psd_p = signal.periodogram(outs_p[i][pop][slc], fs)
        
        cond = (freq <=30) & (freq >=12)
        pow_n = psd_n[cond].sum()
        pow_p = psd_p[cond].sum()
        
        ratios[i,pop] = pow_p/pow_n


# In[28]:


# np.save('ratios.npy', ratios)
ratios = np.load('ratios.npy')


# In[29]:


# plot_power_ratio(ratios, net, tau_ds, boolean=True)
l = plt.imshow(np.log(ratios).T, interpolation='None',
               cmap='seismic', vmin=-.2, vmax=.2 ,)
plt.colorbar(l, extend='both', location='top', ticks=[])
plt.axis('tight')
plt.yticks(ticks=range(min(ratios.shape)), labels=net.pop_names);
plt.xticks(ticks=range(0,100,20), labels=[0, 2e-2, 4e-2, 6e-2, 8e-2])
plt.xlabel(r'$\tau_d$ [s]')
plt.vlines(14, -1, 10, linestyle= '--', color='w')
# plt.xlim(-1, 99)
# plt.savefig('00-beta_ratio_wrong.png',dpi=300, bbox_inches='tight')


# Note the that for STN for instance, there are periods of high beta ratio and low beta ratio. Also for GPe/GPi, at very large decay tiemscales, such periodic windows appear too. Let's look at these high beta power windows of STN and see how the PSD looks like in the beta range. 

# In[30]:


slc = net_n.t>0.5
plt.figure()
    
for case in [22, 39, 60]: # the window centers (roughly)
    nuc = 6 # index of STN
    freq,  normal = signal.periodogram(outs_n[case][nuc][slc],fs)
    freq,abnormal = signal.periodogram(outs_p[case][nuc][slc],fs)

    l = plt.semilogy(freq, normal, label=r"$\tau_d$={} s".format(np.round(tau_ds[case], 3)))
    plt.semilogy(freq, abnormal, '--', color=l[0].get_color())
    
    # let's also print it
    cond = (freq <=30) & (freq >=12)
    pow_n = normal[cond].sum()
    pow_p = abnormal[cond].sum()
    print('For tau_d = {} the ratio is {}.'.format(tau_ds[case], pow_p/pow_n))

plt.plot([],[], 'k', label='healthy')
plt.plot([],[], 'k--', label='Parkinsonian')
plt.legend(fontsize='x-large')

plt.yscale('linear')
plt.xlim(12,30)
plt.ylim(0,6e-4)

plt.xlabel('Frequency [Hz]',fontsize='large')
plt.ylabel('Power Spectrum [arb. unit.]',fontsize='large');
# plt.savefig('more_beta.jpg',dpi=300, bbox_inches='tight')

