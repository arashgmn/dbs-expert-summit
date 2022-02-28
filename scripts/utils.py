import time
import numpy as np
import matplotlib.pyplot as plt

from mass import *

from scipy.fft import fft, rfft, irfft, rfftfreq
from scipy import signal,interpolate
from scipy import optimize


from pdb import set_trace
# import seaborn as sns
# sns.set()
# sns.set_palette('colorblind',n_colors=9)

def compute_stats(net, out):
    means = np.zeros(shape=len(net.pop_names))
    stds = np.zeros(shape=len(net.pop_names))
    
    for i, pop in enumerate(net.pop_names):
        if not net.noisy:
            v = out.y[net.idx_of_pop[pop]] 
        else:
            v = out[:, net.idx_of_pop[pop]]
        nu = net.pops[i].g(v)
        means[i] = nu.mean()
        stds[i] = nu.std()
    return means, stds


def run_seq(net, dt=1e-3, init_dur= 0.5, sim_dur= 2.0, N = 100):
    """
    runs `N` sequential simulation of length `sim_dir` after an initialization
    phase of duration `init_dur`.
    """
    
    sim_dur_idx = int(sim_dur/dt)
    timeseries = np.zeros((len(net.pop_names), int(sim_dur*N/dt)+1))
    t = np.linspace(0, init_dur + sim_dur*N, int(sim_dur*N/dt)+1, endpoint=True)
    v_idx = list(net.idx_of_pop.values())
    
    # initialize
    init = np.zeros(net.size)
    out = net.simulate(t0 = 0, dur = init_dur, dt = dt, ic = init)
    if not net.noisy:
        out = out.y
    timeseries[:,0] = out[v_idx,-1] 
    
    # run simulations
    for n in range(N):
        out = net.simulate(t0 = round(net.t[-1],6), dur = sim_dur, 
                           dt = dt, contin=True, interpolate=True)
        if not net.noisy:
            out = out.y
        timeseries[:,1 + n*sim_dur_idx: 1+ (n+1)*sim_dur_idx] = out[v_idx,1:] 
    
    return t, timeseries


def extract_v(net, out):
    try:
        v = out['y'] # not-noisy and not sequential
    except:
        v = out # noisy or sequential
    if len(v)>len(net.pops): # noisy and not sequenctial
        v = v[list(net.idx_of_pop.values())]
    return v

def filt(v, fs, N=4, w_lp=150):
    b, a = signal.butter(N, w_lp, 'low', fs=fs)
    return signal.filtfilt(b, a, v)
    
    
def plot_v_nu(net, t, out, slc=0.5, lp_filter=False, 
              print_stats=True, save_name=None, extra=None):
    """
    plots the v and nu. Also prints statistics of for t>0.5 s.
    """
    
#     if (not net.noisy) and (not seq):
#         v = out['y'][list(net.idx_of_pop.values())].copy()
    slc= t>slc
    v = extract_v(net, out)
    if lp_filter:
        fs = 1./(t[1]-t[0])
        v = filt(v, fs)
            
    fig, axs = plt.subplots(2,1, sharex=True, figsize=(12,5))
    for i, pop in enumerate(net.pop_names):
        nu = net.pops[i].g(v[i])
        axs[0].plot(t, v[i], label=pop)
        axs[1].plot(t, nu, label=pop)

        if print_stats:
            print(pop+': mean_v= %1.5e, std_v= %1.5e | mean_nu= %.2f, std_nu=%.2f'\
                  %(v[i][slc].mean(), v[i][slc].std(), nu[slc].mean(), nu[slc].std()))
                  
    axs[0].set_ylabel('voltage')
    axs[1].set_ylabel('firing rate')

    for ax in axs:
        ax.legend(loc='upper right',ncol=3)
    ax.set_xlabel('time [s]')
              
    if type(extra)!=type(None):
        eval(extra)
        
    if type(save_name)!= type(None):
        plt.savefig('nuv_'+save_name+'.jpg',dpi=300, 
                    bbox_inches='tight')
        
def plot_time_freq(net, t, out, slc=0.5, sim_dur=2.,
                   lp_filter=False,
                   save_name=None,
                   extra=None):
#     v = out.copy()
#     if not net.noisy:
#         v = out['y'].copy()
#     if not seq:
#         v = v[list(net.idx_of_pop.values())]
#     v = out.copy()
#     if (not net.noisy) and (not seq):
#         v = out['y'][list(net.idx_of_pop.values())].copy()
    
    slc= t>slc
    v = extract_v(net, out)
    dt = t[2]-t[1]
    fs = 1./dt
    nseg = int(sim_dur/dt)
    if lp_filter:
        v = filt(v, fs)
    
    fig, axs = plt.subplots(9,3,figsize=(12,20), sharex='col')
    for k, pop in enumerate(net.pop_names):
            
        x = net.pops[k].g(v[k][slc]) 
        freqs, times, Sxx = signal.spectrogram(x, fs, )#nperseg=nseg, noverlap=0)
        
        axs[k,0].plot(t[slc], x, label=net.pop_names[k], color='C'+str(k))
        axs[k,1].pcolormesh(times, freqs, Sxx, shading='gouraud')

        Sxx_m = Sxx.mean(axis=1)
        Sxx_s = Sxx.std(axis=1)
        axs[k,2].semilogx(freqs, Sxx.mean(axis=1))
        axs[k,2].fill_between(freqs, Sxx_m+Sxx_s, Sxx_m-Sxx_s, alpha=0.4)


        axs[k,1].set_ylim(0,100)
        axs[k,0].set_ylabel('Firing rate [Hz]')
        axs[k,1].set_ylabel('Frequecny [Hz]')
        axs[k,2].set_xlabel('Firing rate [Hz]')
        axs[k,2].set_ylabel('Mean Power Density')

        axs[k,0].legend(loc='upper left')

    #         ax.set_title(w_2_pop[k])
    axs[k,0].set_xlabel('Time [s]')
    axs[k,1].set_xlabel('Time [s]')
    plt.tight_layout()
    
    if type(extra)!=type(None):
        eval(extra)
    
    if type(save_name)!= type(None):
        plt.savefig('time_freq_'+save_name+'.jpg',dpi=300, 
                    bbox_inches='tight')
        
def plot_psd(net, t, out, slc=0.5,
             lp_filter=False, 
             save_name=None,
             extra=None):
    slc= t>slc
    v = extract_v(net, out)
    dt = t[2]-t[1]
    fs = 1./dt
    if lp_filter:
        v = filt(v, fs)
            
#     v = out.copy()
#     if not net.noisy:
#         v = out['y'].copy()
#     if not seq:
#         v = v[list(net.idx_of_pop.values())]
    
#     v = out.copy()
#     if (not net.noisy) and (not seq):
#         v = out['y'][list(net.idx_of_pop.values())].copy()
    
#     v = extract_v(net, out)
#     slc= t>slc
    
#     dt = t[3]-t[2] # the initial dt may be different 
#     fs = 1./dt # integration resolution = sampling rate
    
    fig = plt.figure(figsize=(12,3))
    mean_pow = None
    for k, pop in enumerate(net.idx_of_pop.keys()):
        freq, power = signal.periodogram(v[k][slc], fs)
        plt.loglog(freq, power, label=pop)
        
        if type(mean_pow)==type(None):
            mean_pow = power
        else:
            mean_pow += power
        # find period from the frequency of the first peak
#         T = 1./power[power.argmax()]
#         T_idx = int(T/dt)
#         print(pop+' has a period of {} which span {} index'.format(T, T_idx))
    mean_pow /= len(net.pops)
    plt.loglog(freq, mean_pow,'--k', label='mean')
    plt.legend(ncol=2)
    plt.ylim(1e-16,1)
    plt.tight_layout()
    
    if type(extra)!=type(None):
        eval(extra)
    
    if type(save_name)!= type(None):
        plt.savefig('psd_'+save_name+'.jpg',dpi=300, 
                    bbox_inches='tight')
    
    
def find_harmonic(net, t, out, slc=0.5,
                 lp_filter=False, 
                 prominence=3, how='acf', beta_mode=True,
                 plot = True,
                 save_name=None,
                 extra=None):
    """
    Sums up the PSD of all populations and find their peaks as a proxy to find
    the harmonics. There are two method:
    
    - acf: Computes the first non-zero peak of auto-correlation function. It 
           corresponds to the fastest oscillation always. However, there is 
           no guarantee that the detected oscillation mathces the underlying
           oscillations if there are several harmonics of different strengths.
           In that case, using `acf` method is questionable as it is not able
           to find the *lowest* harmonic, rather a strong overtone of it.
    
    - fft: Detects the peaks of the power spectrum and computes the frequency
           difference between each two consecutive peak. The distance between
           the first two peaks is taken as the minimum harmonic. In case of 
           several harmonics, it infers the harmonics wrongly.
    
    NOTE
    ----
    The first method is more robust w.r.t the prominence value. Also in case
    of chaotic behavior, it always detects the strongest coherent frequency.
    Yet, in these two scenarios, usage of single harmonics is questionable and
    the detected harmonic may not have a clear meaning.
    """
    slc= t>slc
    v = extract_v(net, out)
    dt = t[2]-t[1]
    fs = 1./dt
    if lp_filter:
        v = filt(v, fs)
#     v = out.copy()
#     if not net.noisy:
#         v = out['y'].copy()
#     if not seq:
#         v = v[list(net.idx_of_pop.values())]

#     v = out.copy()
#     if (not net.noisy) and (not seq):
#         v = out['y'][list(net.idx_of_pop.values())].copy()
    
#     v = extract_v(net, out)
#     slc= t>slc
    
#     dt = t[3]- t[2]  
#     fs = 1./dt     
    
#     if filt:
#         b, a = signal.butter(N, w_lp, 'low', fs=fs)
        
    y_sum = None
    # we use the sum of all powers as a proxy. It's more rebust than one.
    for i in range(len(net.pops)):
        sig =v[i][slc]
            
        if how=='fft':
            x, y = signal.periodogram(sig, fs,) # freq, power
        else:
            y = signal.correlate(sig, sig, mode='same', method='fft') # acf
            
        # for numerical stability I scale and clip
        y = (y-y.min())/(y.max()-y.min())
        y = np.clip(y, 1e-7, 1)
        
        if type(y_sum)==type(None):
            y_sum=y
        else:
            y_sum+=y
    
    if how=='acf':
        x = signal.correlation_lags(len(sig), len(sig), mode='same')#time lags
        x = x[len(y)//2:]         # auto-correlation is symmetric
        y_sum = y_sum[len(y)//2:] # auto-correlation is symmetric
        
        peaks, _ = signal.find_peaks(y_sum, prominence=prominence/3., )
        
    if how=='fft':
        peaks, _ = signal.find_peaks(np.log(y_sum),prominence=prominence)
        
    result = 0
    if how=='fft':
        if len(np.diff(x[peaks])):
            result = np.diff(x[peaks]).max() # it is in frequency
    else:
        if len(peaks):
            result = 1./(x[peaks][0]*dt)

#   
    
    if plot:
        plt.figure(figsize=(12,4))
        plt.semilogy(x,y_sum/y_sum.max())
        plt.semilogy(x[peaks], y_sum[peaks]/y_sum.max(),'xk')
        
        if how=='fft':
            plt.xlabel('Freq [Hz]')
            plt.ylabel('Power')
        if how=='acf':
            plt.xlabel('Lag')
            plt.ylabel('Auto-correlation function')

        plt.tight_layout()
        
    if type(extra)!=type(None):
        eval(extra)
    
#       else:
#         if how=='fft':
#             if len(x[peaks]):
#                 tmp = x[peaks]
#             else: # no peak at all
#                 tmp = np.array([0])
#         else:
#             tmp = 1./(x[peaks]*dt)
        
#         result = ((tmp>=12) & (tmp<=30)).sum().astype(bool)
        
    return result

def load_VA_connectivity():
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

    return np.array(W)

def parkinsonian_maker(case, W, configs):
    
    #W = load_VA_connectivity()
    i = 0
    e = 1
    d1= 2
    d2= 3
    p1= 4
    p2= 5
    zeta = 6
    r = 7
    s = 8
    
    if case =='I':
        Chi = 0.6 #mV
        h = 10 #Hz

        W[e,d1] -= Chi
        W[e,d2] -= Chi

        configs['D1']['thr'] -= h*Chi*1e-3 
        configs['D2']['thr'] -= h*Chi*1e-3 

    elif case =='II':
        # xi = 0.1
        # W[e,d1] *= (1-xi) # direct
        # W[e,d2] = (1+xi) # indirect
        W[e,d1] = 0.5 # direct
        W[e,d2] = 1.4 # indirect

    elif case=='III':
        W[p2,p2] = -0.03

    elif case=='IV':
        W[e,e] = W[e,i] = 1.4
        W[i,e] = W[i,i] = -1.6

    elif case=='V':
        W[e,d1] = 0.5 # direct
        W[e,d2] = 1.4 # indirect

        W[p2,p2] = -0.07

        W[e,e] = W[e,i] = 1.4 
        W[i,e] = W[i,i] = -1.6

        configs['GPe']['thr'] = 8e-3 
        configs['STN']['thr'] = 9e-3 

        W[d2,p2] = -0.5

    return W, configs

def sweep_configs(params,  
                  arch, W, configs,  
                  p_mode='V', 
                  t0 = 0, dur=3, dt=2.5e-3, record='voltage',):
    """
    - params: dict 
        each key must match whatever parameter in the configuration that is 
        supposed to change. The values of the dict, must specify the desired
        sweeping values. If more than one keys exist, each sweep is done
        independent of the parameter sweep of the other keys. 
    - arch: object
        the architecture object. Look at `archs` module
    - W: 2D numpy array
        the connectivty matrix of the healty state in mV s. The absolute values
        of the connection strengths would be used. Thus, can have negative or
        positive entries.
    - configs: dict
        the configuration of the mass populations in healty state. Refer to the
        documentation of the `MassPopulation` in the `mass` module.
    - p_mode: str
        mode of parkinsonian according to the [1]. Default 'V' that corresponds
        the full parkinsonian. Either ['I', 'II', 'III', 'IV' ,'V'] are 
        possible.
    - t0: float
        start time of the simulation in seconds
    - dur: float
        duration of the simulation in seoncds
    - dt: float
        timestep of the simulation in seconds
    - record: str
        either 'voltage' or 'all'. The former extracts only voltages from the 
        state variables. The latter returns all (including derivatives).

    """
    output = {}
    for param in params.key():
        output[param] = [[], []]
    
    # these are the reference configuraiton files
    W_n0, configs_n0 = W, configs
    W_p0, configs_p0 = parkinsonian_maker(p_mode, np.array(W), configs) 
    
    net_n = MassNetwork(arch=arch, order=2, configs=configs_n0,
                        W=abs(np.array(W_n0))*1e-3, noisy=False,)
    net_p = MassNetwork(arch=arch, order=2, configs=configs_p0,
                        W=abs(np.array(W_p0))*1e-3, noisy=False,)
    init = np.zeros(net_n.size)
    
    
    # for param in params.key():
    #     print('I am sweeping '+param)
    #     param_range = params[param]

    #     for i, par in enumerate(param_range):
    #         for pop in configs_n.keys():
    #             configs_n[pop]['tau_d'] = tau
    #             configs_p[pop]['tau_d'] = tau
            
    #         arch = BasalGanglia_MullerRobinson()
    #         net_n = MassNetwork(arch=arch, order=2, configs=configs_n,
    #                         W=abs(np.array(W_n))*1e-3, noisy=False,)
    #         net_p = MassNetwork(arch=arch, order=2, configs=configs_p,
    #                         W=abs(np.array(W_p))*1e-3, noisy=False,)

    #         start = time.time()
    #         out_n = net_n.simulate(t0, dur, dt, ic = np.zeros(net_n.size),)
    #         out_p = net_p.simulate(t0, dur, dt, ic = np.zeros(net_p.size),)
            
    #         #this function just extracts the voltages traces and ignores the remaining state variable
    #         outs_n.append(extract_v(net_n, out_n))
    #         outs_p.append(extract_v(net_p, out_p))
            
    #         print('i={} finished in {} seconds.\n'.format(i,round(time.time()-start,3)))
    #     out_n = []
    #     out_p = []



    
    
    
def compute_power_ratio(net, t, out_p, out_n, slc=0.5, fmin=12, fmax=30):
    """
    Finds the ratio between the total power in the beta band in different ppos
    by summing up the psd between `fmin` and `fmax`.
    
    `net` could be the network defined for any of the normal or parkinsonian 
    case. `out_*` arguments, must be only the  populations' voltages and not
    the intermediate state variales, e.g. derivatives.
    """
    
    dt = t[3]-t[2]  
    fs = 1./dt     
    
#     v = out.copy()
#     if not net.noisy:
#         v = out['y'].copy()
#     if not seq:
#         v = v[list(net.idx_of_pop.values())]
#   
    v = out.copy()
    if (not net.noisy) and (not seq):
        v = out['y'][list(net.idx_of_pop.values())].copy()
    slc= t>slc
    
    rel_power = None
    # we use the sum of all powers as a proxy. It's more rebust than one.
    for i in range(len(net.pops)):
        sig =v[i][slc]
        freq, power = signal.periodogram(sig, fs,) # freq, power
        beta_band = (freq<=fmax) & (freq>=fmin)
        power = power[beta_band].sum()/(baseline[i])  # relative power
        if type(rel_power)==type(None):
            rel_power= power
        else:
            rel_power+= power
    return rel_power


def plot_power_ratio(ratios, net, parameter, transpose=True, 
                     log=False, boolean=False,
                     save_name=None, extra=None):
    ratios_ = ratios.copy()
    if transpose:
        ratios_ = ratios_.T
    if log:
        ratios_ = np.log(ratios_)
    
    if not boolean:
        mn, mx = ratios_.min(), ratios_.max()
        vlim = min(abs(mn), abs(mx))
        l = plt.imshow(ratios_, cmap='seismic', vmin=-vlim, vmax=vlim)
    else:
        l = plt.imshow(ratios_>1, cmap='seismic')
        
    plt.colorbar(l,)
    plt.axis('tight')
    plt.yticks(ticks=range(min(ratios_.shape)), labels=net.pop_names);
    plt.xticks(ticks=range(max(ratios_.shape)), labels=np.round(parameter,3),
               rotation=90);
    
    plt.tight_layout()
    
    if type(extra)!=type(None):
        eval(extra)
    
    if type(save_name)!= type(None):
        plt.savefig('pow_ratio_'+save_name+'.jpg',dpi=300, 
                    bbox_inches='tight')