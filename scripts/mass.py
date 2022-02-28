import numpy as np
from scipy.optimize import linprog
from scipy.integrate import solve_ivp
from sdeint import itoint
from pdb import set_trace
import copy

class MassPopulation(object):
    """
    A generic neural population suited for mass network model. 
    In mass model, each population contains `N` neuron which
    emit action potential of strenght `K` to other neurons with
    a rate that is determined by a sigmoidal activation function
    that maps the average voltage of the population to a firing
    rate. 

    Dynamics of the population is governed by a first or second
    order differential equation which is driven by exogenous 
    inputs and afferents from self or other populations. This 
    dynamics has a typical timescale `tau` (for first order or
    second order alpha kernel), or a rise ande decay timescale.
    
    Parameters
    ----------
    tau_r : float
        rise timescale (s)
    tau_d : float
        decay timescale (s)
    tau : float
        if specified, both rise and decay timescales will be equated to it 
        (corresponds to the alpha kernel). Always overrides `tau_r` and `tau_d`!
    KN : float
        strength of all synaptic connections multiplied by the 
        total number of neurons within the poulations (V)
    nu_max: float
        maximum firing rate
    input_ext : float 
        equivalent exogenous input voltage (V)
    noise: float
        the intensity (std) of an additive input white noise (V)
    thr: float
        threshold of the activation function; the inflation 
        point for a sigmoid (V)
    r: float
        steepness of the activation function at the inflation
        point of the activation fucntion (V)
    quiet_eq: bool
        if `True`, neuron will fire with a rate zero at zero equilibrium 
        voltage. Negative voltages, thus may cause *negative firing rates*. 
        To be used in accord with `rectify`.
    rectify: bool
        whether or not the activation function should be rectified to positive
        values only (default `True`).
    field: bool
        determines how afferent are modeled. Default value `False` corresponds 
        to a classical mass model whereas the `True` activates spatially 
        homogeneous wave-like field attenuation due to axonal propagation. 
        Look below. Also cf. van Albada et all 2009a,b. 
    gamma: float
        used if `field=True` to specify the wave dampening rate.
    name: str
        the name of the population (optional)
    
    Notes
    ----
    Afferents are added to the right hand side of the dynamical system as:

    $V_{i \to j} := V_{ij} = p_{ij} N_j N_i K_i g(v_i) = w_{ij}(NK)_i g(v_i)$
    
    if the population does not have a significant axonal length (low 
    propagation). In this view, coupling term $p_{ij}$ determines the 
    connection probability between population `i` with `N_i` neurons to 
    population `j` with size `N_j`. We have absorbed the population specific 
    term (N*K) in a new parameter called `KN`. The connectivity 
    $w_{ij}=p_{ij} N_j$, is similarly the mean connection from population j to
    i (only if $p_{ij}$ is random).
    
    If cells have lengthy axonal harbor, a field description will determine 
    the field upon each firing, by solving:
    
    $1\gamma^2 *[d^2/dt^2 + 2*gamma d/dt + gamma^2] phi = g(v)$
    
    where gamma is the inverse of propagation time of a wave with speed v_e 
    through an axon with length r_e, $gamma = v_e/r_e$. The afferents, then 
    are similar to above, except the term $g$ will be substituted by $\phi_i$.
    """

    #TODO: delays?
    def __init__(self, 
                 tau, tau_r, tau_d, KN, nu_max, 
                 thr = 1, r = .2, 
                 quiet_eq= False, rectify = True, 
                 field = False, gamma = None, 
                 input_ext = 0, noise = 0,
                 name=None):
        
        self.name = name
        self.KN = KN # K*N
        self.input_ext = input_ext
        self.noise = noise
        
        self.tau_r = tau_r
        self.tau_d = tau_d 
        self.tau = tau      
            
        # maximum of firing rate function
        self.nu_max = nu_max
        self.thr = thr
        self.r = r
        self.rectify = rectify
        self.quiet_eq = quiet_eq 
        self.field = field 
        
        if self.field:
            assert type(gamma)!= type(None)
            self.gamma = gamma
            
    def g(self, v):
        """
        activation function; assumed to be a sigmoid with a threshold
        of `thr` and steepness of `r`.
        """
        x = (v-self.thr)/self.r
        # for stability I seperate the domaine into three regions
        nu = np.where(abs(x) < 500, 1./(1. + np.exp(-x)), x)  
        nu = np.where(nu <= -500, 0, nu)
        nu = np.where(nu >= 500, 1, nu)
        
        if self.quiet_eq:
            nu -=  1./(1. + np.exp(self.thr/self.r))
        
        if self.rectify:
            return np.where(nu >= 0, nu*self.nu_max, 0)
        else:
            return nu*self.nu_max



class MassNetwork(object):
    """
    A general class for a mass network model.

    Parameters
    ----------
    arch : architecture-object
        architecture of the network. c.f. `archs.py`
    order : int
        order of the dynamical system (only 1 or 2 are supported)
    field : bool
        if the neural field formulation is to be used (refer to van Albada et 
        al 2009)
    noisy : bool
        if inputs are noisy, then an SDE integrator is used.
    configs : dict 
        configuraitons dictionary. Must have the name of populations as key
        and a configuration dictionary per population as value. The dictionary
        of each population (values of `configs`) can be used to directly 
        specify the parameters of the population, or specify a distriution 
        from which the population parameters are sampled. The distrubtion and
        its arguments must be given as a list. Look below for an example.
    config_type: str
        only `dist` or `direct`. `direct` requires user to enter
        the (necessary) arguments for initiation of a `MassPopulation`
        object, whereas if `dist`, the agruments of `MassPopulation`
        will be drawn from the given distribution function. 
    W : np.array
        the coupling weights between each population (all component must be
        non-negative floats -- inhiitory connections are infered from `arch`)
    ei_balance: bool
        whether or not enforce the EI balance for each structure.
        If `False`, the (absolute) adjacency matrix will be
        interpreted the connectivity matrix. Otherwise a linear
        prgramming will determine the value of each weight such
        that the average input and output have the same strenght
        (in the first moment sense).
    w_min: float
        minimum connectivity value. Must be in [0,1] and less than
        `w_max`. Used if `ei_balance=True`.
    w_max: float
        maximum connectivity value. Must be in [0,1] and greater than
        `w_min`. Used if `ei_balance=True`.
    
    Note
    ----
    Enforcement of E-I balance may fail for networks which do not have
    both type of inhibitory and excitatory to each population. Use
    wisely.
    
    Note
    ----
    If all `tau_r`, `tau_d`, and `tau` are specified, while `order==2`, I 
    assume the alpha kernel, i.e., the values of rise and decay timescales
    will be overridden by the value of `tau`. 
    

    Example
    -------
    For a network with two populations `A` and `B`, one can define the
    configurations in one of the follwing way:

    `configs = {'A': dict(tau = [np.random.beta, (0.1, 0.5)],
                          KN = [np.random.normal, (1, 1e-2)]
                          ),
                'B': dict(...) }
    
    `configs = {'A': dict(tau = 0.3, KN = 0.2),
                'B': dict(...) }
    """
    def __init__(self, arch, configs, # config_type='dist',      # network and population args
                 W=None, ei_balance=False, w_min=0.1, w_max=1, # connectivity settings
                 order=2, noisy = False,        # dynamic settings
                ):
        """
        cofig should be either a set of distributions for each parameter
        or individual dicitonary for all pops. 
        """

        self.arch = arch
        self.pop_names = arch.pops
        self.configs = copy.deepcopy(configs)
        # self.config_type = config_type
        
        self.W = self.arch.get_adjacency() # unweighted
        if type(W)!=type(None):
            self.W = self.W.astype(float)*W  # scale links' weight
        self.adj = self.W.astype(bool)
        self.ei_balance = ei_balance
        self.w_min = w_min
        self.w_max = w_max
        self.adj = self.W.astype(bool)
        
        self.order = order
        self.noisy = noisy
        
        # either dist
        # config = {['dist_K', 'dist_N', 'dist_W', 'dist_tau']}
        # or actual dict per populatiopn
        # config_pop = {'K'= ... , 'N'= ..., W=..., ''}
        self.assert_config()
        self.pops = self.setup_pops()
        self.idx_of_pop, self.size = self.make_idx_mapping()
        
        if self.ei_balance:
            self.balance()
        
        # state
        self.state = None
        self.t = None
        self.dt = None
        
    def assert_config(self):
#         msg = "only 'dist' or 'direct' are possible"
#         assert (self.config_type=='dist' or self.config_type=='direct'), msg
        
        msg = "Some pops have no config! Cross-check keys with populations of your arch."
        set_pops = set(self.arch.pops)
        set_config = set(self.configs.keys())
        assert set_pops.intersection(set_config) == set_pops, msg
        del set_pops, set_config, msg
                
        for pop in self.configs.keys():
            config = self.configs[pop]
            keys = list(config.keys())
            
            # mandatory inputs
            mandatory_args = ['nu_max', 'KN', 'r', 'thr',]
            for arg in mandatory_args:
                assert arg in keys, arg+" is not in config of {}".format(pop)
                
            if self.order==1:
                assert 'tau' in keys, "tau is not in config of {}".format(pop)
            elif 'tau' not in keys:
                assert 'tau_r' in keys, "tau_r is not in config of {}".format(pop)
                assert 'tau_d' in keys, "tau_d is not in config of {}".format(pop)
            
            if 'field' in keys:
                msg = "Population {} used field model. ".format(pop)
                msg_cont = "Please provide gamma (ve/re) for this population."
                if config['field']:
                    assert 'gamma' in keys, msg+msg_cont
                
            # check types
            for key in config.keys():
                if type(config[key])==type([]):
                    func, args = config[key]
                    try:
                        func(*args)
                    except:
                        print('Error in {} population:'.format(pop))
                        print('\tfunction {} and arguments {} are not compatible.'\
                            .format(func, args))
                else:
                    try:
                        type(config[key]) in [float, bool]
                    except:
                        print('Error in {} population:'.format(pop))
                        print('The input {} must be a float or bool.'.format(key))

    def balance(self):
        """
        modifies the connectivity such that all the populations 
        are under excitatory-inhibitory balance.
        """
        
        adj = self.adj.copy() # already bool
        N_pop = len(adj)
        N_w = sum(adj.ravel())
        X = np.zeros(shape=(N_w, N_pop))

        # mapping between adj index and solution index
        mapping = {}
        w_idx= 0
        for i, src in enumerate(self.pops):
            for j, trg in enumerate(self.pops):
                if adj[i,j]:
                    X[w_idx, j] = src.KN*self.W[i,j]
                    mapping[w_idx] = (i,j)
                    w_idx += 1
                    
        c = -np.ones(N_w)   # maximize sum of weights
        b = np.zeros(N_pop) # enforce balance
        bounds = [(self.w_min, self.w_max)]*N_w
        res = linprog(c, A_eq=X.T, b_eq=b, bounds=bounds,)
        if res.success:
            for idx, w in enumerate(res.x):
                self.W[mapping[idx]] = w
        else:
            print (res.message)
            print ('Skipped balancing the weights. Originals will be used.')
    
    
#     def get_index(self, i):
#         """
#         based on the rank of the population `i`, computes the index
#         of its corresponding variable in the linear system.
#         """
#         return self.order*i


    def setup_pops(self):
        """
        assert if all parameters (populations) have a distriution 
        (configuration dictionary). If everything was OK, sets up
        the populations. Otherwise complains.  
        """
        pop_names = self.arch.pops
        pops = []
        
        for pop_name in pop_names:
            config = self.configs[pop_name]
            # every value in the config is either float, bool, or list.
            # Those that are list, provide us a distribution to sample from.
            # so I do just that.
            
            for key in config.keys():
                if type(config[key])==type([]):
                    func, args = config[key]
                    config[key] = func(*args)
            
#             # the following are bool switches
#             if 'field' in config.keys():
#                 field = config['field']
#                 gamma = config['gamma']
#             else:
#                 field = False
#                 gamma = None
            
#             if 'rectify' in config.keys():
#                 rectify = config['rectify']
#             else:
#                 rectify = False
            
#             # the following are either a float (if direct) or a list (if distr)
#             nu_max = config['nu_max']
#             KN = config['KN']
#             thr = config['thr']
#             r = config['r']
            if 'tau' in config.keys(): 
#                 tau = config['tau']
#                 tau_r = None
#                 tau_d = None
                if self.order==1:
                    config['tau_r'] = None
                    config['tau_d'] = None
                else: # in this case alpha kernel must be used: tau_r= tau_d
                    config['tau_r'] = config['tau']
                    config['tau_d'] = config['tau']
                    
            else:  # tau is a mandatory input. Must be set.
                config['tau'] = None
#                 tau = None
#                 tau_r = config['tau_r']
#                 tau_d = config['tau_d']
            
#             if 'input_ext' in config.keys():
#                 input_ext = config['input_ext']
#             else:
#                 input_ext = 0.0
            
            # if they are list, this means they should be sampled from a given
            # distribution
#             if self.config_type =='dist':
#                 nu_max = nu_max[0](*nu_max[1])
#                 KN = KN[0](*KN[1])
#                 thr = thr[0](*thr[1])
#                 r = r[0](*r[1])
                
#                 if 'tau' in config.keys():
#                     tau = tau_r[0](*tau_r[1])
#                     tau_r = None
#                     tau_d = None
#                 else:
#                     tau = None
#                     tau_r = tau_r[0](*tau_r[1])
#                     tau_d = tau_d[0](*tau_d[1])
                
#                 if type(input_ext) == type([]):
#                     input_ext = input_ext[0](*input_ext[1])
                
#                 if 'field' in config.keys():
#                     gamma = gamma[0](*gamma[1])
                    
            config['name'] = pop_name
            pop = MassPopulation(**config)
#                 name= pop_name, KN= KN, nu_max= nu_max,
#                                 tau= tau, tau_r = tau_r, tau_d = tau_d,
#                                 thr= thr, r= r, input_ext= input_ext,
#                                 rectify= rectify, 
#                                 field = field, gamma = gamma
#                                 )
            pops.append(pop)
        
        del pop, config
        return pops

    def make_idx_mapping(self):
        """
        A mapping between the name of the population and the index associated
        with the voltage of the population. 
        """
        idx = 0
        mapping = {}
        for pop in self.pops:
            mapping[pop.name] = idx
            idx += self.order
            if pop.field: # extra variables for the field
                idx+= 2
            
        return mapping, idx
    
    def deterministic_dyn(self, t, y):
        """
        computes the state grdient dy/dt given the state y. 
        
        * 1st order dynamics:
        tau dy/dy + y = I_ext + afferents

        * 2nd order dynamics:
        tau**2 d^2y/dt^2 + 2 tau dy/dt + y = I_ext + afferents
        
        computation of afferents depends status of the `field` attribute of 
        the population. If `field==False`, axonal propagation are ignored:
        
        afferents_i = sum_k W_ik * (NK)_k * g(v_k)
        
        otherwise, the g in the equation above must be replaced by a field 
        described according to:
        
        [d^2/dt^2 + 2*gamma d/dt + gamma**2] \phi_k = (gamma**2) g(v_k)
        
        NOTE: The order of dynamics is independent from the field's evolution.
        Field will be always described by a 2nd-order ODE.
        
        NOTE: the arguments order the revere relative to the deterministic one.
        So I change them in case we are solving an SDE!
        """
        if self.noisy:
            tmp = t.copy()
            t = y.copy()
            y = tmp.copy()
            del tmp
            
        ydot = np.zeros_like(y)
        
        for i, pop in enumerate(self.pops): # i is the rank of target pop
            afferents = 0
            for j, is_adj in enumerate(self.adj[:,i]): # j is rank of source pop
                if is_adj:
                    src_pop = self.pops[j]
                    idx_j = self.idx_of_pop[src_pop.name]
                    if src_pop.field:
                        afferents += self.W[j,i]*src_pop.KN*y[idx_j + self.order]
                    else:
                        afferents += self.W[j,i]*src_pop.KN*src_pop.g(y[idx_j])

            idx_i = self.idx_of_pop[pop.name]
                    
            if self.order==1:
                ydot[idx_i] = (-y[idx_i] + afferents + pop.input_ext)/pop.tau
            
            if self.order==2:
                ydot[idx_i] = y[idx_i+1]
                ydot[idx_i+1] = (-(pop.tau_r+pop.tau_r)*y[idx_i+1]- y[idx_i] + \
                                 afferents + pop.input_ext)/(pop.tau_r*pop.tau_d)
            if pop.field:
                ydot[idx_i+self.order] = y[idx_i + self.order+1]
                ydot[idx_i+self.order+1] = -2*pop.gamma* y[idx_i + self.order + 1]\
                                         + pop.gamma**2*(-y[idx_i + self.order] + pop.g(y[idx_i]) )
                
        return ydot
    
    
    def stochastic_dyn(self, y, t):
        """
        Adds a diogonal white noise to populations. This noise is assumed to
        be added to the afferents.
        
        NOTE: the arguments order the revere relative to the deterministic one!
        """
        g = np.zeros_like(y)
        for i, pop in enumerate(self.pops): # i is the rank of target pop
            W = pop.noise#/np.sqrt(self.dt) # TODO: fix this factor
            idx_i = self.idx_of_pop[pop.name]
                    
            if self.order==1:
                g[idx_i] = W/pop.tau
            if self.order==2:
                g[idx_i+1] = W/(pop.tau_r*pop.tau_d)
                
        return np.diag(g)
    
    def simulate(self, t0, dur, dt=1e-4, ic=None, 
                 contin=False, interpolate = True):
        """
        Integrates the system over a given range. If `contin==True`, the 
        integration continious from the previously computed simulation.
        Otherwise a new simulation will start from zero initial conditions or
        from `ic` if provided. If `intepolate==True`, the simulation voltages
        will be interpolated on timestamps specified by `t0, dt`, and `dur`. 
        It might be benefitial for cases in which the output size should be
        known a priori. Irrelevant for noisy simulations.  
        
        Note
        ----
        The setting of new simulations (duration, time step) can be completely 
        different from the previous simulations.
        """
        self.t = np.linspace(t0, (t0+dur), int(dur/dt)+1, endpoint=True) 
        
        if contin:
            assert type(self.state)!= type(None) 
            print("Continiuing from the previous state.")
            print("-- Start time: %.5f \t Final time %.5f"%(t0, t0 +dur))
            
            ic = self.state
            
        else:
            print("Starting a new simulation. All previous ones are cleared.")
            print("-- Start time: %.5f \t Final time %.5f"%(t0, t0 +dur))
        
            if type(ic) == type(None):
                ic = np.zeros(net.size, dtype=float) 
            else:
                msg = "Initial coniditon doens't match the system size"
                assert len(ic) == self.size, msg
                ic = np.array(ic, dtype=float)
        
        if not self.noisy:
            sol = solve_ivp(self.deterministic_dyn, [self.t[0], self.t[-1]],
                            ic, max_step = dt, first_step = dt/10.)
            self.state = sol.y[:,-1].copy()
            
            if interpolate:
                y_interp = np.zeros((len(sol.y), len(self.t)))
                for idx in range(len(sol.y)):
                    y_interp[idx,:] = np.interp(self.t, sol.t, sol.y[idx])
                
                sol.y = y_interp.copy() # override the solution
                sol.t = self.t          # override the solution
                del y_interp
        else:
            sol = itoint(self.deterministic_dyn, self.stochastic_dyn, 
                         ic, self.t)
            self.state = sol[-1, :].copy()
            sol = sol.T # shape: (Npop, Ntimestamp)
                
        return sol
