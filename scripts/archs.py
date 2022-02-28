import numpy as np

class BasalGanglia(object):
    def __init__(self, has_self_striatal=True):
        self.pops = ['PyrS', 'PyrM', 'PyrD', 'Int',
                    'D1','D2',
                    'GPi', 'GPe', 'STN', 
                    'Thl']
        self.crtx = self.pops[:4]
        self.strtm = self.pops[4:6]
        
        self.has_self_striatal=has_self_striatal
        # the same order as pops (source --> target)
        
        self.make_adj_martix()
        
        
    def make_adj_martix(self):
        self.adj = [
            # source: PyrS
            [-1, 1, 1, 1,
            0, 0,
            0, 0, 0,
            0],
            # source: PyrM
            [1, -1, 0, 1,
            0, 0,
            0, 0, 0,
            0],
            # source: PyrD
            [1, 0, -1, 1,
            1, 1,
            0, 0, 1,
            1],
            # source: Int
            [-1, -1, -1, -1,
            0, 0,
            0, 0, 0,
            0],
            # source: D1
            [0, 0, 0, 0,
            0, 0,
            -1, 0, 0,
            0],
            # source: D2
            [0, 0, 0, 0,
            0, 0,
            0, -1, 0,
            0],
            # source: GPi
            [0, 0, 0, 0,
            0, 0,
            0, 0, 0,
            -1],
            # source: GPe
            [0, 0, 0, 0,
            0, 0,
            -1, 0, -1,
            0],
            # source: STN
            [0, 0, 0, 0,
            0, 0,
            1, 1, 0,
            0],
            # source: Thl
            [0, 0, 1, 1,
            0, 0,
            0, 0, 0,
            0],
        ]
        if self.has_self_striatal:
            self.adj[4][4] = -1
            self.adj[5][5] = -1
        
    
    def get_adjacency(self, asnumpy=True, boolean=False):
        tmp = np.array(self.adj)
        if asnumpy:
            if boolean:
                return tmp.astype(bool)
            else:
                return tmp
        else:
            if boolean:
                return list(tmp.astype(bool))
            else:
                return list(tmp)

            
class BasalGanglia_1pop1mass(object):
    """
    according to:
    https://www.lidsen.com/journals/neurobiology/neurobiology-05-02-095
    """
    def __init__(self, assume_balance):
        self.pops = ['CTX',
                    'D1','D2',
                    'SNc','STN',
                    'GPi','GPe',  
                    'Thl']
        
        self.assume_balance = assume_balance
        # the same order as pops (source --> target)
        self.adj = self.make_adj_martix()
        
    def make_adj_martix(self):
        adj = [
            # source: CTX
            [0, 1, 1, 0, 1, 0, 0, 1], 
            # source: D1
            [0, 0, 0, 0, 0, -1, 0, 0], 
            # source: D2 
            [0, 0, 0, 0, 0, 0, -1, 0], 
            # source: SNc
            [0, 1, -1, 0, 0, 0, 0, 0], 
            # source: STN
            [0, 0, 0, 1, 0, 1, 1, 0],
            # source: GPi
            [0, 0, 0, 0, 0, 0, 0, -1],
            # source: GPe
            [0, 0, 0, 0, -1, -1, 0, 0],
            # source: Thl
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]
        
        if self.assume_balance:
            adj[0][0] = -1 # cortex  inhibits itself
            adj[3][1] = -1 # SNc effectively inhibits D1
            adj[3][3] = -1 # SNc can inhibits itself
            
            
        return np.array(adj)
        
    def get_adjacency(self, boolean=False):
        if boolean:
            return self.adj.astype(bool).copy()
        else:
            return self.adj.copy()
        
        
class BasalGanglia_MullerRobinson(object):
    """
    according to:
    https://www.lidsen.com/journals/neurobiology/neurobiology-05-02-095
    """
    def __init__(self):
        self.pops = ['Int','Pyr', #CTX
                    'D1','D2',
                    'GPi','GPe',  
                    'STN',
                    'RET','REL' #Thl
                    ] 
        
        self.adj = self.make_adj_martix()
        
    def make_adj_martix(self):
        adj = [
            [-1,-1, 0, 0, 0, 0, 0, 0, 0], # source: Int (i)
            [ 1, 1, 1, 1, 0, 0, 1, 1, 1], # source: Pyr (e)
            [ 0, 0,-1, 0,-1, 0, 0, 0, 0], # source: D1  (d1)
            [ 0, 0, 0,-1, 0,-1, 0, 0, 0], # source: D2  (d2)
            [ 0, 0, 0, 0, 0, 0, 0, 0,-1], # source: GPi (p1)
            [ 0, 0, 0, 0,-1,-1,-1, 0, 0], # source: GPe (p2)
            [ 0, 0, 0, 0, 1, 1, 0, 0, 0], # source: STN (zeta)
            [ 0, 0, 0, 0, 0, 0, 0, 0,-1], # source: RET (r)
            [ 1, 1, 1, 1, 0, 0, 0, 1, 0], # source: REL (s)
        ]
            
        return np.array(adj)
        
    def get_adjacency(self, boolean=False):
        if boolean:
            return self.adj.astype(bool).copy()
        else:
            return self.adj.copy()
        
        
class CompeteingEI(object):
    def __init__(self, has_self_connection=True):
        self.pops = ['E', 'I']
        self.has_self_connection = has_self_connection
        self.adj = self.make_adj_martix()
        
    def make_adj_martix(self):
        adj = [[0, 1],[-1, 0]]
        if self.has_self_connection:
            adj[0][0] = 1
            adj[1][1] = -1
        return np.array(adj) 
    
    def get_adjacency(self, boolean=False):
        if boolean:
            return self.adj.astype(bool)
        else:
            return self.adj.copy()