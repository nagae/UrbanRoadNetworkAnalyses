import numpy as np
import scipy as sp
from scipy import sparse, optimize
from operator import itemgetter, attrgetter
import pandas as pd
import re
import time
import glob

default_root = '.'
net_suffix = 'net.tntp'
trips_suffix = 'trips.tntp'

def_max_itr = 1000
def_TOL_Z_rate = 1e-6
def_TOL_x_rate = 1e-4
def_TOL_pi_rate = 1e-4

#
# Parameters for the solution algorithms
#
class FW_Param:
    # initializer 
    def __init__(self, max_itr=def_max_itr, # maximum iteration
                     TOL_Z_rate=def_TOL_Z_rate, # TOL for the change rate in the objective function
                     TOL_x_rate=def_TOL_x_rate, # TOL for the total change rate in x
                     TOL_pi_rate=def_TOL_pi_rate): # TOL for the total change rate in the shortest travel time
        self.max_itr, self.TOL_Z_rate, self.TOL_x_rate, self.TOL_pi_rate = \
            max_itr, TOL_Z_rate, TOL_x_rate, TOL_pi_rate
#
# Network class
#
class Network :
    #
    # initializer
    #
    def __init__(self, root=default_root, dir_name='', net=None, trips=None):
        if net is None or trips is None:
            self.setup(root, dir_name)
            return
        self.net, self.trips = net, trips
        init = list(net.iloc[:,0].astype('int').values)
        term = list(net.iloc[:,1].astype('int').values)
        self.links_bn = tuple(zip(init,term))
        self.nodes_bn = sorted({i for l in self.links_bn for i in l})
        self.num_nodes = len(self.nodes_bn)
        self.num_links = len(self.links_bn)
        self.nid_of = {node_bn:i for (i, node_bn) in enumerate(self.nodes_bn)}
        self.nodes = range(self.num_nodes)
        self.links = tuple(((self.nid_of[init], self.nid_of[term]) for (init, term) in self.links_bn))
        self.lid_of = {l:i for (i, l) in enumerate(self.links)} # インデックス・ペアからリンク・インデックスを取得
        self.capacity = net.iloc[:,2].values.astype('float')
        self.length = net.iloc[:,3].values.astype('float')
        self.fftt = net.iloc[:,4].values.astype('float')
        self.b = net.iloc[:,5].values.astype('float')
        self.power = net.iloc[:,6].values.astype('float')
    #
    # setup from given directory
    #
    def setup(self, root, dir_name):
        net_fname = glob.glob('%s/%s/*_net*' % (root, dir_name))[0]
        trips_fname = glob.glob('%s/%s/*_trips*' % (root, dir_name))[0]
        print("net file:\t%s" % net_fname)
        print("trips file:\t%s" % trips_fname)
        net = read_net(net_fname)
        trips = read_trips(trips_fname)
        self.__init__(net=net, trips=trips)
    #
    # Link cost function and its sparse.csr_matrix representation
    #
    def t(self, x=None):
        if x is None: # if x is omitted, it return the free flow travel time (i.e. t(x=0))
            return self.fftt
        return self.fftt*(1 + self.b*(x/self.capacity)**self.power)
    def t_mat(self, x=None):
        return sparse.csr_matrix((self.t(x), zip(*self.links)))
    
    #
    # Integral of the link performance function and the objective function of the equivalent convex programming 
    #
    def T(self, x):
        return self.fftt*x*(1+self.b/(1+self.power)*(x/self.capacity)**self.power)
    def Z(self, x):
        return sum(self.T(x))
    
    #
    # Shoretst travel time for every O-D pairs (for convergence test)
    #
    def ShortestTravelTime(self, w):
        ret = np.zeros((self.num_nodes, len(self.trips)))
        for oid, orig_bn in enumerate(self.trips):
            orig = self.nid_of[orig_bn]
            ret[:,oid] = sparse.csgraph.dijkstra(w,indices=orig)
        return ret

    
    #
    # All-or-Nothing assignment
    # 
    # All-or-Nothing assignment for a single origin
    def AoN_O2M(self, w, orig_bn):
        # obtain the shortest travel time to each node as well as the shortest-path tree
        orig = self.nid_of[orig_bn]
        pi,pred = sparse.csgraph.dijkstra(w,indices=orig,return_predecessors=True)
        
        x = np.zeros(self.num_links) # Link flow of All-or-Nothing assignment
        X = np.zeros(self.num_nodes) # Total traffic for each node to the corresponding subtree on the shortest path
        for j, pi_j in sorted(enumerate(pi),key=itemgetter(1,0), reverse=True): # sort the destinations by the shortest path in reverse order
            j_bn = self.nodes_bn[j] # 
            i = pred[j] # predecessor of node j
            if i == -9999:
                #if j == orig: continue
                #if j_bn in ntw.trips[orig_bn] and ntw.trips[orig_bn][j_bn] > 0:
                #    print("trip demand for O-D pair(%d,%d) can not be assigned"%(orig_bn,j_bn))
                continue
            lid = self.lid_of[i,j]
            x[lid] += X[j]
            if j_bn in self.trips[orig_bn]:
                x[lid] += self.trips[orig_bn][j_bn]
            X[i] += x[lid]
        return x
    # All-orNothing assignment for every origin
    def AoN(self, w):
        x = np.zeros(self.num_links)
        for orig_bn in self.trips:
            x += self.AoN_O2M(w, orig_bn)
        return x
    # Find the initial solution as an AoN with the free-flow travel time
    def AoN_init(self):
        return self.AoN(self.t_mat())
    
    #
    # Linear search for the optimal step-size 
    #
    def FindAlpha(self, x, y):
        g = lambda alpha: self.Z((1-alpha)*x + alpha*y)
        # res = optimize.minimize_scalar(g, bounds=[0,1], method='golden') # If you would like to use the golden section method
        res = optimize.minimize_scalar(g, bounds=[0,1])
        return res.x
    
    # 
    # Solve the User Equilibrium
    #
    def SolveUE(self, prm=None, return_log=False):
        if prm is None:
            prm = FW_Param()
        # lists for log
        if return_log:
            Zs = list() # objective function at each iteration
            Z_rates = list() # change rate in the objective function
            x_rates = list() # change rate in the link flow
            pi_rates = list() # change rate in the shortest travel time
        # initialize
        t0 = self.t_mat()
        x0 = self.AoN(t0)
        pi0 = self.ShortestTravelTime(t0)
        Z0 = self.Z(x0)
        if return_log:
            Zs.append(self.Z(x0))
        # main loop
        for itr in range(prm.max_itr):
            # Obtain the auxliary solution as the all-or-nothing assignment
            y = self.AoN(self.t_mat(x0))
            # Find the best step size
            alpha = self.FindAlpha(x0,y)
            # Update the tentative solution
            x = x0 + alpha*(y-x0)
            pi = self.ShortestTravelTime(self.t_mat(x))
            Z = self.Z(x)
            # Calculate change rates 
            Z_rate = (Z0-Z)/Z0
            x_rate = np.sqrt(np.sum((x0-x)**2))/(np.sum(x0))
            pi_rate = sum(abs(pi[pi0>0]-pi0[pi0>0])/pi0[pi0>0])
            # Update logs
            if return_log:
                Zs.append(Z)
                Z_rates.append(Z_rate)
                x_rates.append(x_rate)
                pi_rates.append(pi_rate)
            # Convergence check
            if Z_rate < prm.TOL_Z_rate or x_rate < prm.TOL_x_rate or pi_rate < prm.TOL_pi_rate:
                break
            # Memorize the current solution
            x0 = x.copy()
            pi0 = pi.copy()
            Z0 = Z
        if return_log:
            return x, (Zs, Z_rates, x_rates, pi_rates)
        else:
            return x
            

#
# read trip data from ***_trips.tntp
#
def read_trips(fname):
    with open(fname, 'r') as f:
        trips = {}
        all_rows = f.read()
        blocks = all_rows.split('Origin')[1:]
        for k in range(len(blocks)):
            orig_dest = blocks[k].split('\n')
            orig = int(orig_dest[0])
            trips_to_dest = {}
            for d in ''.join(orig_dest[1:]).split(';'):
                m = re.search(r"(.+):(.+)", d)
                if m:
                    dest, val = int(m.group(1)), float(m.group(2))
                    trips_to_dest[dest] = val
            trips[orig] = trips_to_dest
        return trips

#
# read network data from ***_net.tntp
#
def read_net(fname):
    skip_rows = 0
    with open(fname, 'r') as f:
        while True:
            line = f.readline()
            if ('~') in line:
                break
            else:
                skip_rows += 1
    with open(fname, 'r') as f:
        net = pd.read_csv(f, sep="\t", skiprows=skip_rows)
        net.dropna(how="all", inplace=True)
        # lowercase and strip unnecessary spaces for each column
        net.columns = map(str.lower, net.columns)
        net.columns = map(str.strip, net.columns)
        # drop the first column if it is redundant
        while 'init node' not in net.columns[0] and 'tail' not in net.columns[0] and 'from' not in net.columns[0]:
            net.drop(net.columns[0], axis='columns', inplace=True)
        # drop the redundant first/last column
        net.drop(net.columns[-1], axis='columns', inplace=True)

        # return net.dropna().iloc[1:] # iloc[1:]とすると最初のリンクが失われる
        return net.dropna() # 

#
# Demo
#
if __name__ == '__main__':
    t0 = time.time()
    root = 'data'
    dir_name = 'SiouxFalls'
    ntw = TNTP.Network(root, dir_name)
    x = ntw.SolveUE()
    t1 = time.time()
    print("t1-t0:", t1-t0)
    
