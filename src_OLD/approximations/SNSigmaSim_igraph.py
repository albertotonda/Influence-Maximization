import igraph
import random
import numpy
import math
import time

def IC_model(G, a, p):              # a: the set of initial active nodes
                                    # p: the system-wide probability of influence on an edge, in [0,1]
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False               # G: can be either directed or undirected

    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n, mode=igraph.OUT)) - A:
                prob = random.random()	# in the range [0.0, 1.0)
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B

    return len(A)

def WC_model(G, a):                 # a: the set of initial active nodes
                                    # each edge from node u to v is assigned probability 1/in-degree(v) of activating v
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
 
    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n, mode=igraph.OUT)) - A:
                prob = random.random()	# in the range [0.0, 1.0)
                p = 1.0 / G.degree(m, mode=igraph.IN)
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B

    return len(A)

"""
Evaluates a given seed set A, simulated "no_simulations" times.
Returns a tuple: the mean, stdev, and 95% confidence interval.
"""
def evaluate(G, A, p, no_simulations, model):
    results = []

    if model == 'WC':
        for i in range(no_simulations):
            results.append(WC_model(G, A))
    elif model == 'IC':
        for i in range(no_simulations):
            results.append(IC_model(G, A, p))

    return numpy.mean(results), numpy.std(results), 1.96 * numpy.std(results) / math.sqrt(no_simulations)
