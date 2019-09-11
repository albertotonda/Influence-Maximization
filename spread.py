import networkx as nx
import random
import numpy
import math

""" Simulation of spread for Independent Cascade (IC) and Weighted Cascade (WC). 
    Suits (un)directed graphs. 
    Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
"""
def IC_model(G, a, p):              # a: the set of initial active nodes
                                    # p: the system-wide probability of influence on an edge, in [0,1]
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False

    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A: # G.neighbors follows A-B and A->B edges
                prob = random.random() # in the range [0.0, 1.0)
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
 
    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree

    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random.random() # in the range [0.0, 1.0)
                p = 1.0/my_degree_function(m)
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B

    return len(A)

""" Evaluates a given seed set A, simulated "no_simulations" times.
	Returns a 3-tuple: (the mean, stdev, and 95% confidence interval).
"""
def simulation(G, A, p, no_simulations, model):
	results = []

	if model == 'WC':
		for i in range(no_simulations):
			results.append(WC_model(G, A))
	elif model == 'IC':
		for i in range(no_simulations):
			results.append(IC_model(G, A, p))

	return numpy.mean(results), numpy.std(results), 1.96 * numpy.std(results) / math.sqrt(no_simulations)

if __name__ == "__main__":

	G = nx.path_graph(100)
	print(nx.classes.function.info(G))
	print(simulation(G, [0, 2, 4, 6, 8, 10], 0.1, 100, 'IC'))
