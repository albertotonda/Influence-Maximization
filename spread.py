import networkx as nx
import random
import numpy
import math

""" Spread models """

""" Simulation of spread for Independent Cascade (IC) and Weighted Cascade (WC). 
	Suits (un)directed graphs. 
	Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
"""
def IC_model(G, a, p, random_generator):              # a: the set of initial active nodes
	                                # p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)                      # A: the set of active nodes, initially a
	B = set(a)                      # B: the set of nodes activated in the last completed iteration
	converged = False

	while not converged:
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A: # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random() # in the range [0.0, 1.0)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B

	return len(A)

def WC_model(G, a, random_generator):                 # a: the set of initial active nodes
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
				prob = random_generator.random() # in the range [0.0, 1.0)
				p = 1.0/my_degree_function(m)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B

	return len(A)

def IC_model_max_hop(G, a, p, max_hop, random_generator):  # a: the set of initial active nodes
	# p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)  # A: the set of active nodes, initially a
	B = set(a)  # B: the set of nodes activated in the last completed iteration
	converged = False

	while (not converged) and (max_hop > 0):
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:  # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random()  # in the range [0.0, 1.0)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B
		max_hop -= 1

	return len(A)


def WC_model_max_hop(G, a, max_hop, random_generator):  # a: the set of initial active nodes
	# each edge from node u to v is assigned probability 1/in-degree(v) of activating v
	A = set(a)  # A: the set of active nodes, initially a
	B = set(a)  # B: the set of nodes activated in the last completed iteration
	converged = False

	if nx.is_directed(G):
		my_degree_function = G.in_degree
	else:
		my_degree_function = G.degree

	while (not converged) and (max_hop > 0):
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:
				prob = random_generator.random()  # in the range [0.0, 1.0)
				p = 1.0 / my_degree_function(m)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B
		max_hop -= 1

	return len(A)

""" Evaluates a given seed set A, simulated "no_simulations" times.
	Returns a tuple: (the mean, the stdev).
"""
def MonteCarlo_simulation(G, A, p, no_simulations, model, random_generator=None):

	if random_generator is None:
		random_generator = random.Random()
		random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable; TODO evaluate computational cost

	results = []

	if model == 'WC':
		for i in range(no_simulations):
			results.append(WC_model(G, A, random_generator=random_generator))
	elif model == 'IC':
		for i in range(no_simulations):
			results.append(IC_model(G, A, p, random_generator=random_generator))

	return (numpy.mean(results), numpy.std(results))

def MonteCarlo_simulation_max_hop(G, A, p, no_simulations, model, max_hop=2, random_generator=None):
	"""
	calculates approximated influence spread of a given seed set A, with
	information propagation limited to a maximum number of hops
	example: with max_hops = 2 only neighbours and neighbours of neighbours can be activated
	:param G: networkx input graph
	:param A: seed set
	:param p: probability of influence spread (IC model)
	:param no_simulations: number of spread function simulations
	:param model: propagation model
	:param max_hops: maximum number of hops
	:return:
	"""
	if random_generator is None:
		random_generator = random.Random()
		random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable; TODO evaluate computational cost

	results = []

	if model == 'WC':
		for i in range(no_simulations):
			results.append(WC_model_max_hop(G, A, max_hop, random_generator))
	elif model == 'IC':
		for i in range(no_simulations):
			results.append(IC_model_max_hop(G, A, p, max_hop, random_generator))

	return (numpy.mean(results), numpy.std(results))

if __name__ == "__main__":

	G = nx.path_graph(100)
	print(nx.classes.function.info(G))
	print(MonteCarlo_simulation(G, [0, 2, 4, 6, 8, 10], 0.1, 100, 'IC'))
