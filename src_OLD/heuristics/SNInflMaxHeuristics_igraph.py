# This library wants to be a series of heuristics that work like the networkx ones, except that they are using igraph graphs

import igraph
import heapq as hq
#import SNSigmaSim_igraph as SNSim
import time

# The SingleDiscount algorithm by Chen et al. (KDD'09) for any cascade model.
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of highest degree, making discounts if direct neighbours are already chosen.
def single_discount_high_degree_nodes(k, G):

	if G.is_directed() :
		# DOES NOT WORK, MADAFAKA
		print("Sorry, mate. Not implemented.")
	else :
		
		S = []
		ND = {}
		
		for n in G.vs : ND[n] = G.degree(n, mode="ALL")
		
		for i in range(k) :
			# find the node of max degree not already in S
			u = max(set(list(ND.keys())) - set(S), key=(lambda key: ND[key]))
			S.append(u)
			
			# discount out-edges to u from all other nodes in the graph
			# there might be some issues with neighborhood, so we
			# perform some checks before
			neighbors = [ x for x in G.neighbors(u) if x in G.vs ]
			for v in neighbors : ND[v] -= 1
	
	return S
			

    #if G.is_directed():
    #    my_predecessor_function = G.predecessors
    #    my_degree_function = G.out_degree
    #else:
    #    my_predecessor_function = G.neighbors
    #    my_degree_function = G.degree
    #
    #S = []
    #ND = {}
    #for n in G.vs():
    #    ND[n] = my_degree_function(n)
    #
    #for i in range(k):
    #    # find the node of max degree not already in S
    #    u = max(set(list(ND.keys())) - set(S), key=(lambda key: ND[key]))
    #    S.append(u)

     #   # discount out-edges to u from all other nodes
    #    for v in my_predecessor_function(u):
    #        ND[v] -= 1
    #
    #return S
