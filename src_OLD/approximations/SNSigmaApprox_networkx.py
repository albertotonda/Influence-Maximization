from __future__ import print_function
import networkx as nx
from math import pow 

import SNSigmaSim_networkx as SNSim

def read_undirected_graph(f):

	G = nx.read_edgelist(f, comments='#', delimiter=' ', create_using=nx.Graph(), nodetype=int, data=False)
	return G


""" 
Given a graph G with edge probabilities p, and a seed list A, 
computes a numerical approximation of the influence of A in G as the 
Expected Diffusion Value (EDV) from 

Jiang, Q., Song, G., Cong, G., Wang, Y., Si, W., Xie, K.: 
Simulated annealing based influence maximization in social networks. AAAI (2011)

Known: It only approximates well under Independent Cascade propagation, and small p.
"""
def approx_EDV(G, A, p):
    influence = len(A)

    neighbourhood_excl_A = set()
    for v in A:
        neighbourhood_excl_A |= set(G.neighbors(v)) # you want successors (out-neighbours); G.neighbors(v)
    neighbourhood_excl_A -= set(A)                  # does give the successors for both Graphs and DiGraphs

    for v in neighbourhood_excl_A:
        if nx.is_directed(G):
            rv = len(set(G.predecessors(v)) & set(A)) # you want predecessors (in-neighbours)
        else:
            rv = len(set(G.neighbors(v)) & set(A))
        influence += 1 - pow(1 - p, rv)

    return influence


"""
Computes a deterministic representative graph of G with the Probability Sorting (PS) method from

Parchas, P., Gullo, F., Papadias, D., Bonchi, F.: 
Uncertain graph processing through representative instances. ACM Trans. Database Syst. (2015)

Only defined on undirected graphs.
"""
def repr_graph_PS(G, p):
    def dis2(RG, G, v, p):
        return RG.degree(v) - int(round(p * G.degree(v))) # no mode=igraph.IN/OUT; undirected

    # at the end of this process, RG will have all nodes of G, but no edges
    RG = nx.Graph()
    RG.add_nodes_from(G.nodes(data=True))

    # all edges have the same probability p; no need for edge sorting

    # adding vertices to RG
    for e in G.edges:
        dis2_u = dis2(RG, G, e[0], p)
        dis2_v = dis2(RG, G, e[1], p)
        if abs(dis2_u + 1) + abs(dis2_v + 1) < abs(dis2_u) + abs(dis2_v):
            RG.add_edge(e[0], e[1])
    return RG


"""
Given a graph G with edge probabilities p, and a seed list A, 
computes a numerical approximation of the influence of A in G by simulating the propagation model
once over a deterministic representative graph RG of G, computed with various heuristics (PS, ADR, ABM).
"""
def approx_repr_graph(method, G, A, p, propagation_model):
    if method == 'PS':
        RG = repr_graph_PS(G, p)

    #print(RG.nodes())
    #print(RG.edges())
    res = SNSim.evaluate(RG, A, 1.00, 1, propagation_model)
    return res[0]

if __name__ == '__main__':
    # READ GRAPH
    # G = read_undirected_graph("../SN/facebook_combined.txt")
    # G.write_pickle(fname="graphs/facebook_combined_undirected.pickle")
    # G = igraph.Graph().Read_Pickle(fname="graphs/facebook_combined_undirected.pickle") # not faster than reading from .txt
    # G = nx.grid_2d_graph(2, 2, create_using=nx.DiGraph())
    # G = nx.ladder_graph(3, create_using=nx.Graph())
    G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 4)])
    print(G.nodes())
    print(G.edges())

    # A = [0, 1888, 483, 1985, 1800, 107, 3437, 2543, 1684, 1912] # found by MOEA (Evo*'17), influence 322 (p=0.01)
    # A = [1912, 107, 1367, 1810, 1467, 2630, 1791, 2244, 2108, 997] # found with CELF, influence 284 (p=0.01)
    # A = [107] # HIGHDEG, influence 71 (p=0.01)
    # A = [1912] # CELF, influence 146 (p=0.01)
    # A = [0, 10, 20, 30, 40, 100, 1000, 2000, 3000, 4000] # influence 14 (p=0.01)
    A = [0]

    p = 0.75

    #print("Simulation:", SNSim.evaluate(G, LA, p, 200, 'IC'))
    print("EDV approximation:", approx_EDV(G, A, p))
    print("PS approximation:", approx_repr_graph('PS', G, A, p, 'IC'))
