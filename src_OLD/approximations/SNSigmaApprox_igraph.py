import igraph
from math import pow 

import SNSigmaSim_igraph as SNSim

def read_undirected_graph(f):
    with open(f, 'r') as fin:
        G = igraph.Graph()
        V = set()
        E = []

        for line in fin:
            v1, v2 = line.strip().split() # vertex names are strings in igraph
            V.add(v1)
            V.add(v2)
            E.append((v1, v2))

        # print(sorted(list(V)))
        G.add_vertices(list(V)) # e.g., vertex 1912 will have "name" '1912' (a string!) 
        G.add_edges(E)

        # to check that all edges from the file made it into G
        # for e in G.es:
        #     print(G.vs[e.source]['name'], G.vs[e.target]['name'])
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
        neighbourhood_excl_A |= set(G.neighbors(v, mode=igraph.OUT)) # out-neighbours!
    neighbourhood_excl_A -= set(A)

    for v in neighbourhood_excl_A:
        rv = len(set(G.neighbors(v, mode=igraph.IN)) & set(A))
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

    RG = G.copy()
    RG.delete_edges(None) # RG kept all vertices, but no edge

    # all edges have the same probability p; no need for edge sorting

    # adding vertices to RG
    for e in G.es:
        dis2_u = dis2(RG, G, e.source, p)
        dis2_v = dis2(RG, G, e.target, p)
        if abs(dis2_u + 1) + abs(dis2_v + 1) < abs(dis2_u) + abs(dis2_v):
            RG.add_edge(e.source, e.target)
    return RG


"""
Given a graph G with edge probabilities p, and a seed list A, 
computes a numerical approximation of the influence of A in G by simulating the propagation model
once over a deterministic representative graph RG of G, computed with various heuristics (PS, ADR, ABM).
"""
def approx_repr_graph(method, G, A, p, propagation_model):
    if method == 'PS':
        RG = repr_graph_PS(G, p)

    return SNSim.evaluate(RG, A, 1.00, 1, propagation_model)        


if __name__ == '__main__':
    # READ GRAPH
    G = read_undirected_graph("graphs/facebook_combined.txt")
    # G.write_pickle(fname="graphs/facebook_combined_undirected.pickle")
    # G = igraph.Graph().Read_Pickle(fname="graphs/facebook_combined_undirected.pickle") # not faster than reading from .txt
    print(G.summary())

    A = [0, 1888, 483, 1985, 1800, 107, 3437, 2543, 1684, 1912] # found by MOEA (Evo*'17), influence 322 (p=0.01)
    # A = [1912, 107, 1367, 1810, 1467, 2630, 1791, 2244, 2108, 997] # found with CELF, influence 284 (p=0.01)
    # A = [107] # HIGHDEG, influence 71 (p=0.01)
    # A = [1912] # CELF, influence 146 (p=0.01)
    # A = [0, 10, 20, 30, 40, 100, 1000, 2000, 3000, 4000] # influence 14 (p=0.01)
    LA = list(map(str, A)) # the node ids are strings in igraph

    p = 0.1

    print("Simulation:", SNSim.evaluate(G, LA, p, 200, 'IC'))
    #print("EDV approximation:", approx_EDV(G, LA, p))
    #print(repr_graph_PS(G, p).summary())
    #print("PS approximation:", approx_repr_graph('PS', G, LA, p, 'IC'))