# -*- coding: utf-8 -*-

import networkx as nx
import heapq as hq
import SNSigmaSim_networkx as SNSim
import time

# (Kempe) "The high-degree heuristic chooses nodes v in order of decreasing degrees. 
# Considering high-degree nodes as influential has long been a standard approach 
# for social and other networks [3, 83], and is known in the sociology literature 
# as 'degree centrality'."
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of highest degree
def high_degree_nodes(k, G):

    if nx.is_directed(G):
        my_degree_function = G.out_degree
    else:
        my_degree_function = G.degree

    # the list of nodes to be returned; initialization
    H = [(my_degree_function(i), i) for i in G.nodes()[0:k]] 
    hq.heapify(H) # min-heap

    for i in G.nodes()[k:]: # iterate through the remaining nodes
        if my_degree_function(i) > H[0][0]:
            hq.heappushpop(H, (my_degree_function(i), i))
 
    return H

# (Kempe) Greedily adds to the set $S$ select nodes in order of increasing average distance 
# to other nodes in the network; following the intuition that being able to reach other nodes 
# quickly translates into high influence. distance = |V| for disconnected node pairs.
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of lowest average distance to the other nodes
def low_distance_nodes(k, G):

    path_lens = nx.all_pairs_shortest_path_length(G) # contains only existing path lengths

    max_path_len = G.size()
    L = []

    for n in G.nodes():
        # compute the average distance per node
        avg_path_len_n = max_path_len

        if n in path_lens:
            sum_path_len_n = 0
            for m in set(G.nodes()) - set([n]):
                if m in path_lens[n]:
                    sum_path_len_n += path_lens[n][m]
                else:
                    sum_path_len_n += max_path_len
            avg_path_len_n = sum_path_len_n / (G.size() - 1)

        # add the average distance of n to L
        L.append((-avg_path_len_n, n)) # negated distance, to match the min-heap

    # L.sort(reverse=True) # expensive, so heap below
    H = L[0:k] 
    hq.heapify(H) # min-heap

    for i in L[k:]: # iterate through the remaining nodes
        if i[0] > H[0][0]:
            hq.heappushpop(H, i)

    return list(map(lambda x: (-x[0], x[1]), H))

# The SingleDiscount algorithm by Chen et al. (KDD'09) for any cascade model.
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of highest degree, making discounts if direct neighbours are already chosen.
def single_discount_high_degree_nodes(k, G):
    if nx.is_directed(G):
        my_predecessor_function = G.predecessors
        my_degree_function = G.out_degree
    else:
        my_predecessor_function = G.neighbors
        my_degree_function = G.degree

    S = []
    ND = {}
    for n in G.nodes():
        ND[n] = my_degree_function(n)

    for i in range(k):
        # find the node of max degree not already in S
        u = max(set(list(ND.keys())) - set(S), key=(lambda key: ND[key]))
        S.append(u)

        # discount out-edges to u from all other nodes
        for v in my_predecessor_function(u):
            ND[v] -= 1

    return S

# Generalized Degree Discount from Wang et al., PlosOne'16.
# Only designed for Independent Cascade (hence p is passed as an argument) and undirected graphs.
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of highest degree, making discounts if neighbours up to some depth are already chosen.
def generalized_degree_discount(k, G, p):
    if nx.is_directed(G):
        my_predecessor_function = G.predecessors
        my_degree_function = G.out_degree
    else:
        my_predecessor_function = G.neighbors
        my_degree_function = G.degree

    S = []
    GDD = {}
    t = {}

    for n in G.nodes():
        GDD[n] = my_degree_function(n)
        t[n] = 0

    for i in range(k):
        # select the node with current max GDD from V-S
        u = max(set(list(GDD.keys())) - set(S), key=(lambda key: GDD[key]))
        S.append(u)
        NB = set()

        # find the nearest and next nearest neighbors of u and update tv for v in Γ(u)
        for v in my_predecessor_function(u):
            NB.add(v)
            t[v] += 1
            for w in my_predecessor_function(v):
                if w not in S:
                    NB.add(w)
        # update gddv for all v in NB
        for v in NB:
            sumtw = 0
            for w in my_predecessor_function(v):
                if w not in S:
                    sumtw = sumtw + t[w]
            dv = my_degree_function(v)
            GDD[v] = dv - 2*t[v] - (dv - t[v])*t[v]*p + 0.5*t[v]*(t[v] - 1)*p - sumtw*p
            if GDD[v] < 0:
                GDD[v] = 0

    return S

# The algorithm proven to approximate within 63% of the optimal by Kempe, et al. for any cascade model.
# Hugely expensive in time.
# -> Prints (rather than returns) the 1..k nodes of supposedly max influence, and that influence.
# (It gets too time-expensive otherwise.)
def general_greedy(k, G, p, no_simulations, model):
    S = []

    for i in range(k):
        maxinfl_i = (-1, -1)
        v_i = -1
        for v in list(set(G.nodes()) - set(S)):
            eval_tuple = SNSim.evaluate(G, S+[v], p, no_simulations, model)
            if eval_tuple[0] > maxinfl_i[0]:
                maxinfl_i = (eval_tuple[0], eval_tuple[2])
                v_i = v

        S.append(v_i)
        print(i+1, maxinfl_i[0], maxinfl_i[1], S)

# CELF (Leskovec, Cost-effective Outbreak Detection in Networks, KDD07) is proven to approximate within 63% of the optimal.
# -> Prints (rather than returns) the 1..k nodes of supposedly max influence, and that influence.
# (It gets too time-expensive otherwise.)
# -> Does return only the final set of exactly k nodes.
def CELF(k, G, p, no_simulations, model):
    A = []

    max_delta = len(G.nodes()) + 1
    delta = {}
    for v in G.nodes():
        delta[v] = max_delta
    curr = {}

    while len(A) < k:
        for j in set(G.nodes()) - set(A):
            curr[j] = False
        while True:
            # find the node s from V-A which maximizes delta[s]
            max_curr = -1
            s = -1
            for j in set(G.nodes()) - set(A):
                if delta[j] > max_curr:
                    max_curr = delta[j]
                    s = j
            # evaluate s only if curr = False
            if curr[s]:
                A.append(s)
                # the result for this seed set is:
                res = SNSim.evaluate(G, A, p, no_simulations, model)
                print(len(A), res[0], res[2], A, sep=' ') # mean, CI95, A
                break
            else:
                eval_after = SNSim.evaluate(G, A+[s], p, no_simulations, model)
                eval_before = SNSim.evaluate(G, A, p, no_simulations, model)
                delta[s] = eval_after[0] - eval_before[0]
                curr[s] = True

    return A

# This is incomplete; ran into some problems with understanding the CELF++ paper.
def CELFpp(k, G, p, no_simulations, model):
    S = set()
    Q = [] # heap; stores tuples ⟨u.mg1, u.prev best, u.mg2, u.f lag⟩, u = a node
    hq.heapify(Q) # min-heap

    last_seed = None
    cur_best = None # a tuple (u, mg)

    for u in G.nodes():
        # prep u's tuple and add to Q
        u_mg1 = SNSim.evaluate(G, [u], p, no_simulations, model)[0]
        u_prev_best = cur_best
        temp_list = []
        if cur_best:
            temp_list.append(cur_best)
        u_mg2 = SNSim.evaluate(G, [u]+temp_list, p, no_simulations, model)[0]
        u_flag = 0
        hq.heappush(Q, (u, u_mg1, u_prev_best, u_mg2, u_flag)) # the tuple
        if not cur_best or cur_best[1] < u_mg1:
            cur_best = (u, u_mg1)
    while len(S) < k:
        u_tuple = hq.heappop(Q)
        if u_tuple[4] == len(S): # u[4] = u.flag
            S.add(u_tuple[0])
            last_seed = u
        elif u_tuple[2] == last_seed: # u[2] = u.prev_best
            u_tuple[1] = u_tuple[3] # u.mg1 = u.mg2
        else:
            base = SNSim.evaluate(G, list(S), p, no_simulations, model)[0]
            u_tuple[1] = SNSim.evaluate(G, [u]+list(S), p, no_simulations, model)[0] - base
            u_tuple[2] = cur_best
            u_tuple[3] = SNSim.evaluate(G, [u,cur_best[0]]+list(S), p, no_simulations, model)[0] - base
        u_tuple[4] = len(S)
        # update cur best
        pass
        # reinsert u into Q and heapify
        hq.heappush(Q, u_tuple)

    return S

def dump_degree_list(G):
    H = []

    for i in G.nodes():
        H.append((i, G.out_degree(i)))

    return H

"""
# A trial at parallelizing
from multiprocessing import Pool

gl_G = 0
gl_p = 0
gl_no_simulations = 0
gl_model = 0
gl_S = []
gl_k = 0

gl_t0 = -1

def evaluate_mt(v):
    eval_tuple = SNSim.evaluate(gl_G, gl_S+[v], gl_p, gl_no_simulations, gl_model)
    return (eval_tuple[0], v)

def general_greedy_mt(k, G, p, no_simulations, model, no_cores):
    global gl_G, gl_k, gl_p, gl_no_simulations, gl_model, gl_S
    gl_G = G
    gl_k = k
    gl_p = p
    gl_no_simulations = no_simulations
    gl_model = model

    S = [] # the list of nodes to be returned
    gl_S = S

    maxinfl_i = -1
    v_i = -1

    for i in range(k):
        L = list(set(G.nodes()) - set(S))
        pool = Pool(no_cores)
        res = pool.map(evaluate_mt, L)
        # print res

    (maxinfl_i, v_i) = max(res)
    S.append(v_i)
    print("[k=" + str(i) + "] Processing time (s): " + str(time.time() - t0))
    return S, maxinfl_i
"""

if __name__ == "__main__":
    file = '../SN/soc-Epinions1.txt' # A trusts B
    # file = 'wiki-Vote.txt' # A votes B
    # file = 'amazon0302.txt'
    # file = 'web-Google.txt'
    # file = 'CA-GrQc.txt'
    tempG = nx.read_edgelist(file, comments='#', delimiter='\t', create_using=nx.DiGraph(), nodetype=int, data=False)
    G = tempG.reverse() # to get the edges to flow OUT of the influencers

    #file = 'graphs/twitter_combined.txt' # A follows B
    #tempG = nx.read_edgelist(file, comments='#', delimiter=' ', create_using=nx.DiGraph(), nodetype=int, data=False)
    #G = tempG.reverse()

    # file = 'facebook_combined.txt'
    # G = nx.read_edgelist(file, comments='#', delimiter=' ', create_using=nx.Graph(), nodetype=int, data=False)

    # G = nx.grid_2d_graph(10, 5)

    print("Read graph:", len(list(G.nodes())), "nodes", len(list(G.edges())), "edges")

    # compute the seed sets for the upper bound for k
    k = 200
    p = 0.01
    num_sims = 100
    model = 'IC'

    # DEGREE
    # A = high_degree_nodes(k, G)
    # A.sort(reverse=True)
    # A = list(map(lambda x: x[1], A))

    # Single DD
    # A = single_discount_high_degree_nodes(k, G)

    # DISTANCE
    #A = low_distance_nodes(k, G)
    #A.sort()
    #A = list(map(lambda x: x[1], A))

    # Generalized DD
    A = generalized_degree_discount(k, G, p)

    # just to debug the above
    print(A, len(A))

    # evaluate the seed sets obtained by the heuristics above
    for i in range(1, k+1):
       res = SNSim.evaluate(G, A[:i], p, num_sims, model)
       print(i, res[0], res[2], A[:i], sep=' ') # mean, CI95, A

    # CELF
    # A = CELF(k, G, p, num_sims, model)

    # GEN-GREEDY
    # general_greedy(k, G, p, num_sims, model) # this prints rather than returns
