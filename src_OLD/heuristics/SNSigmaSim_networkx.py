# -*- coding: utf-8 -*-

import networkx as nx
import random
import numpy
import math
import time

# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", A is followed by B", "A is trusted by B". 

def IC_model(G, a, p):              # a: the set of initial active nodes
                                    # p: the system-wide probability of influence on an edge, in [0,1]
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False

    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
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
 
    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree

    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random.random()	# in the range [0.0, 1.0)
                p = 1.0/my_degree_function(m)
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B

    return len(A)

# evaluates a given seed set A
# simulated "no_simulations" times
# returns a tuple: the mean, stdev, and 95% confidence interval
def evaluate(G, A, p, no_simulations, model):
    results = []

    if model == 'WC':
        for i in range(no_simulations):
            results.append(WC_model(G, A))
    elif model == 'IC':
        for i in range(no_simulations):
            results.append(IC_model(G, A, p))

    return numpy.mean(results), numpy.std(results), 1.96 * numpy.std(results) / math.sqrt(no_simulations)

# evaluates "no_samples" random seed sets A of size "k"
# each simulated "no_simulations" times
# returns a list of "no_samples" tuples (mean, stdev, and 95% confidence interval)
def RND_evaluate(G, k, p, no_samples, no_simulations, model):
    results = []

    for i in range(no_samples):
        A = random.sample(G.nodes(), k)
        results.append(evaluate(G, A, p, no_simulations, model))

    return results

# local testing
if __name__ == "__main__":                  # in test rather than import mode
    
    p = 0.01                                # "p" only matters for IC_model (not for WC_model)
    k = 10
    model = 'WC'
    no_simulations = 100
    no_samples = 1

    # G = nx.circular_ladder_graph(100).to_directed()
    # G = nx.path_graph(100)
    # G = nx.random_regular_graph(5, 100).to_directed()
    
    # file = 'graphs/soc-Epinions1.txt'
    # G = nx.read_edgelist(file, comments='#', delimiter='\t', create_using=nx.DiGraph(), nodetype=int, data=False)
    file = 'graphs/facebook_combined.txt'
    G = nx.read_edgelist(file, comments='#', delimiter=' ', create_using=nx.Graph(), nodetype=int, data=False)

    # A = [0, 1888, 483, 1985, 1800, 107, 3437, 2543, 1684, 1912] # found by MOEA (Evo*'17), influence 322
    # A = [1912, 107, 1367, 1810, 1467, 2630, 1791, 2244, 2108, 997] # found with CELF, influence 284
    # A = [107] # HIGHDEG, influence 71
    # A = [1912] # CELF, influence 146
    # A = [0, 10, 20, 30, 40, 100, 1000, 2000, 3000, 4000] # influence 14
    A = [107, 1684, 1912, 3437, 0, 2543, 2347, 1888, 1800, 483, 348, 1663, 2266, 1352, 1985, 1730, 1941, 2233, 1431, 2142, 2047, 1199, 686, 1584, 2206, 1768, 2111, 2218, 2384, 1086, 2611, 414, 2410, 1589, 2199, 2229, 1827, 1746, 2839, 1983, 2133, 2078, 1126, 2081, 3363, 1993, 2054, 917, 3101, 1804, 896, 376, 2123, 2289, 1612, 3291, 2464, 2754, 1577, 1277, 1390, 1377, 2560, 2742, 3830, 2598, 475, 1783, 3082, 1559, 2507, 2328, 3426, 3397, 1707, 2240, 1104, 428, 3090, 2283, 3320, 2607, 2951, 1833, 484, 1459, 2268, 2966, 2333, 1917, 3434, 1835, 136, 2944, 637, 1472, 3596, 2220, 1610, 2730, 2282, 3280, 1235, 412, 2986, 1078, 828, 3938, 2244, 2117, 1591, 517, 3232, 56, 1714, 3116, 2786, 3154, 2309, 3545, 2313, 1204, 353, 1070, 67, 2509, 1621, 2915, 2339, 1583, 2719, 713, 3387, 271, 563, 2038, 1014, 1613, 553, 2172, 2863, 3604, 1972, 705, 3980, 2877, 1345, 2087, 1124, 2778, 2724, 366, 698, 322, 1391, 1085, 2073, 1964, 2679, 3521, 1703, 2793, 1951, 538, 1786, 3019, 25, 2294, 1322, 2669, 3136, 373, 1238, 2602, 805, 2171, 1630, 2890, 3302, 925, 1844, 1505, 1729, 26, 2032, 3263, 2659, 3342, 2782, 497, 1943, 1052, 3684, 719, 3035, 3584, 3838, 465, 1622, 1995, 2187, 2716, 119, 2925, 1192, 1709, 3793, 3256, 2500, 897, 3417, 1947, 3152, 422, 3506, 363, 277, 2365, 1864, 3633, 1462, 824, 1367, 3002, 3442, 3201, 252, 1338, 942, 1980, 3078, 3396, 606, 2131, 2344, 1794, 1479, 747, 2056, 3233, 1480, 3119, 3593, 2364, 1831, 2126, 3756, 592, 1604, 3327, 3140, 1820, 1837, 1687, 2132, 1469, 3611, 21, 2676, 694, 1573, 2176, 1845, 3198, 513, 916, 2542, 2001, 1358, 3906, 3026, 3350, 995, 1920, 3680, 1366, 3299, 3162, 3948, 1628, 579, 3117, 3945, 1097, 1702, 3385, 3455, 781, 2129, 1926, 122, 2706, 395, 1574, 2116, 2361, 2088, 389, 1360, 2738, 3076, 1688, 531, 3495, 1919, 2049, 1369, 1359, 1066, 3366, 3214, 856, 1548, 2434, 3330, 745, 1227, 3360, 203, 930, 3829, 1132, 3629, 370, 3054, 3087, 1536, 630, 2325, 2620, 3930, 1911, 1273, 53, 1733, 2729, 2610, 1644, 500, 1960, 2590, 1555, 3172, 2763, 239, 504, 3918, 3758, 1427, 3731, 3348, 1540, 312, 2338, 2412, 1159, 3705, 1910, 2037, 583, 3365, 9, 932, 654, 2772, 404, 3968, 1118, 2397, 1304, 346, 3077, 823, 2018, 908, 1261, 1513, 3178, 3790, 2236, 3324, 1626, 460, 3672, 1570, 1666, 3049, 1988, 1770, 2945, 3628, 3599, 3011, 1619, 3529, 3248, 1549, 1724, 2211, 1596]
 
    p = 0.01
    print(evaluate(G, A, p, 30, 'IC'))

    # tstart = time.process_time()
    # print(RND_evaluate(G, k, p, no_samples, no_simulations, model))
    # print("Elapsed time: ", time.process_time() - tstart)
