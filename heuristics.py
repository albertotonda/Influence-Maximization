import networkx as nx
import heapq as hq
import spread

""" Heuristics for the Influence Maximization problem"""

""" NB:	All prints() should change to logging.
"""

""" (Kempe) "The high-degree heuristic chooses nodes v in order of decreasing degrees. 
	Suits (un)directed graphs. 
	Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
	Returns a list of the form [(degree_i, node_i)]
"""
def high_degree_nodes(k, G):

	if nx.is_directed(G):
		my_degree_function = G.out_degree
	else:
		my_degree_function = G.degree

	# the list of nodes to be returned; initialization
	H = [(my_degree_function(i), i) for i in list(G)[0:k]]
	hq.heapify(H) # min-heap

	for i in list(G)[k:]: # iterate through the remaining nodes
		deg_i = my_degree_function(i)
		if deg_i > H[0][0]:
			hq.heappushpop(H, (deg_i, i))

	return list(map(lambda x: x[1], H))

""" (Kempe) Greedily adds nodes in order of increasing average distance to other nodes 
	in the network. Distance = |V| for disconnected node pairs. 
	(DB) This is the inverse of closeness centrality, so can be rewritten more simply.
	Suits (un)directed graphs. 
	Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
	Returns a list of the form [(avg_distance_i, node_i)]
"""
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

	return list(map(lambda x: x[1], H))

""" The SingleDiscount algorithm by Chen et al. (KDD'09) for any cascade model.
	Calculates the k nodes of highest degree, making discounts if direct neighbours are already chosen.
	Suits (un)directed graphs. 
	Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
	Returns a list of the form [node_i]
"""
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

""" Generalized Degree Discount from Wang et al., PLOS One'16.
	Designed for Independent Cascade (hence p is passed as an argument) and undirected graphs.
	This code works also for directed graphs; assumes the edges point OUT of the influencer,
	Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
	Calculates the k nodes of highest degree, making discounts if neighbours up to some depth are already chosen.
	Returns a list of the form [node_i]
"""
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

""" Proven to approximate within 63% of the optimal by Kempe, et al. for any cascade model.
	Hugely expensive in time: it calls the Monte-Carlo simulation from the spread file.
	Returns a list of the form [node_i]

	Also prints (not only returns) the 1..k nodes of supposedly max influence, and that influence.
	(It's too time-expensive to wait until all iterations are completed.)
	Could change to a yield() instead of print().
"""
def general_greedy(k, G, p, no_simulations, model):
	S = []

	for i in range(k):
		maxinfl_i = (-1, -1)
		v_i = -1
		for v in list(set(G.nodes()) - set(S)):
			eval_tuple = spread.MonteCarlo_simulation(G, S+[v], p, no_simulations, model)
			if eval_tuple[0] > maxinfl_i[0]:
				maxinfl_i = (eval_tuple[0], eval_tuple[1])
				v_i = v

		S.append(v_i)
		print(i+1, maxinfl_i[0], maxinfl_i[1], S)

	return S

""" CELF (Leskovec, Cost-effective Outbreak Detection in Networks, KDD07) proven to approximate within 63% of the optimal.
	Expensive in time: it calls the Monte-Carlo simulation from the spread file, but fewer times than general_greedy.
	Returns a list of the form [node_i].

	Also prints (not only returns) the 1..k nodes of supposedly max influence, and that influence.
"""
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
				res = spread.MonteCarlo_simulation(G, A, p, no_simulations, model)
				print(len(A), res[0], res[1], A, sep=' ') # mean, stdev, A
				break
			else:
				eval_after  = spread.MonteCarlo_simulation(G, A+[s], p, no_simulations, model)
				eval_before = spread.MonteCarlo_simulation(G, A, p, no_simulations, model)
				delta[s] = eval_after[0] - eval_before[0]
				curr[s] = True

	return A

if __name__ == "__main__":

	G = nx.cycle_graph(100, create_using=nx.Graph())
	print(nx.classes.function.info(G))

	k = 10
	p = 0.1
	num_sims = 100
	model = 'IC'

	# A = high_degree_nodes(k, G)
	A = low_distance_nodes(k, G)

	# A = single_discount_high_degree_nodes(k, G)
	# A = generalized_degree_discount(k, G, p)
	# A = general_greedy(k, G, p, num_sims, model) # this prints rather than returns
	# A = CELF(k, G, p, num_sims, model)

	print(A, spread.MonteCarlo_simulation(G, A, p, num_sims, model))
