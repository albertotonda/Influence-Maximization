# Simple script to perform influence maximization using a hierarchical approach; it heavily exploits functions from other scripts in folders around here
# by Alberto Tonda, 2018 <alberto.tonda@gmail.com>

import igraph
import math
import sys
import time

# add folder to Python PATH, we'll import a few functions from there
sys.path.append("../heuristics")
sys.path.append("../multiObjective-inspyred")
sys.path.append("../approximations")
sys.path.append("../evolutionary")

# local scripts
import SNCommunities_igraph as SNCommunities 
import SNInflMaxHeuristics_igraph as SNHeuristics
import SNSigmaSim_igraph as SNSim
import evolutionaryInfluenceMaximization as evolutionary

def main() :
	
	k = 50 # our total budget for the influencers (seed nodes)
	graphFile = "../SN/facebook_combined.txt"
	model = "IC"
	p = 0.01
	
	print("Attempting influence maximization on graph \"%s\", with seed set size k=%d" % (graphFile, k))
	print("Influence propagation model \"%s\", p=%.2f" % (model, p))

	G = SNCommunities.read_undirected_graph(graphFile)
	print("Graph loaded, %d nodes and %d arcs." % (G.vcount(), G.ecount()))
	
	# we also start measuring time!
	startTime = time.time()
	
	# once the graph is loaded, let's locate the communities
	# that are returned as a VertexClustering object
	VC = G.community_multilevel(return_levels=False)
	SNCommunities.document_VC(VC) # some output
	
	# get the list of subgraphs
	subGraphs = sorted( [ C for C in VC.subgraphs() ], key=lambda x : len(x.vs), reverse=True)
	
	# now, we have to allocate our 'k' influencers in a way that is proportional to
	# the size of each community, with a minimum of 1 (I guess)
	proportionalImportance = [ len(C.vs)/float(len(G.vs)) for C in subGraphs ]
	kAllocation = []

	remainingBudget = k
	for i in range(0, len(proportionalImportance)) :
		kLocal = min(remainingBudget, math.floor(proportionalImportance[i] * k))
		if kLocal < 1 : kLocal = 1
		
		kAllocation.append( kLocal )
		remainingBudget -= kLocal
	
	print("Allocation of the initial budget (" + str(k) + "):", kAllocation, " -> sum: " + str(sum(kAllocation)))
	
	# adjust allocation if some budget is still available, or if it has been overexploited
	# NOTE: there is a pathological scenario that is currently NOT taken into account;
	#	what happens is k is less than the number of communities?
	if remainingBudget > 0 :
		i = 0
		while remainingBudget > 0 :
			kAllocation[i] += 1
			remainingBudget -= 1
			i += 1
			if i >= len(kAllocation) : i = 0
	
		print("Refined allocation of the initial budget (" + str(k) + "):", kAllocation, " -> sum: " + str(sum(kAllocation)))
	
	elif remainingBudget < 0 :
		i = 0
		while remainingBudget < 0 :
			if kAllocation[i] > 1 :
				kAllocation[i] -= 1
				remainingBudget += 1
			i += 1
			if i >= len(kAllocation) : i = 0
		print("Refined allocation of the initial budget (" + str(k) + "):", kAllocation, " -> sum: " + str(sum(kAllocation)))
	
	# and now, after the allocation, here we go with the influence maximization!
	globalSeedSet = []
	
	for i in range(0, len(subGraphs)) :
		
		sg = subGraphs[i]
		kLocal = kAllocation[i]
		print("Now maximizing influence in subgraph #%d (size %d), with %d seed nodes, using an EA..." % (i, sg.vcount(), kLocal))
		
		localSeedSet = None
		if kLocal > 1 :
			localSeedSet = evolutionary.EA_igraph( sg, kLocal, model, p, pop_size=20, verbose=True )
		else :
			localSeedSet = SNHeuristics.single_discount_high_degree_nodes(kLocal, sg)
		globalSeedSet.extend( localSeedSet )
	
	# now that we have the global seed set, we can actually compare the average result
	# of influence propagation! horray!
	print("Now simulating influence propagation for the overall seed set...")
	totalTime = time.time() - startTime
	influence = SNSim.evaluate(G, globalSeedSet, p, 100, model)
	percentageOfGraphReached = 100.0 * influence[0] / G.vcount()
	print("Total simulated influence: %s, %.2f%% (seed set found in %.2f s)" % (str(influence), percentageOfGraphReached, totalTime))
	
	print("Now computing quick comparison with straightforward non-hierarchical single discount...")
	startTime = time.time()
	SDseedSet = SNHeuristics.single_discount_high_degree_nodes(k, G)
	totalTime = time.time() - startTime
	SDinfluence = SNSim.evaluate(G, SDseedSet, p, 100, model)
	percentageOfGraphReached = 100.0 * SDinfluence[0] / G.vcount()
	print("Total simulated influence: %s, %.2f%% (seed set found in %.2f s)" % (str(SDinfluence), percentageOfGraphReached, totalTime))
	
	# oops, at the moment the results with the hierarchical thing is worse...
	
	return

if __name__ == "__main__" :
	sys.exit( main() )
