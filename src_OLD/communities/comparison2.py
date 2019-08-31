# Fast script to just compare influence maximization results inside an archive
# by Alberto Tonda, 2018 <alberto.tonda@gmail.com>

import igraph
import math
import sys

# add folder to Python PATH, we'll import a few functions from there
sys.path.append("../heuristics")
sys.path.append("../multiObjective-inspyred")
sys.path.append("../approximations")
sys.path.append("../evolutionary")

# local scripts
import SNCommunities_igraph as SNCommunities 
import SNInflMaxHeuristics_networkx as SNHeuristics
import SNSigmaSim_networkx as SNSim
import evolutionaryInfluenceMaximization as evolutionary

def main() :
	
	k = 200 # our total budget for the influencers (seed nodes)
	graphFile = "../SN/facebook_combined.txt"
	model = "IC"
	p = 0.01
	
	archiveFile = "archive-gen-1000-Facebook-IC001-k200-seed-GDD.csv"
	
	G = SNCommunities.read_undirected_graph(graphFile)
	print("Graph loaded, %d nodes and %d arcs." % (G.vcount(), G.ecount()))
	
	# read the archive
	print("Loading population file...")
	lines = []
	with open(archiveFile, "r") as fp : lines = fp.readlines()
	
	seedSet = []
	lines.pop(0)
	for line in lines :
		tokens = line.rstrip().split(',')
		if float(tokens[1]) == k :
			seedSet = [ int(float(tokens[i])) for i in range(2, len(tokens)) ]
	print("Selected individual with size %d" % len(seedSet))
	
	print("Starting simulation...")
	influence = SNSim.evaluate(G, seedSet, p, 100, model)
	percentageOfGraphReached = 100.0 * influence[0] / G.vcount()
	print("Total simulated influence: %s, %.2f%%" % (str(influence), percentageOfGraphReached))
	
	if True :
		print("Now simulating corresponding heuristic SD individual...")
		heuristicFile = "../heuristic-results/Facebook-IC-p001-SDISC-high_front.txt"
		lines = []
		with open(heuristicFile, "r") as fp : lines = fp.readlines()
		
		seedSet = []
		for line in lines :
			tokens = line.rstrip().split()
			if float(tokens[0]) == k :
				seedSet.append( int(float(tokens[3][1:-1])) )
				for i in range(4, len(tokens)-1) : seedSet.append( int(float(tokens[i][:-1])) )
				seedSet.append( int(float(tokens[-1][:-1])) )
		print("Selected individual with size %d" % len(seedSet))

		print("Starting simulation...")
		influence = SNSim.evaluate(G, seedSet, p, 100, model)
		percentageOfGraphReached = 100.0 * influence[0] / G.vcount()
		print("Total simulated influence: %s, %.2f%%" % (str(influence), percentageOfGraphReached))

	return

if __name__ == "__main__" :
	sys.exit( main() )
