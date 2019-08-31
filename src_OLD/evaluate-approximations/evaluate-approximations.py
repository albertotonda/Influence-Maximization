# This script is used to evaluate the approximations on the concrete case studies
# by Alberto Tonda, 2017 <alberto.tonda@gmail.com>

# first of all, let's make all directories with scripts visible from this one
import sys
sys.path.append("../approximations")
# TODO move influence spread models to another directory and add them from there

# then, import the scripts related to approximations and influence spread models
# NOTE igraph-based stuff was creating issues
#import SNSigmaApprox_igraph as approximations
#import SNSigmaSim_igraph as models
import SNSigmaApprox_networkx as approximations
import SNSigmaSim_networkx as models

# now, import some other useful module
import argparse
import datetime
import os
import sys
import time
import random

def main() :
	
	# a few default values (ideally they can be overrideen by command line)
	minNodes = 1
	maxNodes = 200
	numberOfSeedSets = 10
	numberOfRepetitions = 100
	startingNumberOfSeedSets = 0
	fileName = "results-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M.csv")
	
	# parse command-line arguments
	parser = argparse.ArgumentParser(description="Python script that empirically evaluates goodness of approximations vs influence spread models for a given graphs.\nBy Doina Bucur, Giovanni Iacca, Andrea Marcelli, Giovanni Squillero, Alberto Tonda, 2016 <alberto.tonda@gmail.com>")	
	parser.add_argument("-g", "--graph", help="Target graph file", required=True)
	parser.add_argument("-n", "--number", type=int, help="Number of random seed sets that will be evaluated. Default: " + str(numberOfSeedSets))
	parser.add_argument("-r", "--repetitions", type=int, help="Number of repetitions of the IC simulation. Default: " + str(numberOfRepetitions))
	parser.add_argument("-f", "--fileName", help="Filename. If not specified, the default value will be \"" + fileName + "\".")
	args = parser.parse_args()
	
	# has any default value been overridden?
	if args.number : numberOfSeedSets = args.number
	if args.fileName : fileName = args.fileName
	
	# load the graph
	print("Loading graph \"" + args.graph + "\"...")
	G = approximations.read_undirected_graph(args.graph)
	
	# a list with all the labes of the nodes in the graph (useful for later)
	#nodes = [ v['name'] for v in G.vs ] # unfortunately, this needs to be changed if we use igraph or networkx
	nodes = list(G.nodes)
	print("nodes=", nodes)
	
	# some utility variables for time elapsed
	totalTimeEDV = 0
	totalTimePS = 0
	totalTimeIC = 0
	ratioIC_EDV = 0.0
	
	# statistics list, to be filled
	statistics = []
	
	# actually, it's better to save stuff step by step
	# if the file does not exist, write header 
	if not os.path.exists(fileName) : 
		with open(fileName, "w") as fp :
			fp.write("size,EDV,PS,IC_mean,IC_std,seeds\n")	
	
	else :
		with open(filename, "r") as fp :
			lines = fp.readlines()
			lines.pop(0) # header

		print("File \"%s\" exists, %d seed sets found. Appending results..." % (fileName,len(lines)))
	
	# now, let's start running this bitch!
	for i in range(startingNumberOfSeedSets, numberOfSeedSets) :
		
		# generate a random seed set
		seedSetSize = random.randint(minNodes, maxNodes)
		#seedSetSize = 50 # used for a couple of attempts
		seedSet = [0] * seedSetSize
		for k in range(0, seedSetSize) : seedSet[k] = nodes[ random.randint(0, len(nodes)-1) ]
		
		# let's set up a dictionary for the current seed set
		seedSetDict = dict()
		seedSetDict["seedSet"] = seedSet
		seedSetDict["size"] = seedSetSize
		
		print("Created seed set #" + str(i+1) + "/" + str(numberOfSeedSets) + ", of size " + str(seedSetSize) + "!")
		
		# now, let's start the evaluations!
		sys.stdout.write("\tEvaluating approximation EDV...")
		start = time.time()
		seedSetDict["EDV"] = approximations.approx_EDV(G, seedSet, 0.05)
		end = time.time()
		seedSetDict["EDV_time"] = end - start
		print("[%.4f s]" % seedSetDict["EDV_time"])
		
		sys.stdout.write("\tEvaluating approximation PS...")
		start = time.time()
		seedSetDict["REPRGRAPH"] = approximations.approx_repr_graph('PS', G, seedSet, 0.05, 'IC')
		end = time.time()
		seedSetDict["REPRGRAPH_time"] = end - start
		print("[%.4f s]" % seedSetDict["REPRGRAPH_time"])
		
		sys.stdout.write("\tRunning simulation with IC...")
		start = time.time()
		seedSetDict["IC"] = models.evaluate(G, seedSet, 0.05, numberOfRepetitions, 'IC')
		end = time.time()
		seedSetDict["IC_time"] = end - start
		print("[%.4f s]" % seedSetDict["IC_time"])
		
		statistics.append( seedSetDict )
		
		totalTimeEDV += seedSetDict["EDV_time"]
		totalTimePS += seedSetDict["REPRGRAPH_time"]
		totalTimeIC += seedSetDict["IC_time"]
		ratioIC_EDV += float(seedSetDict["IC_time"]/seedSetDict["EDV_time"])
		
		# write stuff to file, appending results
		with open(fileName, "a") as fp :
			fp.write( str(seedSetDict["size"]) )
			fp.write( "," + str(seedSetDict["EDV"]) )
			fp.write( "," + str(seedSetDict["REPRGRAPH"][0]) )
			fp.write( "," + str(seedSetDict["IC"][0]) )
			fp.write( "," + str(seedSetDict["IC"][1]) )
			
			for sn in seedSetDict["seedSet"] : fp.write("," + str(sn))
			
			fp.write("\n")
	
	# now, let's output some stats related to time elapsed
	print("\nThe average ratio between the time needed to evaluate IC and EDV is:", (ratioIC_EDV/len(statistics)))
	print("Another way to put IC/EDV:", float(totalTimeIC/totalTimeEDV))
	print("IC/PS:", float(totalTimeIC/totalTimePS))
	
	return

if __name__ == "__main__" :
	sys.exit(main())
