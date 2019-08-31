# Simple Python script to compare approximations (such as EDV) with full simulations.
# by Alberto Tonda, 2018 <alberto.tonda@gmail.com>

import argparse
import matplotlib
import numpy as np
import os
import re as regex
import sys

matplotlib.use("Agg") # I don't really know what it does...

import matplotlib.animation as manimation        
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

# import a couple of utility things in a relative path
sys.path.append("../approximations")
import SNSigmaSim_networkx as simulations
import SNSigmaApprox_networkx as approximations

def loadPointsFromFile(fileName) :
	
	lines = []
	with open(fileName, "r") as fp : lines = fp.readlines()
	lines.pop(0) # remove header
	
	points = []
	for line in lines :
		tokens = line.rstrip().split(',')
		points.append( [ int(float(x)) for x in tokens[2:] ] ) 
	
	return points

def computeFitnessFront(G, simulationPoints, probability, numberOfRepetitions, simulationModel) :
	
	fitness = []
	for i in range(0, len(simulationPoints)) : # TODO modify here for shorter trial runs
		print("Running simulation for point %d/%d..." % (i+1, len(simulationPoints)))
		fitness.append( (len(simulationPoints[i]), simulations.evaluate(G, simulationPoints[i], probability, numberOfRepetitions, simulationModel)[0]) )

	return fitness

def main() :
	
	# a few hard-coded default values
	numberOfRepetitions = 100
	simulationModel = 'IC'
	probability = 0.01
	fileSimulations = "simulations.csv"
	
	# parse arguments: there might be several heuristics
	parser = argparse.ArgumentParser(description="Python script to compare influence maximization with approximations or simulations.\nBy Doina Bucur, Giovanni Iacca, Andrea Marcelli, Giovanni Squillero, Alberto Tonda, 2018 <alberto.tonda@gmail.com>")	
	
	parser.add_argument("-a", "--approximations", nargs='+', help="List of CSV files with points in a Pareto front, obtained by different approximations. Supports a list of tuples <approximation_name> <file_name> <approximation_name> <file_name> ...", required=True)
	parser.add_argument("-s", "--simulation", help="CSV files with points in a Pareto front, obtained by simulation.", required=True)
	parser.add_argument("-g", "--graph", help="Name of the graph. Used to re-run the points find by the approximations.")
	parser.add_argument("-hu", "--heuristics", nargs='+', help="List of CSV files with points in a Pareto front, obtained by different heuristics. Supports a list of tuples <heuristic_name> <file_name>...")
	
	# TODO simulation type (IC/WC) and probability
	# TODO graph directed/undirected? 
	# TODO number of simulations?
	
	args = parser.parse_args()
	
	# extra check: the list of arguments for --approximations has to be even
	if len(args.approximations) % 2 != 0 :
		sys.stderr.write("Error: the list of approximations should follow the scheme <approximation_name> <file_name> <approximation_name> <file_name> ...")
		sys.exit(0)
	
	# load the graph
	print("Loading graph \"%s\"..." % args.graph)
	G = approximations.read_undirected_graph(args.graph)
	
	# this dictionary will store all the Pareto fronts to be later plotted
	paretoFronts = dict()
	
	# read the simulation Pareto front
	print("Loading and running simulation Pareto front \"%s\"..." % args.simulation)
	
	if not os.path.exists(fileSimulations) :
		simulationPoints = loadPointsFromFile(args.simulation)
		paretoFronts["Simulations " + simulationModel] = computeFitnessFront(G, simulationPoints, probability, numberOfRepetitions, simulationModel)
		
		with open(fileSimulations, "w") as fp :
			fp.write("size,influence\n")
			for point in paretoFronts["Simulations " + simulationModel] :
				fp.write( str(point[0]) + "," + str(point[1]) + "\n")
	
	else :
		print("Found file called \"%s\": loading values from there..." % fileSimulations)
		with open(fileSimulations, "r") as fp :
			lines = fp.readlines()
			lines.pop(0) # remove header
			
			paretoFronts["Simulations " + simulationModel] = []
			for line in lines :
				tokens = line.rstrip().split(',')
				paretoFronts["Simulations " + simulationModel].append( (int(tokens[0]), float(tokens[1])) )
	
	# let's go with the approximations! 
	i = 0
	while i < len(args.approximations) :
		print("Loading and running approximation Pareto front \"%s\"..." % args.approximations[i+1])
		fileApproximation = args.approximations[i] + ".csv"
		
		if not os.path.exists(fileApproximation) :
			points = loadPointsFromFile( args.approximations[i+1] )
			paretoFronts[args.approximations[i]] = computeFitnessFront(G, points, probability, numberOfRepetitions, simulationModel)
		
			with open(fileApproximation, "w") as fp :
				fp.write("size,influence\n")
				for point in paretoFronts[args.approximations[i]] :
					fp.write( str(point[0]) + "," + str(point[1]) + "\n")
		
		else :
			print("Found approximation file \"%s\", loading values from there..." % fileApproximation)
			
			with open(fileApproximation, "r") as fp :
				lines = fp.readlines()
				lines.pop(0)
				
				paretoFronts[args.approximations[i]] = []
				for line in lines :
					tokens = line.rstrip().split(',')
					paretoFronts[args.approximations[i]].append( (int(tokens[0]), float(tokens[1])) )
		i += 2
	
	# finally, let's plot everything!
	fig = plt.figure()
	ax = plt.subplot(111)
	
	# let's create a rainbow cycle, using matplotlib's color map, that will be used for colors later
	color = cm.rainbow( np.linspace(0, 1, len(paretoFronts)) )

	keys = [ k for k in paretoFronts ]
	print("Keys:", keys)
	for i in range(0, len(keys)) :
		x = [ a[1] for a in paretoFronts[keys[i]] ]
		y = [ a[0] for a in paretoFronts[keys[i]] ]
		#print("x for " + keys[i] + ":", x)
		#print("y for " + keys[i] + ":", y)
		ax.plot(x, y, '.', c=color[i], label=keys[i])

	ax.set_xlabel("influence")
	ax.set_ylabel("nodes in the seed set")
	ax.set_title("Comparison of simulation vs heuristics")
	ax.legend(loc='best')
	
	plt.savefig("compareApproximations.pdf")
	
	return

if __name__ == "__main__" :
	sys.exit( main() )
