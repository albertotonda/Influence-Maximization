# Python script to evaluate influence of a network, using inspyred's NSGA-II
# by Alberto Tonda, 2016 <alberto.tonda@gmail.com>

# TODO:
# - convert all horrible "printf" stuff into nice logging

import argparse
import inspyred
import networkx as nx
import logging
import math
import numpy
import os
import random
import re as regex

from time import time

import sys
sys.path.append("../approximations")

import SNSigmaApprox_networkx as approximations

# now, this is here mainly because I am experiencing issues with matplotlib on an OpenSUSE server I am using
matplotlibOK = True
try :
    import matplotlib as mpl
    mpl.use('Agg') # this line makes plots work even if there is no display specified (e.g. in a "screen" session)
    import matplotlib.pyplot as plt
except ImportError :
    sys.stderr.write("Cannot import matplotlib. All plots will be disabled.")
    matplotlibOK = False

def writeStatus(filePointer, args) :
    
    # now, most of the information should be here
    ec = args["_ec"]
    
    
    return

def readStatus(filePointer) :
    
    return

def IC_model(G, a, p):              # a: the set of initial active nodes
                                    # p: the system-wide probability of influence on an edge, in [0,1]
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False

    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random.random()  # in the range [0.0, 1.0)
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
                prob = random.random()  # in the range [0.0, 1.0)
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
def evaluate(G, A, p, no_simulations, model):
    results = []

    if model == 'IC':
        for i in range(no_simulations):
            results.append(IC_model(G, A, p))
    
    elif model == 'WC':
        for i in range(no_simulations):
            results.append(WC_model(G,A))

    return numpy.mean(results)

def nsga2evaluate(candidates, args) :
    fitness = []
    cache = args["cache"]
    individualsFile = args["individualsFile"]
    approximation = args["approximation"]
    timingFile = args["timingFile"]

    for A in candidates :
        phenotype = sorted(set(A)) # this consumes processing power, but it might also save a bunch of time
        tuplePhenotype = tuple(phenotype)
        if tuplePhenotype not in cache :
            
            startTime = time() # measure time
            if approximation == "edv" :
                fitness.append( inspyred.ec.emo.Pareto( [ approximations.approx_EDV(args["G"], A, args["p"]), 1.0 / float(len(set(A))) ] ) )
            
            elif approximation == "ps" :
                fitness.append( inspyred.ec.emo.Pareto( [ approximations.approx_repr_graph('PS', args["G"], A, args["p"], args["model"]), 1.0 / float(len(set(A))) ] ) )
            else :
                fitness.append( inspyred.ec.emo.Pareto( [ evaluate(args["G"], A, args["p"], args["no_simulations"], args["model"]), 1.0 / float(len(set(A))) ] ) )
            endTime = time() # measure time
            cache[tuplePhenotype] = fitness[-1]
            
            # write the time needed to run a fitness evaluation to file
            with open(timingFile, "a") as fp : fp.write( str(endTime - startTime) + "\n")
        else :
            fitness.append( cache[tuplePhenotype] )
    
    # dump everything to file
    #with open(individualsFile, "a") as fp :
    #   for i, c in enumerate(candidates) :
    #       fp.write( str(fitness[i][0]) + "," + str(1.0 / fitness[i][1]))
    #       for x in c : fp.write("," + str(x))
    #       fp.write("\n")
    return fitness

@inspyred.ec.generators.diversify # decorator that makes it impossible to generate copies
def nsga2generate(random, args) :
    
    phenotype_len = args["phenotype_len"]
    nodes = args["nodes"]

    # extract random number in 1,phenotype_len
    individual_size = random.randint(1, phenotype_len)
    individual = [0] * individual_size
    logging.info( "Creating individual of size %d, with genes ranging from %d to %d" % (individual_size, nodes[0], nodes[-1]) )
    for i in range(0, individual_size) : individual[i] = nodes[ random.randint(0, len(nodes)-1) ]
    logging.info(individual)
    return individual

@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover, even if it isn't in this case :-)
def nsga2superOperator(random, candidate1, candidate2, args) :
    
    children = []
    #print("I received candidate1:", candidate1, " and candidate2:", candidate2)
    
    # uniform choice of operator
    randomChoice = random.randint(0,3)
    
    if randomChoice == 0 :
        children = nsga2crossover(random, list(candidate1), list(candidate2), args)
    elif randomChoice == 1 :
        children.append( nsga2alterationMutation(random, list(candidate1), args) )
    elif randomChoice == 2 :
        children.append( nsga2insertionMutation(random, list(candidate1), args) )
    elif randomChoice == 3 :
        children.append( nsga2removalMutation(random, list(candidate1), args) )
    
    # purge the children from "None" and empty arrays
    children = [c for c in children if c is not None and len(c) > 0]
    
    # this should probably be commented or sent to logging
    for c in children : logging.debug("randomChoice=%d : from parent of size %d, created child of size %d" % (randomChoice, len(candidate1), len(c)) )

    return children

#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def nsga2alterationMutation(random, candidate, args) :
    
    #print("nsga2alterationMutation received this candidate:", candidate)
    nodes = args["nodes"]

    mutatedIndividual = list(set(candidate))

    # choose random place
    gene = random.randint(0, len(mutatedIndividual)-1)
    mutatedIndividual[gene] = nodes[ random.randint(0, len(nodes)-1) ]

    return mutatedIndividual

#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def nsga2insertionMutation(random, candidate, args) :
    
    #print("nsga2insertionMutation received this candidate:", candidate)
    phenotype_len = args["phenotype_len"]
    nodes = args["nodes"]
    mutatedIndividual = list(set(candidate))

    if len(mutatedIndividual) < phenotype_len :
        mutatedIndividual.append( nodes[ random.randint(0, len(nodes)-1) ] )
        return mutatedIndividual
    else :
        return None

#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def nsga2removalMutation(random, candidate, args) :
    
    #print("nsga2removalMutation received this candidate:", candidate)
    mutatedIndividual = list(set(candidate))

    if len(candidate) > 1 :
        gene = random.randint(0, len(mutatedIndividual)-1)
        mutatedIndividual.pop(gene)
        return mutatedIndividual
    else :
        return None

#@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover
def nsga2crossover(random, candidate1, candidate2, args) :
    
    children = []   
    phenotype_len = args["phenotype_len"]

    #print("nsga2crossover received this candidate:", candidate1)
    parent1 = list(set(candidate1))
    parent2 = list(set(candidate2))
    
    # choose random cut point
    cutPoint1 = random.randint(0, len(parent1)-1)
    cutPoint2 = random.randint(0, len(parent2)-1)
    
    # children start as empty lists
    child1 = []
    child2 = []
    
    # swap stuff
    for i in range(0, cutPoint1) : child1.append( parent1[i] )
    for i in range(0, cutPoint2) : child2.append( parent2[i] )
    
    for i in range(cutPoint1, len(parent2)) : child1.append( parent2[i] )
    for i in range(cutPoint2, len(parent1)) : child2.append( parent1[i] )
    
    # reduce children to minimal form
    child1 = list(set(child1))
    child2 = list(set(child2))
    
    # return the two children
    if len(child1) > 0 and len(child1) <= phenotype_len : children.append( child1 )
    if len(child2) > 0 and len(child2) <= phenotype_len : children.append( child2 )

    return children


def my_observer(population, num_generations, num_evaluations, args):
    
    # get time for previous generation and update 'timePreviousGeneration' with current time
    timePreviousGeneration = args['timePreviousGeneration']
    currentTime = time()
    timeElapsed = currentTime - timePreviousGeneration
    args['timePreviousGeneration'] = currentTime

    best = max(population)
    print('[{0:.2f} s] Generation {1:6} -- {2}'.format(timeElapsed, num_generations, best.fitness))
    #print(population)
    
    if matplotlibOK :
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("NSGA-II, generation=%d, evaluations=%d" % (num_generations, num_evaluations))
        ax.set_xlabel("influence")
        ax.set_ylabel("nodes in the seed set")
        
        # fix axes
        axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        #axes.set_ylim([1,400])

        x = [ k.fitness[0] for k in population ]
        y = [ 1.0 / float(k.fitness[1]) for k in population ]
        ax.plot(x, y, 'bo', label="individual in the population")
        plt.savefig("Generation-" + str(num_generations) + ".png")
        fig.clf()
        plt.close(fig)

    # save current generation and archive into a file?
    populationFileName = "population-gen-" + str(num_generations) + ".csv"
    archiveFileName = "archive-gen-" + str(num_generations) + ".csv"
    
    with open(populationFileName, "w") as fp :
        # header
        # TODO actually write stuff here
        fp.write("influence_fit0,length_fit1,nodes\n")
        
        for i in population :
            fp.write(str(i.fitness[0]) + "," + str(1.0/float(i.fitness[1])))
            for node in i.candidate :
                fp.write("," + str(node))
            fp.write("\n")
        
    archive = args["_ec"].archive
    if archive != None :
        with open(archiveFileName, "w") as fp :
            # header
            fp.write("influence_fit0,length_fit1,nodes\n")
            
            for i in archive :
                fp.write(str(i.fitness[0]) + "," + str(1.0/float(i.fitness[1])))
                for node in i.candidate :
                    fp.write("," + str(node))
                fp.write("\n")

def main() :
    
    # hard-coded values to be moved here
    typesOfApproximations = ["edv", "ps"]
    
    # initialize parser and parse arguments
    parser = argparse.ArgumentParser(description="Python script that evolves candidate seed nodes for an influence maximization problem in a social network, given a target graph and an influence propagation model.\nBy Doina Bucur, Giovanni Iacca, Andrea Marcelli, Giovanni Squillero, Alberto Tonda, 2016 <alberto.tonda@gmail.com>") 
    parser.add_argument("-g", "--graph", help="Target graph file")
    parser.add_argument("-m", "--model", choices=['IC', 'WC'], help="Influence propagation model. Values: 'IC' for Independent Cascade, 'WC' for Waterfall Cascade. Default value: 'WC'")
    parser.add_argument("-p", "--probability", type=float, help="Probability value, used for the IC propagation model, only. Default value: 0.01")
    parser.add_argument("-rc", "--recovery", help="Recovery file name to restart the evolution from.")
    parser.add_argument("-n", "--numberOfSimulations", type=int, help="Number of evaluations for each individual. Default value: 100.")
    parser.add_argument("-s", "--seedingFile", help="List of individuals that are used to seed the initial population.")
    parser.add_argument("-u", "--undirected", action='store_true', help="Flag that specifies that the graph is undirected. Default: the graph is considered to be directed.")
    parser.add_argument("-rv", "--reverse", action='store_true', help="Flag that specifies that the DIRECTED graph needs to be reversed. Default: the graph is not reversed.")
    parser.add_argument("-a", "--approximation", help="Type of approximation to be used in the experiments. Default value: None (simulations). Valid values: " + str(typesOfApproximations))

    # evolutionary arguments    
    parser.add_argument("-mn", "--maxNodes", type=int, help="Maximum values of nodes in a candidate seed set. Default value: 200")
    parser.add_argument("-mg", "--maxGenerations", type=int, help="Maximum generations for the evolutionary algorithm. Default value: 10,000")
    parser.add_argument("-mu", "--mu", type=int, help="Population size for the evolutionary algorithm. Default value: 100")
    parser.add_argument("-mc", "--maxCpus", type=int, help="Max number of CPUs to be used. Default value: 4")

    args = parser.parse_args()
    
    # initialize logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    fh = logging.FileHandler('log.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # now, there are a few cases that are acceptable
    # graph specified and recovery not specified
    if (args.graph == None and args.recovery == None) or (args.graph != None and args.recovery != None) :
        logging.warning("Error: either \"-g/--graph\" or \"-r/--recovery\" arguments have to be specified, not both.")
        parser.print_help()
        sys.exit(0) 

    # define a few constants in the problem
    input_graph=""
    p = 0.01
    model = "WC"
    no_simulations = 100
    current_generation = 0
    max_generations = 10000
    pop_size = 2000
    max_cpus = 4
    max_nodes = 200
    approximation = None
    #prng = numpy.random.RandomState() # this might be replaced by the random state read through the recovery part
    prng = random.Random()
    prng.seed(time()) 
    
    # override default values with arguments from command line (if they have been specified)
    if args.mu : pop_size = args.mu
    if args.numberOfSimulations : no_simulations = args.numberOfSimulations
    if args.model : model = args.model
    if args.maxGenerations : max_generations = args.maxGenerations
    if args.maxCpus : max_cpus = args.maxCpus
    if args.maxNodes : max_nodes = args.maxNodes
    if args.approximation : 
        if args.approximation in typesOfApproximations :
            approximation = args.approximation
        else :
            logging.warning("Error: unrecognized approximation \"%s\" specified on command line. Look at the help for valid values." % args.approximation)
            sys.exit(0)
    
    # some information will be written to the log file
    logging.info("Population size:" + str(pop_size))
    logging.info("Number of simulations:" + str(no_simulations))
    logging.info("Model:" + str(model))
    logging.info("Max generations:" + str(max_generations))
    logging.info("Max cpus:" + str(max_cpus))
    logging.info("Max nodes:" + str(max_nodes))
    logging.info("Approximation:" + str(approximation))
    
    # now, let's check if a graph name was specified
    if args.graph != None : 
        input_graph = args.graph
        if args.model != None : model = args.model
        if args.probability != None : p = args.probability
    else :
        # a recovery file was specified; in that case, let's parse it to get the proper information
        with open(args.recovery, "r") as fp :
            lines = fp.readlines()
        # TODO all parsing...actually, we'll need two files: last population and last archive
    
    # in any case, let's read the graph
    logging.info("Reading graph...")
    G= None
    
    # check whether the graph is directed or undirected, and read data accordingly
    if args.undirected :
        logging.info("The graph is undirected...")
        G = nx.read_edgelist(input_graph, comments='#', delimiter=' ', create_using=nx.Graph(), nodetype=int, data=False)
    else :
        logging.info("The graph is directed...")
        G = nx.read_edgelist(input_graph, comments='#', delimiter=' ', create_using=nx.DiGraph(), nodetype=int, data=False)
    
        if args.reverse :
            logging.info("The graph needs to be reversed...")
            G = G.reverse()

    # this "list( G.nodes() )" has been added because the new version of networkx has a weird behavior with "nodes = G.nodes()"
    nodes = list( G.nodes() )
    logging.info("Found a total of %d nodes" % (len(nodes)) )
    phenotype_len = len(nodes)

    # write the header of the CSV file, along with some information
    individualsFile = "status.csv"
    with open(individualsFile, "w") as fp : 
        #fp.write("# random_state=" + str(prng.get_state()) + "\n")
        fp.write("# graph=\"" + input_graph + "\"\n")
        fp.write("# propagation_model=\"" + model + "\"\n")
        fp.write("# pop_size=" + str(pop_size) + "\n")
        fp.write("# current_generation=" + str(current_generation) + "\n")
        fp.write("# max_generations=" + str(max_generations) + "\n")
        fp.write("# max_nodes=" + str(max_nodes) + "\n")
        fp.write("influence_fit0,length_fit1,nodes\n")
    
    # set up the evolutionary stuff
    logging.info("Setting up NSGA-II...")
    ea = inspyred.ec.emo.NSGA2(prng)
    ea.observer = my_observer
    ea.variator = [nsga2superOperator]  
    ea.terminator = inspyred.ec.terminators.generation_termination
    
    # if a seeding file has been specified, read the list of seed individuals and add them to the generator function
    # (using standard inspyred stuff)
    seedIndividuals = []
    if args.seedingFile :
        logging.info("Reading seeding file \"" + args.seedingFile + "\"...")
        
        # this is an attempt to create a "smart" parsing function, able to parse
        # both a file with one individual per line, and the heuristic results of Doina
        # with the format: <numberOfNodes> <fitness> <stdv> [<node1>, <node2>, ... , <nodeN>]
        with open(args.seedingFile, "r") as fp :
            lines = fp.readlines()
            
            for line in lines :
                # try to capture the group "[<whatever>]" with regex
                # if success, replace line with the group
                matches = regex.search("\[(.*)\]", line)
                if matches != None : line = matches.group(1)

                individual = [ float(x) for x in line.rstrip().split(',') ]
                seedIndividuals.append( individual )

        # the list is later passed to the "evolve" method using the "seeds" argument (see below)
        logging.info("File read successfully.")

    # custom stuff
    final_pop = ea.evolve(
            generator=nsga2generate,
            #evaluator = nsga2evaluate,
            pop_size=pop_size,
            maximize=True,
            max_generations=max_generations,
            seeds=seedIndividuals, 
            #selector = my_selector,
            # this part is defined to use multi-process evaluations
            evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
            mp_evaluator=nsga2evaluate, 
            mp_num_cpus=max_cpus,
            # extra arguments, will be passed to the functions in the "args" dictionary
            approximation = approximation,
            phenotype_len = max_nodes,
            p = p,
            model = model,
            no_simulations = no_simulations,
            G = G,
            nodes = nodes,
            cache = dict(),
            individualsFile = individualsFile,
            timingFile = "time-evaluations.csv",
            timePreviousGeneration = time() # this will be updated in the observer 
            )
    
    # plot the final result
    print("final_pop:", final_pop)

    # load file 
    print("Saving figure to file \"" + individualsFile[:-4] + ".pdf\"...")
    final_arc = ea.archive
    x = []
    y = []
    for f in final_arc :
        x.append(f.fitness[0])
        y.append(1.0 / float(f.fitness[1]))
    
    # also load all evaluated individuals
    # influenceValues = []
    # nodesValues = []
    # with open(individualsFile, "r") as fp :
    #   lines = fp.readlines()
    #   # pop the header
    #   header = lines.pop(0)
        
    #   for line in lines :
    #       tokens = line.rstrip().split(',')
    #       influenceValues.append( float(tokens[0]) )
    #       nodesValues.append( float(tokens[1]) )

    # if matplotlibOK :
    #   fig = plt.figure()
    #   ax = fig.add_subplot(111)
    #   ax.set_title("NSGA-II, population size=%d, max generations=%d" % (pop_size, max_generations))
    #   ax.set_xlabel("influence")
    #   ax.set_ylabel("nodes in the seed set")
    #   ax.plot(influenceValues, nodesValues, 'bo', label="all evaluated individuals")
    #   ax.plot(x, y, 'ro', label="last Pareto front")
    #   plt.savefig(individualsFile[:-4] + ".pdf")
    
    
if __name__ == "__main__" :
    sys.exit(main())
