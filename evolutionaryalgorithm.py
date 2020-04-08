"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. Ideally, it will eventually contain both the single-objective (maximize influence with a fixed amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. This relies upon the inspyred Python library for evolutionary algorithms."""

# general libraries
import inspyred
import logging
import random

from time import time, strftime

# local libraries
import spread

"""
Multi-objective evolutionary influence maximization. Parameters:
    G: networkx graph
    p: probability of influence spread 
    no_simulations: number of simulations
    model: type of influence propagation model
    population_size: population of the EA (default: value)
    offspring_size: offspring of the EA (default: value)
    max_generations: maximum generations (default: value)
    min_seed_nodes: minimum number of nodes in a seed set (default: 1)
    max_seed_nodes: maximum number of nodes in a seed set (default: 1% of the graph size)
    n_threads: number of threads to be used for concurrent evaluations (default: 1)
    random_gen: already initialized pseudo-random number generation
    initial_population: individuals (seed sets) to be added to the initial population (the rest will be randomly generated)
    population_file: name of the file that will be used to store the population at each generation (default: file named with date and time)
    """
def moea_influence_maximization(G, p, no_simulations, model, population_size=100, offspring_size=100, max_generations=100, min_seed_nodes=None, max_seed_nodes=None, n_threads=1, random_gen=random.Random(), initial_population=None, population_file=None) :

    # initialize multi-objective evolutionary algorithm, NSGA-II
    logging.debug("Setting up NSGA-II...")

    # check if some of the parameters are set; otherwise, use default values
    nodes = list(G.nodes)
    if min_seed_nodes == None :
        min_seed_nodes = 1
        logging.debug("Minimum size for the seed set has been set to %d" % min_seed_nodes)
    if max_seed_nodes == None : 
        max_seed_nodes = int( 0.1 * len(nodes))
        logging.debug("Maximum size for the seed set has been set to %d" % max_seed_nodes)
    if population_file == None :
        ct = time()
        population_file = strftime("%Y-%m-%d-%H-%M-%S-population.csv")

    ea = inspyred.ec.emo.NSGA2(random_gen)
    ea.observer = ea_observer
    ea.variator = [nsga2_super_operator]
    ea.terminator = inspyred.ec.terminators.generation_termination
    
    # start the evolutionary process
    final_population = ea.evolve(
        generator = nsga2_generator,
        evaluator = nsga2_evaluator,
        maximize = True,
        seeds = initial_population,
        pop_size = population_size,
        num_selected = offspring_size,
        max_generations = max_generations,

        # all arguments below will go inside the dictionary 'args'
        G = G,
        p = p,
        model = model,
        no_simulations = no_simulations,
        nodes = nodes,
        n_threads = n_threads,
        min_seed_nodes = min_seed_nodes,
        max_seed_nodes = max_seed_nodes,
        population_file = population_file,
        time_previous_generation = time(), # this will be updated in the observer
    )

    # extract seed sets from the final Pareto front/archive
    seed_sets = [ [individual.candidate, individual.fitness[0]] for individual in ea.archive ] 

    return seed_sets

def nsga2_evaluator(candidates, args) :

    n_threads = args["n_threads"]
    G = args["G"]
    p = args["p"]
    model = args["model"]
    no_simulations = args["no_simulations"]

    # we start with a list where every element is None
    fitness = [None] * len(candidates)

    # depending on how many threads we have at our disposal,
    # we use a different methodology
    # if we just have one thread, let's just evaluate individuals old style 
    if n_threads == 1 :
        for index, A in enumerate(candidates) :

            # TODO sort phenotype, use cache...? or manage sorting directly during individual creation? see lines 108-142 in src_OLD/multiObjective-inspyred/sn-inflmax-inspyred.py 
            # TODO maybe if we make sure that candidates are already sets before getting here, we could save some computational time
            A_set = set(A)

            # TODO consider std inside the fitness in some way?
            influence_mean, influence_std = spread.MonteCarlo_simulation(G, A_set, p, no_simulations, model)
            fitness[index] = inspyred.ec.emo.Pareto([influence_mean, 1.0 / float(len(A_set))])

    else :
        
        # create a threadpool, using the local module
        import threadpool
        thread_pool = threadpool.ThreadPool(n_threads)

        # create thread lock, to be used for concurrency
        import threading
        thread_lock = threading.Lock()

        # create list of tasks for the thread pool, using the threaded evaluation function
        tasks = [ (G, p, A, no_simulations, model, fitness, index, thread_lock) for index, A in enumerate(candidates) ]
        thread_pool.map(nsga2_evaluator_threaded, tasks)

        # start thread pool and wait for conclusion
        thread_pool.wait_completion()

    return fitness

def nsga2_evaluator_threaded(G, p, A, no_simulations, model, fitness, index, thread_lock, thread_id) :

    # TODO add logging?
    A_set = set(A)
    influence_mean, influence_std = spread.MonteCarlo_simulation(G, A_set, p, no_simulations, model)

    # lock data structure before writing in it
    thread_lock.acquire()
    fitness[index] = inspyred.ec.emo.Pareto([influence_mean, 1.0 / float(len(A_set))])  
    thread_lock.release()

    return 

def ea_observer(population, num_generations, num_evaluations, args) :

    time_previous_generation = args['time_previous_generation']
    currentTime = time()
    timeElapsed = currentTime - time_previous_generation
    args['time_previous_generation'] = currentTime

    best = max(population)
    logging.info('[{0:.2f} s] Generation {1:6} -- {2}'.format(timeElapsed, num_generations, best.fitness))

    # TODO write current state of the ALGORITHM to a file (e.g. random number generator, time elapsed, stuff like that)
    # write current state of the population to a file
    population_file = args["population_file"]
    
    # find the longest individual
    max_length = len(max(population, key=lambda x : len(x.candidate)).candidate)

    with open(population_file, "w") as fp :
        # header, of length equal to the maximum individual length in the population
        fp.write("n_nodes,influence")
        for i in range(0, max_length) : fp.write(",n%d" % i)
        fp.write("\n")

        # and now, we write stuff, individual by individual
        for individual in  population :

            # check if fitness is an iterable collection (e.g. a list) or just a single value
            if hasattr(individual.fitness, "__iter__") :
                fp.write("%d,%.4f" % (1.0 / individual.fitness[1], individual.fitness[0]))
            else :
                fp.write("%d,%.4f" % ( len(set(individual.candidate)), individual.fitness))

            for node in individual.candidate :
                fp.write(",%d" % node)

            for i in range(len(individual.candidate), max_length - len(individual.candidate)) :
                fp.write(",")

            fp.write("\n")

    return

# TODO is there a way to have a multi-threaded generation of individuals?
@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover, even if it isn't in this case :-)
def nsga2_super_operator(random, candidate1, candidate2, args) :

    children = []

    # uniform choice of operator
    randomChoice = random.randint(0,3)
    
    if randomChoice == 0 :
        children = nsga2_crossover(random, list(candidate1), list(candidate2), args)
    elif randomChoice == 1 :
        children.append( ea_alteration_mutation(random, list(candidate1), args) )
    elif randomChoice == 2 :
        children.append( nsga2_insertion_mutation(random, list(candidate1), args) )
    elif randomChoice == 3 :
        children.append( nsga2_removal_mutation(random, list(candidate1), args) )
    
    # this should probably be commented or sent to logging
    for c in children : logging.debug("randomChoice=%d : from parent of size %d, created child of size %d" % (randomChoice, len(candidate1), len(c)) )

    # purge the children from "None" and empty arrays
    children = [c for c in children if c is not None and len(c) > 0]
    
    return children

#@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover
def nsga2_crossover(random, candidate1, candidate2, args) : 

    children = []   
    max_seed_nodes = args["max_seed_nodes"]

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
    if len(child1) > 0 and len(child1) <= max_seed_nodes : children.append( child1 )
    if len(child2) > 0 and len(child2) <= max_seed_nodes : children.append( child2 )

    return children

#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def ea_alteration_mutation(random, candidate, args) :
    
    #print("nsga2alterationMutation received this candidate:", candidate)
    nodes = args["nodes"]

    mutatedIndividual = list(set(candidate))

    # choose random place
    gene = random.randint(0, len(mutatedIndividual)-1)
    mutatedIndividual[gene] = nodes[ random.randint(0, len(nodes)-1) ]

    return mutatedIndividual

#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def nsga2_insertion_mutation(random, candidate, args) :
    
    max_seed_nodes = args["max_seed_nodes"]
    nodes = args["nodes"]
    mutatedIndividual = list(set(candidate))

    if len(mutatedIndividual) < max_seed_nodes :
        mutatedIndividual.append( nodes[ random.randint(0, len(nodes)-1) ] )
        return mutatedIndividual
    else :
        return None

# TODO take into account minimal seed set size
#@inspyred.ec.variators.mutator # decorator that defines the operator as a mutation
def nsga2_removal_mutation(random, candidate, args) :
    
    mutatedIndividual = list(set(candidate))

    if len(candidate) > 1 :
        gene = random.randint(0, len(mutatedIndividual)-1)
        mutatedIndividual.pop(gene)
        return mutatedIndividual
    else :
        return None

@inspyred.ec.generators.diversify # decorator that makes it impossible to generate copies
def nsga2_generator(random, args) :
    
    min_seed_nodes = args["min_seed_nodes"]
    max_seed_nodes = args["max_seed_nodes"]
    nodes = args["nodes"]
    logging.debug("Min seed set size: %d; Max seed set size: %d" % (min_seed_nodes, max_seed_nodes))

    # extract random number in 1,max_seed_nodes
    individual_size = random.randint(min_seed_nodes, max_seed_nodes)
    individual = [0] * individual_size
    logging.debug( "Creating individual of size %d, with genes ranging from %d to %d" % (individual_size, nodes[0], nodes[-1]) )
    for i in range(0, individual_size) : individual[i] = nodes[ random.randint(0, len(nodes)-1) ]
    logging.debug(individual)

    return individual

"""Single-objective evolutionary influence maximization. Parameters:
    k: seed set size
    G: networkx graph
    p: probability of influence spread 
    no_simulations: number of simulations
    model: type of influence propagation model
    population_size: population of the EA (default: value)
    offspring_size: offspring of the EA (default: value)
    max_generations: maximum generations (default: value)
    n_threads: number of threads to be used for concurrent evaluations (default: 1)
    random_gen: already initialized pseudo-random number generation
    initial_population: individuals (seed sets) to be added to the initial population (the rest will be randomly generated)
    """
def ea_influence_maximization(k, G, p, no_simulations, model, population_size=100, offspring_size=100, max_generations=100, n_threads=1, random_gen=random.Random(), initial_population=None, population_file=None) :

    # initialize a generic evolutionary algorithm
    logging.debug("Initializing Evolutionary Algorithm...")

    # check if some of the optional parameters are set; otherwise, use default values
    nodes = list(G.nodes)
    if population_file == None :
        ct = time()
        population_file = strftime("%Y-%m-%d-%H-%M-%S-population.csv")

    # instantiate a basic EvolutionaryComputation object, that is "empty" (no default methods defined for any component)
    # so we will need to define every method
    ea = inspyred.ec.EvolutionaryComputation(random_gen)
    ea.observer = ea_observer
    ea.variator = [ea_super_operator]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.selector = inspyred.ec.selectors.tournament_selection # default size is 2
    ea.replacer = inspyred.ec.replacers.plus_replacement

    # start the evolutionary process
    final_population = ea.evolve(
        generator = ea_generator,
        evaluator = ea_evaluator,
        maximize = True,
        seeds = initial_population,
        pop_size = population_size,
        num_selected = offspring_size,
        max_generations = max_generations,

        # all arguments below will go inside the dictionary 'args'
        k = k,
        G = G,
        p = p,
        model = model,
        no_simulations = no_simulations,
        nodes = nodes,
        n_threads = n_threads,
        population_file = population_file,
        time_previous_generation = time(), # this will be updated in the observer
    )

    best_individual = max(final_population)
    best_seed_set = best_individual.candidate
    spread = best_individual.fitness

    return best_seed_set, spread

@inspyred.ec.generators.diversify # decorator that makes it impossible to generate copies
def ea_generator(random, args) :

    # k is the size of the seed sets
    k = args["k"]
    nodes = args["nodes"]

    # extract random number in 1,max_seed_nodes
    individual = [0] * k
    logging.debug( "Creating individual of size %d, with genes ranging from %d to %d" % (k, nodes[0], nodes[-1]) )
    for i in range(0, k) : individual[i] = nodes[ random.randint(0, len(nodes)-1) ]
    logging.debug(individual)

    return individual

@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover, even if it isn't in this case :-)
def ea_super_operator(random, candidate1, candidate2, args) :

    k = args["k"]
    children = []

    # uniform choice of operator
    randomChoice = random.randint(0,1)
    
    # one-point crossover or mutation that swaps exactly one node with another
    if randomChoice == 0 :
        children = inspyred.ec.variators.n_point_crossover(random, [list(candidate1), list(candidate2)], args)
    elif randomChoice == 1 :
        children.append( ea_alteration_mutation(random, list(candidate1), args) )
    
    # this should probably be commented or sent to logging
    for c in children : logging.debug("randomChoice=%d : from parent of size %d, created child of size %d" % (randomChoice, len(candidate1), len(c)) )

    # purge the children from "None" and arrays of the wrong size
    children = [c for c in children if c is not None and len(set(c)) == k]

    return children

def ea_evaluator(candidates, args) :

    n_threads = args["n_threads"]
    G = args["G"]
    p = args["p"]
    model = args["model"]
    no_simulations = args["no_simulations"]

    # we start with a list where every element is None
    fitness = [None] * len(candidates)

    # depending on how many threads we have at our disposal,
    # we use a different methodology
    # if we just have one thread, let's just evaluate individuals old style 
    if n_threads == 1 :
        for index, A in enumerate(candidates) :

            # TODO sort phenotype, use cache...? or manage sorting directly during individual creation? see lines 108-142 in src_OLD/multiObjective-inspyred/sn-inflmax-inspyred.py 
            # TODO maybe if we make sure that candidates are already sets before getting here, we could save some computational time
            A_set = set(A)

            # TODO consider std inside the fitness in some way?
            influence_mean, influence_std = spread.MonteCarlo_simulation(G, A_set, p, no_simulations, model)
            fitness[index] = influence_mean

    else :
        
        # create a threadpool, using the local module
        import threadpool
        thread_pool = threadpool.ThreadPool(n_threads)

        # create thread lock, to be used for concurrency
        import threading
        thread_lock = threading.Lock()

        # create list of tasks for the thread pool, using the threaded evaluation function
        tasks = [ (G, p, A, no_simulations, model, fitness, index, thread_lock) for index, A in enumerate(candidates) ]
        thread_pool.map(ea_evaluator_threaded, tasks)

        # start thread pool and wait for conclusion
        thread_pool.wait_completion()

    return fitness

def ea_evaluator_threaded(G, p, A, no_simulations, model, fitness, index, thread_lock, thread_id) :

    # TODO not sure that this is needed
    A_set = set(A)

    # run spread simulation
    influence_mean, influence_std = spread.MonteCarlo_simulation(G, A_set, p, no_simulations, model)

    # lock shared resource, write in it, release
    thread_lock.acquire()
    fitness[index] = influence_mean
    thread_lock.release()

    return

# this main here is just to test the current implementation
if __name__ == "__main__" :

    # initialize logging
    import logging
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG) # TODO switch between INFO and DEBUG for less or more in-depth logging
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S') 
 
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    import load
    k = 5
    G = load.read_graph("graphs/facebook_combined_undirected.txt")
    p = 0.01
    model = 'WC'
    no_simulations = 100
    max_generations = 5
    n_threads = 1
    random_seed = 42

    prng = random.Random()
    if random_seed == None: 
        random_seed = time()
    logging.debug("Random number generator seeded with %s" % str(random_seed))
    prng.seed(random_seed)

    seed_sets = moea_influence_maximization(G, p, no_simulations, model, population_size=16, offspring_size=16, random_gen=prng, max_generations=max_generations, n_threads=n_threads)
    #seed_set, spread = ea_influence_maximization(k, G, p, no_simulations, model, population_size=16, offspring_size=16, random_gen=prng, max_generations=max_generations, n_threads=n_threads)
