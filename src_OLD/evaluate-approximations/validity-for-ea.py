# This is a simple script, aiming at evaluating the validity of approximation metrics for an EA
# In practice, an EA only cares about relative values (e.g. is value for individual A bigger than value for individual B)
# and not about the absolute validity of the metric
# by Alberto Tonda, 2017 <alberto.tonda@gmail.com>

import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pandas import read_csv
from scipy import stats

# there is only one argument, super-simple script
metricsFile = sys.argv[1]

dataFrame = read_csv(metricsFile)

# now, here are the interesting columns
approximations = ["EDV", "PS"]
simulationMean = "IC_mean"
simulationStd = "IC_std"

# first, let's find all different values of size for the seed sets
sizes = sorted( dataFrame["size"].unique() )
print("Different sizes of the seed sets:", sizes)

# dictionaries later used for statistics
allErrors = dict()
allDistancesError = dict()
allDistancesCorrect = dict()
allApproximationDifferenceCorrect = dict()
allApproximationDifferenceError = dict()

for approximation in approximations : 
	allErrors[approximation] = []
	allDistancesError[approximation] = []
	allDistancesCorrect[approximation] = []
	allApproximationDifferenceCorrect[approximation] = []
	allApproximationDifferenceError[approximation] = []

# now, let's select all rows with a certain size
for size in sizes :
	
	selection = dataFrame.loc[ dataFrame["size"] == size ]
	numberOfRows = len(selection)
	# the '//' here return integers
	numberOfCombinations = int(math.factorial(numberOfRows) // math.factorial(2) // math.factorial(numberOfRows-2))
	print("\nThere are " + str(numberOfRows) + " rows with seed set size " + str(size))
	
	# let's create a couple of dictionaries to keep track of the errors
	errors = dict()
	errorsWithinSD = dict()
	distanceWhenError = dict()
	distanceWhenCorrect = dict()
	approximationDifferenceWhenError = dict()
	approximationDifferenceWhenCorrect = dict()
	
	for approximation in approximations :
		errors[approximation] = 0
		errorsWithinSD[approximation] = 0
		distanceWhenError[approximation] = []
		distanceWhenCorrect[approximation] = []
		approximationDifferenceWhenError[approximation] = []
		approximationDifferenceWhenCorrect[approximation] = []

	# now, let's examine all couples of lines, to see if values of simulations and approximations are coherent
	list1 = list(selection.iterrows())
	combinations = list( itertools.combinations(list1, 2) )

	for c in combinations :
		
		s1 = c[0][1]
		s2 = c[1][1]

		# for every approximation, does the difference between values have the same sign as the difference
		# between the simulations?
		for approximation in approximations :
			
			# the first check is pretty straightforward
			if np.sign( s1[approximation] - s2[approximation] ) != np.sign( s1[simulationMean] - s2[simulationMean] ) :
			#if ( s1[approximation] > s2[approximation] and s1[simulationMean] < s2[simulationMean] ) or ( s1[approximation] < s2[approximation] and s1[simulationMean] > s2[simulationMean] ) :
				errors[approximation] += 1
				errorsWithinSD[approximation] += 1
				distanceWhenError[approximation].append( abs(s1[simulationMean] - s2[simulationMean]) )
				approximationDifferenceWhenError[approximation].append( abs(s1[approximation] - s2[approximation]) )
			
				# the second check is less trivial: are the two IC simulations *really* separable using a statistical
				# tests? If not, then no matter what results is returned by the approximation, it will be "correct"
				distribution1 = np.random.normal(s1["IC_mean"], s1["IC_std"], 1000)
				distribution2 = np.random.normal(s2["IC_mean"], s2["IC_std"], 1000)
				#ks, p = stats.ks_2samp(distribution1, distribution2)
				ks, p = stats.ttest_ind(distribution1, distribution2)
				#print("t=" + str(ks) + ", p=" + str(p))
				
				# if the two distributions coming from the simulations are non-separable, then the error is not an error
				# NOTE for Facebook, it NEVER enters this check...so, all simulations seem to be separable
				if p > 0.05 : errorsWithinSD[approximation] -= 1
				
			else :
				# difference between approximations and simulations goes in the same direction
				distanceWhenCorrect[approximation].append( abs(s1[simulationMean] - s2[simulationMean]) )
				approximationDifferenceWhenCorrect[approximation].append( abs(s1[approximation] - s2[approximation]) )
	
	for approximation in approximations :
		print("For size " + str(size) + ", approximation " + str(approximation) + " reported " + str(errors[approximation]) + "/" + str(numberOfCombinations) + " errors (" + str(float(errors[approximation])/numberOfCombinations * 100) + "%).")
		print("For size " + str(size) + ", approximation " + str(approximation) + " reported " + str(errors[approximation]) + "/" + str(numberOfCombinations) + " errors (taking into account non-separable distributions).")
		print("Mean difference between simulations in case of errors: %.2f +/- %.2f" % (np.mean(distanceWhenError[approximation]), np.std(distanceWhenError[approximation])))
		print("Mean difference between simulations in correct cases: %.2f +/- %.2f" % ( np.mean(distanceWhenCorrect[approximation]), np.std(distanceWhenCorrect[approximation])))
		print("Mean difference between approximations in case of errors: %.2f +/- %.2f" % (np.mean(approximationDifferenceWhenError[approximation]), np.std(approximationDifferenceWhenError[approximation])))
		print("Mean difference between approximations in correct cases: %.2f +/- %.2f" % (np.mean(approximationDifferenceWhenCorrect[approximation]), np.std(approximationDifferenceWhenCorrect[approximation])))
				
		allErrors[approximation].append( float(errors[approximation])/numberOfCombinations )
		allDistancesError[approximation].append( (np.mean(distanceWhenError[approximation]), np.std(distanceWhenError[approximation])) )
		allDistancesCorrect[approximation].append( (np.mean(distanceWhenCorrect[approximation]), np.std(distanceWhenCorrect[approximation])) )
		allApproximationDifferenceError[approximation].append( (np.mean(approximationDifferenceWhenError[approximation]), np.std(approximationDifferenceWhenError[approximation])) )
		allApproximationDifferenceCorrect[approximation].append( (np.mean(approximationDifferenceWhenCorrect[approximation]), np.std(approximationDifferenceWhenCorrect[approximation])) )

print("\nAnd now, for some final stats:")
for approximation in approximations :

	print("Average error for approximation " + approximation + " is:" + str(np.mean(allErrors[approximation])) )
	print("Plotting figures...")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("Errors of comparison (different sign) for approximation " + approximation)
	ax.set_xlabel("size of the seed set")
	ax.set_ylabel("percentage of errors")
	
	plt.plot(np.arange(1, 201), allErrors[approximation])
	
	fig.savefig("errors-" + approximation + ".png")
	plt.close(fig)
	
	# also, a figure that shows how mean difference changes in cases of errors and correct behavior
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("Mean difference between simulations")
	ax.set_xlabel("size of the seed set")
	ax.set_ylabel("mean difference")
	
	distancesCorrect = [ x[0] for x in allDistancesCorrect[approximation] ]
	distancesError = [ x[0] for x in allDistancesError[approximation] ]

	plt.plot(np.arange(1, 201), distancesCorrect, label='mean difference (correct)')
	plt.plot(np.arange(1, 201), distancesError, label='mean difference (error)')
	
	ax.legend(loc='best')
	fig.savefig("mean-simulation-distance-" + approximation + ".png")
	plt.close(fig)
	
	# also, a figure that shows mean differences for approximations
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("Mean difference between approximations")
	ax.set_xlabel("size of the seed set")
	ax.set_ylabel("mean difference")
	
	distancesCorrect = [ x[0] for x in allApproximationDifferenceCorrect[approximation] ]
	distancesError = [ x[0] for x in allApproximationDifferenceError[approximation] ]
	
	plt.plot(np.arange(1, 201), distancesCorrect, label='mean difference (correct)')
	plt.plot(np.arange(1, 201), distancesError, label='mean difference (error)')
	
	ax.legend(loc='best')
	fig.savefig("mean-approximation-distance-" + approximation + ".png")
	plt.close(fig)
	
	# TODO save results in a .csv file; the relevant information is going to be
	# size of the seed set, number of cases, percentage of errors, mean distance simulation (error), std (error), mean distance simulation (correct), std (correct), mean distance approximation (correct), std (correct), ...and the same for errors
	
	with open("summary-" + approximation + ".csv", "w") as fp :
		
		# header
		fp.write("size_seed_set,instances,error_percentage,mean_difference_error_simulation,std_error_simulation,mean_difference_correct_simulation,std_correct_simulation,mean_difference_error_approximation,std_error_approximation,mean_difference_correct_approximation,std_correct_approximation\n")
		
		# now, let's write the data!
		for size in range(0, 200) :

			fp.write( str(size+1) ) # size of the seed set
			fp.write( "," + str(len(dataFrame.loc[dataFrame["size"] == size+1])) ) # instances found
			fp.write( "," + str(allErrors[approximation][size]) ) # error-to-total ratio

			fp.write( "," + str(allDistancesError[approximation][size][0]) )
			fp.write( "," + str(allDistancesError[approximation][size][1]) )

			fp.write( "," + str(allDistancesCorrect[approximation][size][0]) )
			fp.write( "," + str(allDistancesCorrect[approximation][size][1]) )
			
			fp.write( "," + str(allApproximationDifferenceError[approximation][size][0]) )
			fp.write( "," + str(allApproximationDifferenceError[approximation][size][1]) )
			
			fp.write( "," + str(allApproximationDifferenceCorrect[approximation][size][0]) )
			fp.write( "," + str(allApproximationDifferenceCorrect[approximation][size][1]) )
			
			fp.write( "\n" )
