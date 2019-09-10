import argparse
import datetime
import logging
import os
import sys

from logging.handlers import RotatingFileHandler

def main() :
	
	# get command-line arguments
	args = parse_command_line()

	# create folder with unique name
	folderName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
	folderName += "-unique-name"
	if not os.path.exists(folderName) : os.makedirs(folderName)

	# initialize logging, using a logger that smartly manages disk occupation
	initialize_logging(folderName)

	# start program
	logging.info("Hi, I am a program, starting now!")

	return

def initialize_logging(folderName=None) :
	logger = logging.getLogger('')
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S') 

	# the 'RotatingFileHandler' object implements a log file that is automatically limited in size
	if folderName != None :
		fh = RotatingFileHandler( os.path.join(folderName, "log.log"), mode='a', maxBytes=100*1024*1024, backupCount=2, encoding=None, delay=0 )
		fh.setLevel(logging.DEBUG)
		fh.setFormatter(formatter)
		logger.addHandler(fh)

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	
	return

def parse_command_line() :
	
	parser = argparse.ArgumentParser(description="Python script that evolves candidate land uses for Small Agricultural Regions.\nBy Francesco Accatino and Alberto Tonda, 2017-2019 <alberto.tonda@gmail.com>")
	
	# required argument
	#parser.add_argument("-sar", "--sar", help="File containing the list of all the SARs (Small Agricultural Regions).", required=True)	
	
	# list of elements, type int
	#parser.add_argument("-rid", "--regionId", type=int, nargs='+', help="List of regional IDs. All SARs belonging to regions with these IDs will be included in the optimization process.")
	
	# flag, it's just true/false
	#parser.add_argument("-sz", "--startFromZero", action='store_true', help="If this flag is set, the algorithm will include an individual with genome '0' [0,0,...,0] in the initial population. Useful to improve speed of convergence, might cause premature convergence.")
		
	args = parser.parse_args()
	
	return args

if __name__ == "__main__" :
	sys.exit( main() )
