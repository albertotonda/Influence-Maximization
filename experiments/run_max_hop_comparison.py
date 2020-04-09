import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import sys

# add folder to Python path and import local scripts
sys.path.append("../")
from spread import MonteCarlo_simulation_max_hop
from spread import MonteCarlo_simulation
from evolutionaryalgorithm import moea_influence_maximization  

def main() :

    # a few hard-coded values
    pop_size = 100
    offspring_size = 100
    max_generations = 100
    evaluator = MonteCarlo_simulation_max_hop
    #evaluator = MonteCarlo_simulation

    # create folder for experiment and initialize logger with console INFO and file DEBUG information
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-max-hop-comparison"
    if not os.path.isdir(folder_name) : os.mkdir(folder_name)

    log_name = os.path.join(folder_name, "log.log")

    # create an additional logger
    logger = logging.getLogger(log_name)

    # format log file
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s",
                                  "%Y-%m-%d %H:%M:%S")

    # the 'RotatingFileHandler' object implements a log file that is automatically limited in size
    fh = RotatingFileHandler(log_name,
                             mode='a',
                             maxBytes=100*1024*1024,
                             backupCount=2,
                             encoding=None,
                             delay=0)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Logging started, on \"" + log_name + "\"!")

    # now, here is the meat of the program

    # at the end of the process, close logging
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


    return

if __name__ == "__main__" :
    sys.exit( main() )
