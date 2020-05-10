import logging
import networkx as nx

""" Graph loading """

def read_graph(filename, nodetype=int):

	graph_class = nx.DiGraph() # all graph files are directed
	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)

	msg = ' '.join(["Read from file", filename, "the directed graph\n", nx.classes.function.info(G)])
	logging.info(msg)

	return G

if __name__ == '__main__':

	logger = logging.getLogger('')
	logger.setLevel(logging.DEBUG)
	read_graph("graphs/Email_URV.txt")
