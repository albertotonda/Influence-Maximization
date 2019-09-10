import logging
import networkx as nx

def read_graph(filename, directed=False, nodetype=int):

	graph_class = nx.DiGraph() if directed else nx.Graph()
	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)

	msg = ' '.join(["Read from file", filename, "the", "directed" if directed else "undirected", "graph\n",
		nx.classes.function.info(G)])
	logging.info(msg)

	return G

if __name__ == '__main__':
	logger = logging.getLogger('')
	logger.setLevel(logging.DEBUG)
	read_graph("graphs/facebook_combined_undirected.txt")
