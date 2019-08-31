import igraph
import numpy as np

def read_undirected_graph(f):
    with open(f, 'r') as fin:
        G = igraph.Graph()
        V = set()
        E = []

        for line in fin:
            if line.startswith('#'):
                continue
            v1, v2 = line.strip().split() # vertex names are strings in igraph
            V.add(v1)
            V.add(v2)
            E.append((v1, v2))
            # E.append((v2, v1))

        G.add_vertices(list(V)) # e.g., vertex 1912 will have "name" '1912' (a string!) 
        G.add_edges(E)

        # to check that all edges from the file made it into G
        # for e in G.es:
        #     print(G.vs[e.source]['name'], G.vs[e.target]['name'])
        return G

def draw_communities(G, VC, outname, bbox=(0, 0, 1400, 1400)):
    # layout = G.layout_fruchterman_reingold(maxiter=150)
    layout = G.layout_lgl(maxiter=300)
    igraph.drawing.plot(obj=VC, vertex_size=8, layout=layout, bbox=bbox, keep_aspect_ratio=True, target=outname + ".pdf")

def document_VC(VC):
    print("-->", "comm. detection, final modularity:", VC.modularity)
    print("-->", "total num. communities of all sizes:", len(VC.subgraphs()))
    comm_sizes = []
    for C in VC.subgraphs():
        comm_sizes.append(len(C.vs))
    print("-->", "community sizes, mean and sample stdev:", np.mean(comm_sizes), np.std(comm_sizes, ddof=1))
    print("-->", "list of community sizes:", sorted(comm_sizes, reverse=True))

if __name__ == '__main__':
    G = read_undirected_graph("../SN/twitter_combined.txt")
    print(G.summary())

    # compute final communities
    VC = G.community_multilevel(return_levels=False)
    document_VC(VC)

    draw_communities(G, VC, "twitter_communities")

    # show intermediate communities    
    VCL = G.community_multilevel(return_levels=True) # a list of VCs
    for (VC, i) in zip(VCL, range(len(VCL))):
        print("Level", i)
        document_VC(VC)
