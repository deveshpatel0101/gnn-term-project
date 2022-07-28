
import networkx as nx


def visualize(edges_u, edges_v):
    G = nx.Graph()
    G.add_nodes_from(set(edges_u + edges_v))
    G.add_edges_from(zip(edges_u, edges_v))

    nx.draw(G, node_size=0)
