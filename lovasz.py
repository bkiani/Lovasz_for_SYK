import numpy as np
import cvxopt.base
import cvxopt.solvers
import itertools
import networkx as nx
import cvxpy as cp
import numpy as np


def parse_graph(G, complement=False):
    '''
    Takes a Sage graph, networkx graph, or adjacency matrix as argument, and returns
    vertex count and edge list for the graph and its complement.
    '''

    if type(G).__module__+'.'+type(G).__name__ == 'networkx.classes.graph.Graph':
        import networkx
        G = networkx.convert_node_labels_to_integers(G)
        nv = len(G)
        edges = [ (i,j) for (i,j) in G.edges() if i != j ]
        c_edges = [ (i,j) for (i,j) in networkx.complement(G).edges() if i != j ]
    else:
        if type(G).__module__+'.'+type(G).__name__ == 'sage.graphs.graph.Graph':
            G = G.adjacency_matrix().numpy()

        G = np.array(G)

        nv = G.shape[0]
        assert len(G.shape) == 2 and G.shape[1] == nv
        assert np.all(G == G.T)

        edges   = [ (j,i) for i in range(nv) for j in range(i) if G[i,j] ]
        c_edges = [ (j,i) for i in range(nv) for j in range(i) if not G[i,j] ]

    for (i,j) in edges:
        assert i < j
    for (i,j) in c_edges:
        assert i < j

    if complement:
        (edges, c_edges) = (c_edges, edges)

    return (nv, edges, c_edges)



def construct_graph(n, k):
    """
    Construct a graph where each node is a k-subset of {1, ..., n} and
    edges exist between nodes if the intersection of their subsets is odd.
    """
    G = nx.Graph()

    # Generate all k-subsets of {1, ..., n}
    nodes = list(itertools.combinations(range(1, n+1), k))

    # Add nodes to the graph
    G.add_nodes_from(nodes)

    # Add edges based on the intersection condition
    for i, subset1 in enumerate(nodes):
        for j, subset2 in enumerate(nodes):
            if i < j and len(set(subset1).intersection(set(subset2))) % 2 == 0:
                G.add_edge(subset1, subset2)
                G.add_edge(subset2, subset1)

    return G



def lovasz_theta(G):
    n = len(G.nodes)
    Gc = nx.complement(G)

    X = cp.Variable((n, n), symmetric=True)
    constraints = []
    constraints += [X[i, i] == 1 for i in range(n)]
    constraints += [X[i, j] == 1 for i, j in Gc.edges]

    prob = cp.Problem(cp.Minimize(cp.lambda_max(X)), constraints)

    prob.solve(solver='MOSEK')  # or use 'SCS', 'MOSEK', etc.

    return prob.value


def subset_graph(G, n):
    G = G.copy()
    choices = np.random.choice(len(G.nodes()), size=n, replace=False)
    nodes = list(G.nodes())
    choices = [nodes[i] for i in choices]
    return G.subgraph(choices)



if __name__ == "__main__":
    
    outs = []
    for n in range(4,7):
        print(n)
        G0 = construct_graph(n, 2)
        N = len(G0.nodes())
        print(len(G0.nodes()))
        theta = lovasz_theta(G0)
        print(f"The Lovász number of the graph for n={n} and k={2} is: {theta}")
        outs.append(theta)
