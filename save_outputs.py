import numpy as np
import cvxopt.base
import cvxopt.solvers
import itertools
import networkx as nx
import cvxpy as cp
import numpy as np
import random
from itertools import combinations
from lovasz import lovasz_theta
import pandas as pd
import math
from scipy.special import comb  # To calculate n choose k
import time




def generate_nodes(n, p):
    # generates all possible p-combinations of list of {1,...,n}
    return list(itertools.combinations(range(1, n + 1), p))

def construct_graph(n, p):
    nodes = generate_nodes(n, p)
    G = nx.Graph()
    
    # Add nodes to the graph
    for i, node in enumerate(nodes):
        G.add_node(i, subset=node)
    
    # Add edges to the graph
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], start=i+1):
            shared_integers = set(node1) & set(node2)
            if len(shared_integers) % 2 == 1:
                G.add_edge(i, j)
    
    return G


def construct_random_graph(n, k, N):
    """
    Construct a graph with N unique random k-subsets of {1, ..., n} and
    edges exist between nodes if the intersection of their subsets is odd.
    """
    G = nx.Graph()
    all_elements = list(range(1, n + 1))
    
    # Generate N unique random k-subsets of {1, ..., n}
    unique_nodes = set()
    while len(unique_nodes) < N:
        new_node = tuple(sorted(random.sample(all_elements, k)))
        unique_nodes.add(new_node)
    
    nodes = list(unique_nodes)
    print(len(nodes))
    
    # Add nodes to the graph
    for i, node in enumerate(nodes):
        G.add_node(i, subset=node)
    
    # Add edges to the graph
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], start=i+1):
            shared_integers = set(node1) & set(node2)
            if len(shared_integers) % 2 == 1:
                G.add_edge(i, j)
    
    return G


def output(n, k, N):
    # Calculates Lovasz number of given inputs
    G = construct_random_graph(n, k, N)
    theta = lovasz_theta(G)
    return theta

# Initialize lists to store the results
results = []

# Loop over values of n and k
for _ in range(10):
    for n in [18, 22, 26, 30]:
        for k in [4]:

            # Calculate the combination n choose k
            nCk = comb(n, k)
            # Generate values of N from, logarithmically spaced
            N_values = np.logspace(math.log10(nCk * 3e-4), math.log10(nCk * 2e-1), num=20)
            
            # Check that N is in range where code runs fast enough
            N_values = [int(N) for N in N_values if N >= 20 and N <= 350]
            
            # Loop over these N values and compute the output
            for N in N_values:
                # time start
                start = time.time()

                # Compute the output
                result = output(n, k, N)

                # time end
                end = time.time()

                # Store the result along with the values of n, k, and N
                results.append([n, k, N, result, end-start])

                # Create a DataFrame from the results
                df = pd.DataFrame(results, columns=['n', 'k', 'N', 'output', 'time'])

                # Save the DataFrame to a CSV file
                df.to_csv('results/output_results_agg.csv', index=False)



print("Results saved.")
