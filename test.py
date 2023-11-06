#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from play_ground.min_cost_problem_graph import minimum_cost_flow_problem_graph
from play_ground.solve_min_cost_graph import solve_min_cost_flow_graph

import warnings
import scipy.sparse as sp
from joblib import Parallel
from joblib import delayed
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow

def test_minimum_cost_flow_problem_graph():
    # Setup graph
    X = np.array([
        [0, 0],
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0],
        [4, 4]
    ])
    
    A = [1,1,1,1,1,5,0,0]
    
    C = np.array([
        [0, 0],
        [4, 4]
    ])
    
    
    cluster_sizes = [5,4]

    D = euclidean_distances(X, C, squared=True)

    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(X, A, C, D, cluster_sizes)

    assert edges.shape[0] == len(costs)
    assert edges.shape[0] == len(capacities)
    assert len(np.unique(edges)) == len(supplies)
    assert costs.sum() > 0
    assert supplies.sum() == 0
    

def test_solve_min_cost_flow_graph():
    # Setup graph
    X = np.array([
        [0, 0],
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0],
        [4, 4]
    ])
    
    A = [1,1,1,1,1,5,0,0]
    
    C = np.array([
        [0, 0],
        [4, 4]
    ])
    
    
    cluster_sizes = [5,5]

    D = euclidean_distances(X, C, squared=True)

    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(X, A, C, D, cluster_sizes)
    labels = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

    cluster_size = pd.Series(labels).value_counts()

    assert (cluster_size > size_max).sum() == 0
    assert (cluster_size < size_min).sum() == 0
    
if __name__ == "__main__":
    # test_minimum_cost_flow_problem_graph()
    test_solve_min_cost_flow_graph()
