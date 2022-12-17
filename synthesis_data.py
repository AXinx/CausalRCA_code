#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:47:53 2022

@author: ruyuexin
"""
import numpy as np
import networkx as nx
from typing import Callable
from sklearn.preprocessing import StandardScaler
import pickle as pkl
#from graph import Node
#from graph import MemoryGraph

_SLI = 0

def _uniform_weight(rng: np.random.Generator) -> float:
    segments = [(-2.0, -0.5), (0.5, 2.0)]
    low, high = rng.choice(segments)
    return rng.uniform(low=low, high=high)


def generate_sedag(
    num_node: int,
    num_edge: int,
    weight_generator: Callable[[np.random.Generator], float] = _uniform_weight,
    rng: np.random.Generator = None):
    """
    Generate a weighted directed acyclic graph with a single end.

    The first node with index of 0 is the only end that does not have results.

    Returns:
        a matrix, where matrix[i, j] != 0 means j is the cause of i
    """
    num_edge = min(max(num_edge, num_node - 1), int(num_node * (num_node - 1) / 2))
    if rng is None:
        rng = np.random.default_rng()
    matrix = np.zeros((num_node, num_node))
    # Make the graph connected
    for cause in range(1, num_node):
        result = rng.integers(low=0, high=cause)
        matrix[result, cause] = weight_generator(rng)
    num_edge -= num_node - 1
    while num_edge > 0:
        cause = rng.integers(low=1, high=num_node)
        result = rng.integers(low=0, high=cause) 
        if not matrix[result, cause]: 
            matrix[result, cause] = weight_generator(rng) 
            num_edge -= 1 
    return matrix 


def generate_case(
    weight: np.ndarray,
    length_normal: int = 100, #1440
    fault_duration: int = 1, #fault duration 
    length_abnormal: int = 100, #10
    beta: float = 1e-1,
    tau: float = 3,
    sigmas: np.ndarray = None,
    fault: np.ndarray = None,
    rng: np.random.Generator = None):
#) -> SimCase:
    # pylint: disable=too-many-arguments, too-many-locals
    """
    Generate a case

    Parameters:
        weight: The inversed matrix of I - A
    """
    if rng is None:
        rng = np.random.default_rng()
    length_abnormal = max(length_abnormal, fault_duration) #10，2

    num_node, _ = weight.shape
    data= np.zeros((0, num_node))
    if sigmas is None:
        sigmas = rng.standard_exponential(num_node)

    values = rng.standard_normal(num_node) * sigmas
    
    # Generate a series of x with x^{(t)} = A x^{(t)} + x^{(t - 1)} + epsilon^{(t)}
    # or x^{(t)} = A^{(\prime)} (x^{(t - 1)} + epsilon^{(t)})
    # where A^{(\prime)} = \sum_{i=0}^{num_node} A^{i}
    for _ in range(length_normal):
        values = weight @ (beta * values + rng.standard_normal(num_node) * sigmas)
        data = np.append(data, [values], axis=0)
    sli_mean = data[:, _SLI].mean() #first column
    sli_sigma = data[:, _SLI].std()
    # Inject a fault
    if fault is None:
        num_causes = 1 #min(rng.poisson(1) + 1, num_node) 
        causes = rng.choice(num_node, size=num_causes, replace=False) #from num_node select num_causes
        print(causes)
        fault = np.zeros(num_node)
        alpha = rng.standard_exponential(size=num_causes)
        epsilon = rng.standard_normal(num_node)
        while True: #choose a better fault value
            fault[causes] = alpha
            sli_value: float = np.dot(
                weight[_SLI, :], beta * values + (epsilon + fault) * sigmas
            )
            if abs(sli_value - sli_mean) > tau * sli_sigma:
                break
            alpha *= 2
    else:
        causes: np.ndarray = np.where(fault)[0]
        assert causes.size

    # Faulty data
    for _ in range(fault_duration):
        values = weight @ (
            beta * values + (rng.standard_normal(num_node) + fault) * sigmas
        )
        data = np.append(data, [values], axis=0) #add fault data
    for _ in range(length_abnormal - fault_duration): #fault data propagate serveral steps
        values = weight @ (beta * values + rng.standard_normal(num_node) * sigmas)
        data = np.append(data, [values], axis=0)

    scaler = StandardScaler().fit(data[:length_normal, :])
    data = np.around(scaler.transform(data), decimals=3)

    details = dict(
        fault=fault,
        sigmas=sigmas,
        stds=scaler.scale_,
        weight=weight,
    )
    return data, causes.tolist(), length_normal, details
    
    # return SimCase(
    #     data=data,
    #     causes=set(causes.tolist()),
    #     length_normal=length_normal,
    #     details=details,
    # )

num_node = 50
num_edge = 100
num_cases: int = 1
rng = None #np.random.Generator
if rng is None:
    rng = np.random.default_rng()

# A = Generate weighted DAG
matrix = generate_sedag(num_node=num_node, num_edge=num_edge, rng=rng)
print(matrix)
prod = np.eye(num_node)
weight = np.eye(num_node)  # The reversed matrix of I - A
for _ in range(1, num_node):
    prod = prod @ matrix #p = m^n
    weight += prod #w = m^n + m^(n-1) + ... + m^1 + m^0(I)
print('*****')
#print(weight)
#print(rng)
for _ in range(num_cases):
    cases = [generate_case(weight=weight, rng=rng)]

graph = nx.from_numpy_matrix(matrix, parallel_edges=True, create_using=nx.DiGraph)
nx.draw(graph, with_labels=True)

with open('./data_synthesis/synthesis_data.pkl', 'wb') as syn_d:
    pkl.dump(cases, syn_d)
with open('./data_synthesis/synthesis_graph.pkl', 'wb') as syn_g:
    pkl.dump(graph, syn_g)
    
#return cases, MemoryGraph(graph)


# rng = np.random.default_rng()
# num_node = 5
# num_edge = 5
# # A = Generate weighted DAG
# matrix = generate_sedag(num_node=num_node, num_edge=num_edge, rng=rng)
# print(matrix)
# prod = np.eye(num_node)
# weight = np.eye(num_node)  # The reversed matrix of I - A
# for _ in range(1, num_node):
#     prod = prod @ matrix #p = m^n
#     weight += prod #w = m^n + m^(n-1) + ... + m^1 + m^0(I)
# print('*****')
# print(weight)

# res = generate_case(weight)

# def generate(
#     num_node: int, num_edge: int, num_cases: int = 5, rng: np.random.Generator = None):
#     """
#     Generate a dataset with the same graph and serveral cases
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     # A = Generate weighted DAG
#     matrix = generate_sedag(num_node=num_node, num_edge=num_edge, rng=rng)
#     print(matrix)
#     prod = np.eye(num_node)
#     weight = np.eye(num_node)  # The reversed matrix of I - A
#     for _ in range(1, num_node):
#         prod = prod @ matrix #p = m^n
#         weight += prod #w = m^n + m^(n-1) + ... + m^1 + m^0(I)
#     print('*****')
#     print(weight)
#     print(rng)
#     cases = [generate_case(weight=weight, rng=rng) for _ in range(num_cases)]

#     # graph = nx.DiGraph(
#     #     (
#     #         (
#     #             Node(entity=ENTITY, metric=str(cause)),
#     #             Node(entity=ENTITY, metric=str(result)),
#     #         )
#     #         for result, cause in zip(*np.where(matrix))
#     #     )
#     # )
#     graph = nx.from_numpy_matrix(matrix, parallel_edges=True, create_using=nx.DiGraph)
#     nx.draw(graph, with_labels=True)
#     return cases, MemoryGraph(graph)
#     #return SimDataset(cases=cases, graph=MemoryGraph(graph))
    
# a,b = generate(5,5)

