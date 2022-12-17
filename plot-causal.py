#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:39:50 2022

@author: ruyuexin
"""
import pandas as pd
import numpy as np
import networkx as nx
import graphviz
import matplotlib.pyplot as plt

from graphviz import Digraph

# dot = Digraph(name="Latency-causal", comment="the test", format="pdf")
# dot.attr('graph', shape='ellipse')

# dot.node(name='0', label='Front-end')
# dot.node(name='1', label='User')
# dot.node(name='2', label='Catalogue')
# dot.node(name='3', label='Orders',color='red')
# dot.node(name='4', label='Carts')
# dot.node(name='5', label='Payment')
# dot.node(name='6', label='Shipping')

# #dot.edge('a', 'b', label="ab\na-b")
# dot.edges(['01', '04'])
# dot.edges(['14', '12'])
# dot.edges(['24', '25'])
# dot.edges(['34', '30'])
# dot.edges(['56'])
# dot.edges(['63'])
# #print(dot.source)

# dot.view(filename="latency-causal1", directory="./")

#CPU hog 
# adj = [[ 0, 0.35656879, 0, 0, 0.52349655, 0, 0],
#        [0, 0, 0.38960124, 0,-0.82802432, 0, 0],
#        [0, 0, 0, 0,-1.37088886, 0.43533847, 0],
#        [ 0.35736854, 0, 0, 0, 1.55882351, 0, 0],
#        [ 0, 0, 0, 0, 0, 0, 0],
#        [ 0, 0, 0, 0, 0, 0, 0.43354373],
#        [ 0, 0, 0, 0.31099542,  0, 0, 0]]
# order:[(3, 0.1706901986333939), (2, 0.16792817429208914), (6, 0.16348391365993775), (1, 0.15898591907760504), (0, 0.15879379167103705), (5, 0.15868943123736554), (4, 0.02142857142857143)]

# Memory leak
# adj = [[ 0, 0, 0, 0,-3.53118238, -0.53664182, -1.05381845],
#         [ 0, 0, 0, 0, 1.37671733, 0.35987646, -0.33310958],
#         [ 0, 0, 0, 0, 0.38208851, -0.73689545, -0.70728362],
#         [ 0, 0, 0, 0, 2.20324368, 0.49664891, 2.13268007],
#         [ 0, 0, 0, 0, 0, 0.30660962, 0],
#         [ 0, 0, 0, 0, 0, 0, 0],
#         [ 0, 0, 0, 0,-0.38129383, 0, 0]]
#order: [(3, 0.23677700595710105), (0, 0.23458748727360365), (2, 0.2181375665419254), (1, 0.21445858292561665), (4, 0.03432590949570191), (6, 0.03184146025583939), (5, 0.029871987550211955)]

# Network delay
adj = [[ 0, 0,-0.49381467, 0.50704032, 0.84738591, 0, -2.99303902],
        [-0.81469407, 0, 0,-0.49817637, 0, 0, -0.83116868],
        [ 0, 0, 0, 0, 0, 0, 0.50311925],
        [ 0, 0, 0, 0,-1.5951983, 0, 2.24059324],
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0.58793024, -0.65266837, 0, 0, 0.8461652, 0, 0.67365238],
        [ 0, 0, 0, 0, 0, 0, 0]]
#payment: [(5, 0.5208429050050171), (1, 0.17796152357794767), (0, 0.14088911247401992), (3, 0.06293880910176598), (2, 0.03428643828518632), (4, 0.03154060577803149), (6, 0.03154060577803149)]

names = ['Front-end', 'User', 'Catalogue', 'Orders', 'Carts', 'Payment', 'Shipping']
#n_colors = ['White', 'White', 'White', 'Red', 'White', 'White', 'White']
n_colors = ['White', 'White', 'White', 'White', 'White', 'Red', 'White']

options = {
    "font_size": 11, 
    "node_size": 3500, 
    "node_color": n_colors, 
    "edgecolors": "black", 
    "linewidths": 1, 
    #"width": 5, 
}

adj_name = pd.DataFrame(adj, index=names, columns=names) 
G = nx.from_pandas_adjacency(adj_name, create_using=nx.DiGraph) 
edges = G.edges() 
e_colors = [] 
e_width = [] 
for u,v in edges: 
    e_width.append(G[u][v]['weight']) 
    if G[u][v]['weight'] > 0: 
        e_colors.append('b') 
    if G[u][v]['weight'] < 0: 
        e_colors.append('g') 
plt.subplots(figsize=(12,8))
pos=nx.shell_layout(G)
nx.draw(G, pos=pos, with_labels=True, edge_color=e_colors, width=np.abs(e_width), **options) 
plt.savefig('latency-causal-network.pdf')


