# filename : contains input filename with full path 
# file should have at least two columns named 'subject' and 'object'
# rootnode: specify the url for root node of the hierarchy
# target_filename: save embeddings with key name
# lambda_factor : tuneable factor to create embeddings, read http://ceur-ws.org/Vol-2600/paper16.pdf


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timeit
import csv
from sklearn import tree, linear_model
import matplotlib.pyplot as plt
import math
import itertools
from multiprocessing import Pool, cpu_count
from anytree import Node, RenderTree
import os
import networkx as nx
import glob


def create_embeddings(filename, rootnode, target_filename, lambda_factor=0.6):
    df = pd.read_csv(filename)
    # Create the Directed Graph 
    try:
        G = nx.from_pandas_edgelist(df,
                            source='parent',
                            target='child',
                            create_using=nx.DiGraph())
    except KeyError:
        G = nx.from_pandas_edgelist(df,
                            source='object',
                            target='subject',
                            create_using=nx.DiGraph())
    # create tree by specifying root node
    tree = nx.bfs_tree(G, rootnode) #
    # find level of node(shortest path from root to current node)
    optional_attrs = nx.shortest_path_length(tree ,rootnode)
    nx.set_node_attributes(tree ,  optional_attrs, 'node_level' )
    
    ls_leafnodes = [node for node in tree.nodes()]
    pairs = list(itertools.product(ls_leafnodes, repeat=2)) # create pair of all nodes 
    all_ancestors = nx.algorithms.all_pairs_lowest_common_ancestor(tree, pairs=pairs) # get lowest common ancestors of alll pairs of nodes


    # replace ancestor node with its level in the hierarchy
    ls_ancestors_levels = {}
    for i in all_ancestors:
        ls_ancestors_levels[i[0]] = tree.node[i[1]]['node_level'] 
        
    chunked_data = [[k[0],k[1], v] for k, v in ls_ancestors_levels.items()]
    df_nodes = pd.DataFrame(chunked_data)
    df_nodes = df_nodes.rename(columns= {0:'node1', 1:'node2', 2:'weight'})
    depth = df_nodes.weight.max() # find the maximum levels in the hierarchy

    # create adjancey matrix
    vals = np.unique(df_nodes[['node1', 'node2']])
    df_nodes = df_nodes.pivot(index='node1', columns='node2', values='weight'
                      ).reindex(columns=vals, index=vals, fill_value=0)

    df_adjacency = df_nodes.apply( lambda x:  np.power(  lambda_factor, depth - x))

    # set diagnoal to 1
    pd.DataFrame.set_diag = set_diag
    df_adjacency.set_diag(1)
    df_adjacency.fillna(0, inplace=True)

    df_adjacency.to_csv(target_filename)
    
def set_diag(self, values): 
    n = min(len(self.index), len(self.columns))
    self.values[tuple([np.arange(n)] * 2)] = values
