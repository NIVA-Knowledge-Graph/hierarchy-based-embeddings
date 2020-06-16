
"""
Input Required

1. A csv file containing hierarchy in the form of two columns: parent node and child node or (subject object)
2. Root node of the hierarchy

Output: Dataframe, each row represnts embeddings for a uniqe value.

Steps for calculating semantic embeddings

Load the hierarchy from CSV into the networkx Graph
Convert the graph into tree by specifying the root node of the hierarchy
Create an attribute "node-level" for each node to store the level of the node
Create unique paris for all nodes
For each pair(i,j), find it's lowest common ancestor and replace it with it's level
Create adjancey matrix where row and columns represents all nodes and each element(i,j) represents level of lowest common ancestor
Use any similarity function in range (0,1) to calculate similarity (here we calculate similarity using our proposed measure: hierarchy-based semantic similarity)
"""

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

from utils import create_embeddings

root_dict = { 'Inorganic_': 'http://id.nlm.nih.gov/mesh/D007287',
                'Organic_chemicals' : 'http://id.nlm.nih.gov/mesh/D009930',
               'Hetrocyclic_compounds' :'http://id.nlm.nih.gov/mesh/D006571',
                'Polycyclic_compounds':        'http://id.nlm.nih.gov/mesh/D011083',
                'Macromolecular_substances':    'http://id.nlm.nih.gov/mesh/D046911',
                'Hormones_Hormone_Substitutes_Hormone_Antagonists':   'http://id.nlm.nih.gov/mesh/D006730',
                'Enzymes_and_Coenzymes' :        'http://id.nlm.nih.gov/mesh/D045762',
                 'Carbohydrates':      'http://id.nlm.nih.gov/mesh/D002241',
                  'Lipids':    'http://id.nlm.nih.gov/mesh/D008055',
                 'Amino_Acids_Peptides_Proteins' :   'http://id.nlm.nih.gov/mesh/D000602',
                   'Nucleic_Acids_Nucleotides_Nucleosides':   'http://id.nlm.nih.gov/mesh/D009706',
                  'Complex_Mixtures':  'http://id.nlm.nih.gov/mesh/D045424',
                   'Biological_Factors' : 'http://id.nlm.nih.gov/mesh/D001685',
                    'biomedical_Dental_Materials':'http://id.nlm.nih.gov/mesh/D001697',
                   'Pharmaceutical_Preparations': 'http://id.nlm.nih.gov/mesh/D004364',
                    'Chemical_Actions_Uses' : 'http://id.nlm.nih.gov/mesh/D020164',
                    'taxonomy': 'https://www.ncbi.nlm.nih.gov/taxonomy/taxon/1'}


# fetch all csv files that contains hierarchy pair (subject object pair)
filepath = os.getcwd()
file_list = glob.glob(filepath +'/hierarchies/*.csv')

# 0 < lambda_factor < 1
#
lambda_factor = 0.7
for key, root_node in root_dict.items():
    if key in str(file_list):
        filename = [i for i in file_list if key in i]
        for file in filename:
            print ('Processing file ..',file.rpartition("/")[2] )
            create_embeddings(file,  root_node , './Embeddings/all_nodes_'+file.rpartition("/")[2], lambda_factor)
