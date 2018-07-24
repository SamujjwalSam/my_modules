#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Tools for files
__description__ :
__project__     : my_modules
__author__      : 'Samujjwal Ghosh'
__version__     :
__date__        : June 2018
__copyright__   : "Copyright (c) 2018"
__license__     : "Python"; (Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html)

__classes__     :

__variables__   :

__methods__     :

TODO            : 1.
"""

import sys,platform
if platform.system() == 'Windows':
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research')
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research\Datasets')
else:
    sys.path.append('/home/cs16resch01001/codes')
    sys.path.append('/home/cs16resch01001/datasets')


import my_modules as mm

import networkx as nx
import scipy as sp

def nx2csr(G):
    """ Return the graph adjacency matrix as a SciPy sparse matrix.
Parameters:
G (graph) � The NetworkX graph used to construct the NumPy matrix.
nodelist (list, optional) � The rows and columns are ordered according to the nodes in nodelist. If nodelist is None, then the ordering is produced by G.nodes().
dtype (NumPy data-type, optional) � A valid NumPy dtype used to initialize the array. If None, then the NumPy default is used.
weight (string or None optional (default=�weight�)) � The edge attribute that holds the numerical value used for the edge weight. If None then all edge weights are 1.
format (str in {�bsr�, �csr�, �csc�, �coo�, �lil�, �dia�, �dok�}) � The type of the matrix to be returned (default �csr�). For some algorithms different implementations of sparse matrices can perform better. See [1] for details.
Returns: M � Graph adjacency matrix.

Return type: SciPy sparse matrix
"""

    A = nx.to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight='weight', format='csr')

def csr2nx(G):
    """Creates a new graph from an adjacency matrix given as a SciPy sparse matrix.
Parameters:
A (scipy sparse matrix) � An adjacency matrix representation of a graph
parallel_edges (Boolean) � If this is True, create_using is a multigraph, and A is an integer matrix, then entry (i, j) in the matrix is interpreted as the number of parallel edges joining vertices i and j in the graph. If it is False, then the entries in the adjacency matrix are interpreted as the weight of a single edge joining the vertices.
create_using (NetworkX graph) � Use specified graph for result. The default is Graph()
edge_attribute (string) � Name of edge attribute to store matrix numeric value. The data will have the same type as the matrix entry (int, float, (real,imag)).
"""
    A = nx.from_scipy_sparse_matrix(A, parallel_edges=False, create_using=None, edge_attribute='weight')

def main():
    pass


if __name__ == "__main__": main()
