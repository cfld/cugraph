# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.traversal.gunrock_sssp cimport gunrock_sssp as c_gunrock_sssp
from cugraph.structure.graph_primtypes cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from cugraph.structure import graph_primtypes_wrapper
import cudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib


def gunrock_sssp(input_graph, single_source=0):
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    # Prepare graph
    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32])

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]

    n_nodes = input_graph.number_of_vertices()
    n_edges = input_graph.number_of_edges(directed_edges=True)

    cdef GraphCSRView[int,int,float] graph_float
    graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>c_weights, n_nodes, n_edges)
    
    # Prepare output
    df = cudf.DataFrame()
    df['vertex']       = cudf.Series(np.zeros(n_nodes, dtype=np.int32))
    df['distances']    = cudf.Series(np.zeros(n_nodes, dtype=np.float32))
    df['predecessors'] = cudf.Series(np.zeros(n_nodes, dtype=np.int32))

    cdef uintptr_t c_identifier   = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_distances    = df['distances'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_predecessors = df['predecessors'].__cuda_array_interface__['data'][0];

    # Run
    c_gunrock_sssp[int,int,float](
        graph_float, 
        single_source, 
        <float*>c_distances, 
        <int*>c_predecessors
    );
    graph_float.get_vertex_identifiers(<int*>c_identifier)

    return df