# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.structure.graph_primtypes cimport *
from libcpp cimport bool

cdef extern from "algorithms.hpp" namespace "cugraph::gunrock":

    cdef void gunrock_sssp[VT,ET,WT](
        const GraphCSRView[VT,ET,WT] &graph,
        int single_source,
        WT *distances,
        VT *predecessors) except +