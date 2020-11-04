import sys
from cugraph.traversal import gunrock_sssp_wrapper
from cugraph.utilities import check_nx_graph
from cugraph.utilities import df_score_to_dictionary

def gunrock_sssp(G, single_source=0):
    G, isNx = check_nx_graph(G)

    df = gunrock_sssp_wrapper.gunrock_sssp(G, max_iter, tol)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        d1 = df_score_to_dictionary(df[["vertex", "distances"]], "distances")
        d2 = df_score_to_dictionary(df[["vertex", "predecessors"]], "predecessors")
        df = (d1, d2)

    return df