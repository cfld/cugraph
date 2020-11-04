from cugraph.traversal import gunrock_sssp_wrapper

def gunrock_sssp(G, single_source=0):
    # !! More transformations could be done here ...
    return gunrock_sssp_wrapper.gunrock_sssp(G, single_source)