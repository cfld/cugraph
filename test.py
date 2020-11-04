
import cudf
import cugraph
import numpy as np
import pandas as pd
from scipy.io import mmread

coo = mmread('/home/ubuntu/projects/essentials/examples/sssp/chesapeake.mtx')

src, dst = coo.nonzero()
weight   = np.ones(src.shape[0]).astype(np.float32)

df  = pd.DataFrame({'src' : src, 'dst' : dst, 'weight' : weight})
cdf = cudf.DataFrame(df)

G = cugraph.Graph()
G.from_cudf_edgelist(cdf, source='src', destination='dst', edge_attr='weight')

z = cugraph.traversal.gunrock_sssp(G, 0)
z = z.sort_values('vertex').reset_index(drop=True)
print(z)

