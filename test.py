
import cudf
import cugraph

gdf = cudf.read_csv("/home/ubuntu/software/cugraph/datasets/karate.csv", header=None, sep=' ')
gdf.columns = ('src', 'dst', 'weight')

G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='src', destination='dst')

z = cugraph.link_analysis.hits(G)
print(z)

