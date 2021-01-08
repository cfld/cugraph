#include <algorithms.hpp>
#include <graph.hpp>

#include <utilities/error.hpp>

#include <gunrock/applications/sssp.hxx>

namespace cugraph {

namespace gunrock {

namespace E = ::gunrock; // gunrock/essentials

template <typename vertex_t, typename edge_t, typename weight_t>
void gunrock_sssp(cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
          vertex_t single_source,
          weight_t *distances,
          vertex_t *predecessors)
{  
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "Invalid API parameter: graph.edge_data should be of size V");
  CUGRAPH_EXPECTS(distances != nullptr, "Invalid API parameter: distances array should be of size V");
  CUGRAPH_EXPECTS(predecessors != nullptr, "Invalid API parameter: predecessors array should be of size V");

  auto G = E::graph::build::from_csr<E::memory::memory_space_t::device, E::graph::view_t::csr>(
    graph.number_of_vertices,
    graph.number_of_vertices,
    graph.number_of_edges,
    graph.offsets,
    graph.indices,
    graph.edge_data
  );

  float elapsed = E::sssp::run(
    G, single_source, distances, predecessors
  );
}

template void gunrock_sssp(cugraph::GraphCSRView<int32_t, int32_t, float> const &,
                   int32_t,
                   float *,
                   int32_t *);

}  // namespace gunrock

}  // namespace cugraph