/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * ---------------------------------------------------------------------------*
 * @brief wrapper calling gunrock's HITS analytic
 * --------------------------------------------------------------------------*/

#include <algorithms.hpp>
#include <graph.hpp>

#include <utilities/error.hpp>

#include <gunrock/gunrock.h>
#include <gunrock/applications/sssp/sssp_implementation.hxx>

namespace cugraph {

namespace gunrock {

// using namespace ::gunrock;
// using namespace ::gunrock::memory;

template <typename vertex_t, typename edge_t, typename weight_t>
void gunrock_sssp(cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
          vertex_t single_source,
          weight_t *distances,
          vertex_t *predecessors)
{  
  CUGRAPH_EXPECTS(distances != nullptr, "Invalid API parameter: distances array should be of size V");
  CUGRAPH_EXPECTS(predecessors != nullptr, "Invalid API parameter: predecessors array should be of size V");

  auto G = ::gunrock::graph::build::from_csr_t<::gunrock::memory::memory_space_t::device>(
    graph.number_of_vertices,
    graph.number_of_vertices,
    graph.number_of_edges,
    graph.offsets,
    graph.indices,
    graph.edge_data
  );

  auto meta = ::gunrock::graph::build::meta_t<vertex_t, edge_t, weight_t>(
    graph.number_of_vertices,
    graph.number_of_vertices,
    graph.number_of_edges
  );

  float elapsed = ::gunrock::sssp::run(
    G,
    meta,
    single_source,
    distances,
    predecessors
  );
}

template void gunrock_sssp(cugraph::GraphCSRView<int32_t, int32_t, float> const &,
                   int32_t,
                   float *,
                   int32_t *);

}  // namespace gunrock

}  // namespace cugraph