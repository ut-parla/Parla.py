/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"

#include "BFS_SSSP.h"

#include <iostream>
#include <deque>
#include <type_traits>


enum Exec { SERIAL, PARALLEL };

enum Algo { AsyncTile = 0, Async, SyncTile, Sync, Sync2pTile, Sync2p };

const char* const ALGO_NAMES[] = {"AsyncTile", "Async",      "SyncTile",
                                  "Sync",      "Sync2pTile", "Sync2p"};

const int NSLOTS = 16;

galois::SharedMemSys* G = NULL;

struct NodeData {
    unsigned int distances[NSLOTS];
};

using Graph =
    galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type;
//::with_numa_alloc<true>::type;

using GNode = Graph::GraphNode;

constexpr static const bool TRACK_WORK          = false;
constexpr static const unsigned CHUNK_SIZE      = 256u;
constexpr static const ptrdiff_t EDGE_TILE_SIZE = 256;

using BFS = BFS_SSSP<Graph, unsigned int, false, EDGE_TILE_SIZE>;

using UpdateRequest       = BFS::UpdateRequest;
using Dist                = BFS::Dist;
using SrcEdgeTile         = BFS::SrcEdgeTile;
using SrcEdgeTileMaker    = BFS::SrcEdgeTileMaker;
using SrcEdgeTilePushWrap = BFS::SrcEdgeTilePushWrap;
using ReqPushWrap         = BFS::ReqPushWrap;
using OutEdgeRangeFn      = BFS::OutEdgeRangeFn;
using TileRangeFn         = BFS::TileRangeFn;

struct EdgeTile {
  Graph::edge_iterator beg;
  Graph::edge_iterator end;
};

struct EdgeTileMaker {
  EdgeTile operator()(Graph::edge_iterator beg,
                      Graph::edge_iterator end) const {
    return EdgeTile{beg, end};
  }
};

struct NodePushWrap {

  template <typename C>
  void operator()(C& cont, const GNode& n, const char* const _parallel) const {
    (*this)(cont, n);
  }

  template <typename C>
  void operator()(C& cont, const GNode& n) const {
    cont.push(n);
  }
};

struct EdgeTilePushWrap {
  Graph& graph;

  template <typename C>
  void operator()(C& cont, const GNode& n, const char* const _parallel) const {
    BFS::pushEdgeTilesParallel(cont, graph, n, EdgeTileMaker{});
  }

  template <typename C>
  void operator()(C& cont, const GNode& n) const {
    BFS::pushEdgeTiles(cont, graph, n, EdgeTileMaker{});
  }
};

struct OneTilePushWrap {
  Graph& graph;

  template <typename C>
  void operator()(C& cont, const GNode& n, const char* const _parallel) const {
    (*this)(cont, n);
  }

  template <typename C>
  void operator()(C& cont, const GNode& n) const {
    EdgeTile t{graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
               graph.edge_end(n, galois::MethodFlag::UNPROTECTED)};

    cont.push(t);
  }
};

template <bool CONCURRENT, typename T, typename P, typename R>
void asyncAlgo(Graph& graph, GNode source, const P& pushWrap,
               const R& edgeRange, int slot) {

  namespace gwl = galois::worklists;
  // typedef PerSocketChunkFIFO<CHUNK_SIZE> dFIFO;
  using FIFO = gwl::PerSocketChunkFIFO<CHUNK_SIZE>;
  using BSWL = gwl::BulkSynchronous<gwl::PerSocketChunkLIFO<CHUNK_SIZE>>;
  using WL   = FIFO;

  using Loop =
      typename std::conditional<CONCURRENT, galois::ForEach,
                                galois::WhileQ<galois::SerFIFO<T>>>::type;

  constexpr bool useCAS = CONCURRENT && !std::is_same<WL, BSWL>::value;

  Loop loop;

  galois::GAccumulator<size_t> BadWork;
  galois::GAccumulator<size_t> WLEmptyWork;

  graph.getData(source).distances[slot] = 0;
  galois::InsertBag<T> initBag;

  if (CONCURRENT) {
    pushWrap(initBag, source, 1, "parallel");
  } else {
    pushWrap(initBag, source, 1);
  }

  loop(galois::iterate(initBag),
       [&](const T& item, auto& ctx) {
         constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

         const auto& sdist = graph.getData(item.src, flag).distances[slot];

         if (TRACK_WORK) {
           if (item.dist != sdist) {
             WLEmptyWork += 1;
             return;
           }
         }

         const auto newDist = item.dist;

         for (auto ii : edgeRange(item)) {
           GNode dst   = graph.getEdgeDst(ii);
           auto& ddata = graph.getData(dst, flag).distances[slot];

           while (true) {

             Dist oldDist = ddata;

             if (oldDist <= newDist) {
               break;
             }

             if (!useCAS ||
                 __sync_bool_compare_and_swap(&ddata, oldDist, newDist)) {

               if (!useCAS) {
                 ddata = newDist;
               }

               if (TRACK_WORK) {
                 if (oldDist != BFS::DIST_INFINITY) {
                   BadWork += 1;
                 }
               }

               pushWrap(ctx, dst, newDist + 1);
               break;
             }
           }
         }
       },
       galois::wl<WL>(), galois::loopname("runBFS"), galois::no_conflicts());

  if (TRACK_WORK) {
    galois::runtime::reportStat_Single("BFS", "BadWork", BadWork.reduce());
    galois::runtime::reportStat_Single("BFS", "EmptyWork",
                                       WLEmptyWork.reduce());
  }
}

#define MARK() std::cout << "Galois: " << __FUNCTION__ << ":" << __LINE__ << " " << G << std::endl

template <bool CONCURRENT, typename T, typename P, typename R>
void syncAlgo(Graph& graph, GNode source, const P& pushWrap,
              const R& edgeRange, int slot) {

  using Cont = typename std::conditional<CONCURRENT, galois::InsertBag<T>,
                                         galois::SerStack<T>>::type;
  using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                         galois::StdForEach>::type;

  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

  Loop loop;

  Cont* curr = new Cont();
  Cont* next = new Cont();

  Dist nextLevel              = 0u;
  graph.getData(source, flag).distances[slot] = 0u;

    MARK();

  if (CONCURRENT) {
    pushWrap(*next, source, "parallel");
  } else {
    pushWrap(*next, source);
  }

    MARK();

  assert(!next->empty());

  while (!next->empty()) {

    std::swap(curr, next);
    next->clear();
    ++nextLevel;

      MARK();

      loop(galois::iterate(*curr),
         [&](const T& item) {
//             MARK();
           for (auto e : edgeRange(item)) {
             auto dst = graph.getEdgeDst(e);
             // if(dst == 13 || dst == 2 || dst == 51) std::cout<<" node " <<
             // dst << " visited"<<std::endl;
             auto& dstData = graph.getData(dst, flag).distances[slot];

             if (dstData == BFS::DIST_INFINITY) {
               dstData = nextLevel;
               pushWrap(*next, dst);
             }
           }
         },
         galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
         galois::loopname("Sync"));
  }

  delete curr;
  delete next;
}

template <bool CONCURRENT, typename P, typename R>
void sync2phaseAlgo(Graph& graph, GNode source, const P& pushWrap,
                    const R& edgeRange, int slot) {

  using NodeCont =
      typename std::conditional<CONCURRENT, galois::InsertBag<GNode>,
                                galois::SerStack<GNode>>::type;
  using TileCont =
      typename std::conditional<CONCURRENT, galois::InsertBag<EdgeTile>,
                                galois::SerStack<EdgeTile>>::type;

  using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                         galois::StdForEach>::type;

  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

  Loop loop;

  NodeCont activeNodes;
  TileCont edgeTiles;

  Dist nextLevel              = 0u;
  graph.getData(source, flag).distances[slot] = 0u;

  activeNodes.push(source);

  while (!activeNodes.empty()) {

    loop(galois::iterate(activeNodes),
         [&](const GNode& src) {
           pushWrap(edgeTiles, src);
         },
         galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
         galois::loopname("activeNodes"));

    ++nextLevel;
    activeNodes.clear();

    loop(galois::iterate(edgeTiles),
         [&](const EdgeTile& item) {
           for (auto e : edgeRange(item)) {
             auto dst      = graph.getEdgeDst(e);
             auto& dstData = graph.getData(dst, flag).distances[slot];

             if (dstData == BFS::DIST_INFINITY) {
               dstData = nextLevel;
               activeNodes.push(dst);
             }
           }
         },
         galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
         galois::loopname("edgeTiles"));

    edgeTiles.clear();
  }
}

template <bool CONCURRENT>
void runAlgo(Graph& graph, const GNode& source, Algo algo, int slot) {

  switch (algo) {
  case AsyncTile:
    asyncAlgo<CONCURRENT, SrcEdgeTile>(
        graph, source, SrcEdgeTilePushWrap{graph}, TileRangeFn(), slot);
    break;
  case Async:
    asyncAlgo<CONCURRENT, UpdateRequest>(graph, source, ReqPushWrap(),
                                         OutEdgeRangeFn{graph}, slot);
    break;
  case SyncTile:
    syncAlgo<CONCURRENT, EdgeTile>(graph, source, EdgeTilePushWrap{graph},
                                   TileRangeFn(), slot);
    break;
  case Sync:
    syncAlgo<CONCURRENT, GNode>(graph, source, NodePushWrap(),
                                OutEdgeRangeFn{graph}, slot);
    break;
  case Sync2pTile:
    sync2phaseAlgo<CONCURRENT>(graph, source, EdgeTilePushWrap{graph},
                               TileRangeFn(), slot);
    break;
  case Sync2p:
    sync2phaseAlgo<CONCURRENT>(graph, source, OneTilePushWrap{graph},
                               TileRangeFn(), slot);
    break;
  default:
    std::cerr << "ERROR: unkown algo type" << std::endl;
  }
}

void init_galois() {
    MARK();
    G = new galois::SharedMemSys();
}

Graph* load_file(const std::string& filename) {
    MARK();
  Graph* graph = new Graph();
  std::cout << "Reading from file: " << filename << std::endl;
  galois::graphs::readGraph(*graph, filename);
  std::cout << "Read " << graph->size() << " nodes, " << graph->sizeEdges()
            << " edges" << std::endl;
  return graph;
}

void bfs(Graph* pGraph, int iSource, int slot) {
  Algo algo = Sync;
  bool execution = true;
  Graph& graph = *pGraph;

  if (iSource >= graph.size()) {
    std::cerr << "failed to set source: " << iSource << "\n";
    abort();
  }

  auto it = graph.begin();
  std::advance(it, iSource);
  GNode source = *it;

//  size_t approxNodeData = 4 * (graph.size() + graph.sizeEdges());
//  // size_t approxEdgeData = graph.sizeEdges() * sizeof(typename
//  // Graph::edge_data_type) * 2;
//  galois::preAlloc(8 * numThreads +
//                   approxNodeData / galois::runtime::pagePoolSize());
//
//  galois::reportPageAlloc("MeminfoPre");

  galois::do_all(galois::iterate(graph),
                 [&](GNode n) { graph.getData(n).distances[slot] = BFS::DIST_INFINITY; });
  graph.getData(source).distances[slot] = 0;

  std::cout << "Running " << ALGO_NAMES[algo] << " algorithm with "
            << (bool(execution) ? "PARALLEL" : "SERIAL") << " execution "
            << std::endl;

  if (execution == SERIAL) {
    runAlgo<false>(graph, source, algo, slot);
  } else if (execution == PARALLEL) {
    runAlgo<true>(graph, source, algo, slot);
  } else {
    std::cerr << "ERROR: unknown type of execution passed to -exec"
              << std::endl;
    std::abort();
  }
}

unsigned int read_distance(Graph* pGraph, int node, int slot) {
    Graph& graph = *pGraph;
    auto it = graph.begin();
    std::advance(it, node);
    GNode n = *it;
    return graph.getData(n).distances[slot];
}