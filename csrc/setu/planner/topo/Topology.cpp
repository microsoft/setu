#include "planner/topo/Topology.h"

namespace setu::planner::topo {

void Topology::AddLink(const Participant& src, const Participant& dst,
                       Link link) {
  adj_[src].push_back({dst, link});
  if (!adj_.contains(dst)) {
    adj_[dst] = {};
  }
}

void Topology::AddBidirectionalLink(const Participant& src,
                                    const Participant& dst, Link link) {
  AddLink(src, dst, link);
  AddLink(dst, src, link);
}

std::vector<Link> Topology::QueryLinks(const Participant& src,
                                       const Participant& dst) const {
  std::vector<Link> result;
  auto it = adj_.find(src);
  if (it == adj_.end()) {
    return result;
  }
  for (const auto& [v, link] : it->second) {
    if (v == dst) {
      result.push_back(link);
    }
  }
  return result;
}

std::optional<Link> Topology::QueryBestLink(const Participant& src,
                                            const Participant& dst,
                                            std::size_t bytes) const {
  auto links = QueryLinks(src, dst);
  if (links.empty()) {
    return std::nullopt;
  }
  return std::ranges::min(links, [bytes](const Link& a, const Link& b) {
    return a.TransferTimeUs(bytes) < b.TransferTimeUs(bytes);
  });
}

// Dijkstra's algorithm
std::optional<Path> Topology::ShortestPath(
    const Participant& src, const Participant& dst,
    std::function<float(const Link&)> cost_fn) const {
  if (src == dst) {
    return Path({src}, {});
  }

  std::unordered_map<Participant, float> dist;
  std::unordered_map<Participant, Participant> prev;
  std::unordered_map<Participant, Link> prev_link;

  using PQEntry = std::pair<float, Participant>;
  std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<>> pq;

  dist[src] = 0.0f;
  pq.push({0.0f, src});

  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();

    if (u == dst) {
      break;
    }
    if (d > dist[u]) {
      continue;
    }

    auto it = adj_.find(u);
    if (it == adj_.end()) {
      continue;
    }

    for (const auto& [v, link] : it->second) {
      float new_dist = d + cost_fn(link);
      if (!dist.contains(v) || new_dist < dist[v]) {
        dist[v] = new_dist;
        prev.insert_or_assign(v, u);
        prev_link.insert_or_assign(v, link);
        pq.push({new_dist, v});
      }
    }
  }

  if (!dist.contains(dst)) {
    return std::nullopt;
  }

  std::vector<Participant> hops;
  std::vector<Link> links;
  Participant current = dst;
  while (!(current == src)) {
    hops.push_back(current);
    links.push_back(prev_link.at(current));
    current = prev.at(current);
  }
  hops.push_back(src);

  std::ranges::reverse(hops);
  std::ranges::reverse(links);

  return Path(std::move(hops), std::move(links));
}

std::vector<Topology::Edge> Topology::GetEdges() const {
  std::vector<Edge> edges;
  for (const auto& [src, neighbors] : adj_) {
    for (const auto& [dst, link] : neighbors) {
      edges.emplace_back(src, dst, link);
    }
  }
  return edges;
}

}  // namespace setu::planner::topo