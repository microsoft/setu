#pragma once

#include "commons/StdCommon.h"
#include "planner/Participant.h"

namespace setu::planner::topo {
using setu::planner::Participant;

struct Link {
  explicit Link(float latency_us_param, float bandwidth_gbps_param,
                std::optional<std::string> tag_param = std::nullopt)
      : latency_us(latency_us_param),
        bandwidth_gbps(bandwidth_gbps_param),
        tag(std::move(tag_param)) {}

  float TransferTimeUs(std::size_t bytes) const {
    return latency_us + static_cast<float>(bytes) / (bandwidth_gbps * 1e3f);
  }

  [[nodiscard]] std::string ToString() const {
    return std::format("Link(latency={}us, bandwidth={}GB/s, tag={})",
                       latency_us, bandwidth_gbps, tag);
  }

  float latency_us;
  float bandwidth_gbps;
  std::optional<std::string> tag;
};

struct Path {
  explicit Path(std::vector<Participant> hops_param,
                std::vector<Link> links_param)
      : hops(std::move(hops_param)),
        links(std::move(links_param)),
        total_latency_us(ComputeTotalLatency(this->links)),
        bottleneck_bandwidth_gbps(ComputeBottleneckBandwidth(this->links)) {}

  float TransferTimeUsPipelined(std::size_t bytes) const {
    if (links.empty()) {
      return 0.0;
    }
    return total_latency_us +
           static_cast<float>(bytes) / (bottleneck_bandwidth_gbps * 1e3f);
  }

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "Path(hops={},links={},total_latency={}us,bottleneck_bandwidth={}GB/s)",
        hops, links, total_latency_us, bottleneck_bandwidth_gbps);
  }

  std::vector<Participant> hops;
  std::vector<Link> links;
  float total_latency_us;
  float bottleneck_bandwidth_gbps;

 private:
  static float ComputeTotalLatency(const std::vector<Link>& links) {
    return std::accumulate(
        links.begin(), links.end(), 0.0f,
        [](float sum, const Link& l) { return sum + l.latency_us; });
  }

  static float ComputeBottleneckBandwidth(const std::vector<Link>& links) {
    if (links.empty()) return std::numeric_limits<float>::max();
    return std::ranges::min(links, {}, &Link::bandwidth_gbps).bandwidth_gbps;
  }
};

class Topology {
 public:
  void AddLink(const Participant& src, const Participant& dst, Link link);

  void AddBidirectionalLink(const Participant& src, const Participant& dst,
                            Link link);

  std::vector<Link> QueryLinks(const Participant& src,
                               const Participant& dst) const;

  std::optional<Link> QueryBestLink(const Participant& src,
                                    const Participant& dst,
                                    std::size_t bytes) const;

  std::optional<Path> ShortestPath(
      const Participant& src, const Participant& dst,
      std::function<float(const Link&)> cost_fn) const;

 private:
  std::unordered_map<Participant, std::vector<std::pair<Participant, Link>>>
      adj_;
};

using TopologyPtr = std::shared_ptr<Topology>;

}  // namespace setu::planner::topo
