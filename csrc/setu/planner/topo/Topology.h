#pragma once

#include "commons/StdCommon.h"
#include "commons/utils/Serialization.h"
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

  void Serialize(setu::commons::BinaryBuffer& buffer) const {
    setu::commons::utils::BinaryWriter writer(buffer);
    writer.WriteFields(latency_us, bandwidth_gbps, tag);
  }

  static Link Deserialize(const setu::commons::BinaryRange& range) {
    setu::commons::utils::BinaryReader reader(range);
    auto [lat, bw, t] =
        reader.ReadFields<float, float, std::optional<std::string>>();
    return Link(lat, bw, std::move(t));
  }

  float latency_us;
  float bandwidth_gbps;
  std::optional<std::string> tag;
};

struct Path {
  Path() = default;

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

  void Serialize(setu::commons::BinaryBuffer& buffer) const {
    setu::commons::utils::BinaryWriter writer(buffer);
    // Write hops manually since Participant is TriviallySerializable
    // but not default-constructible (c10::Device has no default ctor),
    // and the generic vector reader uses resize() which requires it.
    writer.Write<std::uint32_t>(static_cast<std::uint32_t>(hops.size()));
    for (const auto& hop : hops) writer.Write(hop);
    writer.Write(links);
  }

  static Path Deserialize(const setu::commons::BinaryRange& range) {
    setu::commons::utils::BinaryReader reader(range);
    // Read hops manually to avoid vector::resize() on non-default-
    // constructible Participant.
    const auto num_hops = reader.Read<std::uint32_t>();
    std::vector<Participant> h;
    h.reserve(num_hops);
    for (std::uint32_t i = 0; i < num_hops; ++i) {
      h.push_back(reader.Read<Participant>());
    }
    auto l = reader.Read<std::vector<Link>>();
    return Path(std::move(h), std::move(l));
  }

  std::vector<Participant> hops;
  std::vector<Link> links;
  float total_latency_us{0.0f};
  float bottleneck_bandwidth_gbps{0.0f};

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

  using Edge = std::tuple<Participant, Participant, Link>;

  [[nodiscard]] std::vector<Edge> GetEdges() const;

 private:
  std::unordered_map<Participant, std::vector<std::pair<Participant, Link>>>
      adj_;
};

using TopologyPtr = std::shared_ptr<Topology>;

}  // namespace setu::planner::topo
