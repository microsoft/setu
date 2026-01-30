#pragma once

#include <nccl.h>
#include "planner/Planner.h"

namespace setu::planner::backends::nccl {
using setu::commons::datatypes::CopySpec;
using setu::metastore::MetaStore;

class NCCLPlanner : public Planner {
public:
    Plan Compile(CopySpec& copy_spec, const MetaStore& metastore) override;
private:
    std::map<std::set<Device>, ncclUniqueId> comm_cache_;
}
}