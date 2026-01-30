#pragma once

#include "commons/StdCommon.h"

namespace setu::ir {

struct SendInstr {};
struct RecvInstr {};
struct CopyInstr {};
struct InitCommInstr {};
struct UseCommInstr {};

using Instruction =
    std::variant<SendInstr, RecvInstr, CopyInstr, InitCommInstr, UseCommInstr>;
using Program = std::vector<Instruction>;
}  // namespace setu::ir
