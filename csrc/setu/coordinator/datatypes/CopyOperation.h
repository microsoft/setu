#pragma once
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/CopySpec.h"
namespace setu::coordinator::datatypes {
using setu::commons::CopyOperationId;
using setu::commons::datatypes::CopySpecPtr;
struct CopyOperation {
    CopyOperationId id;
    CopySpecPtr spec;
};
using CopyOperationPtr = std::shared_ptr<CopyOperation>;
}