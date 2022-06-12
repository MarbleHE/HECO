#ifndef HECO_IR_BGV_BGVDIALECT_H_
#define HECO_IR_BGV_BGVDIALECT_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the C++ class declaration for this Dialect (no define necessary for this one)
#include "heco/IR/BGV/BGVDialect.h.inc"

// Include the C++ class (and associated functions) declarations for this Dialect's types
#define GET_TYPEDEF_CLASSES
#include "heco/IR/BGV/BGVTypes.h.inc"

// Include the C++ class (and associated functions) declarations for this Dialect's operations
#define GET_OP_CLASSES
#include "heco/IR/BGV/BGV.h.inc"

#endif // HECO_IR_BGV_BGVDIALECT_H_
