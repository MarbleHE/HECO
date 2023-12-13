#ifndef HECO_IR_EVA_EVADIALECT_H
#define HECO_IR_EVA_EVADIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the C++ class declaration for this Dialect (no define necessary for this one)
#include "heco/IR/EVA/EVADialect.h.inc"

// Include the C++ class (and associated functions) declarations for this Dialect's types
#define GET_TYPEDEF_CLASSES
#include "heco/IR/EVA/EVATypes.h.inc"

// Include the C++ class (and associated functions) declarations for this Dialect's operations
#define GET_OP_CLASSES
#include "heco/IR/EVA/EVA.h.inc"

#include "heco/IR/EVA/EVAAnalyses.h"

#endif // HECO_IR_EVA_EVADIALECT_H
