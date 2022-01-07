#ifndef ABC_FHE_FHEDIALECT_H
#define ABC_FHE_FHEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// Include the C++ class declaration for this Dialect (no define necessary for this one)
#include "abc/IR/FHE/FHEDialect.h.inc"

// Include the C++ class (and associated functions) declarations for this Dialect's types
#define GET_TYPEDEF_CLASSES
#include "abc/IR/FHE/FHETypes.h.inc"

// Include the C++ class (and associated functions) declarations for this Dialect's operations
#define GET_OP_CLASSES
#include "abc/IR/FHE/FHE.h.inc"

#endif // ABC_FHE_FHEDIALECT_H
