#ifndef AST_OPTIMIZER_MLIR_FHE_FHEDIALECT_H_
#define AST_OPTIMIZER_MLIR_FHE_FHEDIALECT_H_

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

#include "abc/IR/FHE/FHEOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "abc/IR/FHE/FHEOps.h.inc"

#define GET_TYPEDEF_CLASSES
#include "abc/IR/FHE/FHEOpsTypes.h.inc"

#endif // AST_OPTIMIZER_MLIR_FHE_FHEDIALECT_H_
