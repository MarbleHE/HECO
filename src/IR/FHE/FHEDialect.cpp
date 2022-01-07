#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "../../../include/abc/IR/FHE/FHEDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace fhe;

#include "abc/IR/FHE/FHEOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "abc/IR/FHE/FHEOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// FHE dialect.
//===----------------------------------------------------------------------===//

void FHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "abc/IR/FHE/FHEOps.cpp.inc"
  >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "abc/IR/FHE/FHEOpsTypes.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "abc/IR/FHE/FHEOps.cpp.inc"