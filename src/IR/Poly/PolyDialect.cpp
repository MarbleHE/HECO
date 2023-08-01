#include "heco/IR/Poly/PolyDialect.h"
#include <iostream>
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace heco;
using namespace poly;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heco/IR/Poly/PolyTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "heco/IR/Poly/Poly.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd Poly dialect + additional boilerplate
//===----------------------------------------------------------------------===//
#include "heco/IR/Poly/PolyDialect.cpp.inc"

void PolyDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "heco/IR/Poly/Poly.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "heco/IR/Poly/PolyTypes.cpp.inc"
        >();
}