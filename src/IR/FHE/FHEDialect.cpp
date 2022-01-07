#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "abc/IR/FHE/FHEDialect.h"

using namespace mlir;
using namespace fhe;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "abc/IR/FHE/FHETypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "abc/IR/FHE/FHE.cpp.inc"

::mlir::LogicalResult fhe::MultiplyOp::inferReturnTypes(::mlir::MLIRContext *context,
                                                        ::llvm::Optional<::mlir::Location> location,
                                                        ::mlir::ValueRange operands,
                                                        ::mlir::DictionaryAttr attributes,
                                                        ::mlir::RegionRange regions,
                                                        ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access operands
  // when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely "packaged" inside the operation class.
  auto op = MultiplyOpAdaptor(operands, attributes,regions);
  CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
  CiphertextType type_y = op.y().getType().dyn_cast<CiphertextType>();
  assert(type_x && type_y && "Inputs to fhe.multiply must be of type fhe.ctxt."); // Should never trigger
  assert(type_x.getElementType() == type_y.getElementType() && "Inputs to fhe.multiply must have same elementType.");
  auto new_size = (type_x.getSize() - 1) + (type_y.getSize() - 1) + 1;
  inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
  return ::mlir::success();
}

::mlir::LogicalResult fhe::RelinearizeOp::inferReturnTypes(::mlir::MLIRContext *context,
                                                           ::llvm::Optional<::mlir::Location> location,
                                                           ::mlir::ValueRange operands,
                                                           ::mlir::DictionaryAttr attributes,
                                                           ::mlir::RegionRange regions,
                                                           ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = RelinearizeOpAdaptor(operands, attributes,regions);
  CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
  assert(type_x && "Input to fhe.relinearize must be of type fhe.ctxt."); // Should never trigger
  assert(type_x.getSize()==3 && "Size of input to fhe.relinearize must be three!");
  inferredReturnTypes.push_back(CiphertextType::get(context, 2, type_x.getElementType()));
  return ::mlir::success();
}



//===----------------------------------------------------------------------===//
// FHE dialect definitions
//===----------------------------------------------------------------------===//
#include "abc/IR/FHE/FHEDialect.cpp.inc"
void FHEDialect::initialize() {

  // Registers all the Types into the FHEDialect class
  addTypes<
#define GET_TYPEDEF_LIST
#include "abc/IR/FHE/FHETypes.cpp.inc"
  >();

  // Registers all the Operations into the FHEDialect class
  addOperations<
#define GET_OP_LIST
#include "abc/IR/FHE/FHE.cpp.inc"
  >();

}

// TODO (Q&A): The <dialect>::parseType function seem to be generic boilerplate. Can we make TableGen generate them for us?
mlir::Type FHEDialect::parseType(::mlir::DialectAsmParser &parser) const {
  mlir::StringRef typeTag;
  if (parser.parseKeyword(&typeTag))
    return {};
  mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, typeTag, genType);
  if (parseResult.hasValue())
    return genType;
  parser.emitError(parser.getNameLoc(), "unknown fhe type: ") << typeTag;
  return {};
}

// TODO (Q&A): The <dialect>::printType function seem to be generic boilerplate. Can we make TableGen generate them for us?
void FHEDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &os) const {
  if (mlir::failed(generatedTypePrinter(type, os)))
    llvm::report_fatal_error("unknown type to print");
}