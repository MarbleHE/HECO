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
  //TODO: check all elements properly
  auto t = op.x().getType().front().dyn_cast<SecretType>().getPlaintextType();
  inferredReturnTypes.push_back(SecretType::get(context, t));
  return ::mlir::success();
}

::mlir::LogicalResult fhe::AddOp::inferReturnTypes(::mlir::MLIRContext *context,
                                                        ::llvm::Optional<::mlir::Location> location,
                                                        ::mlir::ValueRange operands,
                                                        ::mlir::DictionaryAttr attributes,
                                                        ::mlir::RegionRange regions,
                                                        ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access operands
  // when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely "packaged" inside the operation class.
  auto op = AddOpAdaptor(operands, attributes,regions);
  //TODO: check all elements properly
  auto t = op.x().getType().front().dyn_cast<SecretType>().getPlaintextType();
  inferredReturnTypes.push_back(SecretType::get(context, t));
  return ::mlir::success();
}

::mlir::LogicalResult fhe::SubOp::inferReturnTypes(::mlir::MLIRContext *context,
                                                   ::llvm::Optional<::mlir::Location> location,
                                                   ::mlir::ValueRange operands,
                                                   ::mlir::DictionaryAttr attributes,
                                                   ::mlir::RegionRange regions,
                                                   ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access operands
  // when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely "packaged" inside the operation class.
  auto op = SubOpAdaptor(operands, attributes,regions);
  //TODO: check all elements properly
  auto t = op.x().getType().front().dyn_cast<SecretType>().getPlaintextType();
  inferredReturnTypes.push_back(SecretType::get(context, t));
  return ::mlir::success();
}

::mlir::LogicalResult fhe::ResolveOp::inferReturnTypes(::mlir::MLIRContext *context,
                                                   ::llvm::Optional<::mlir::Location> location,
                                                   ::mlir::ValueRange operands,
                                                   ::mlir::DictionaryAttr attributes,
                                                   ::mlir::RegionRange regions,
                                                   ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access operands
  // when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely "packaged" inside the operation class.
  auto op = ResolveOpAdaptor(operands, attributes,regions);
  //TODO: check all elements properly
  auto t = op.x().getType().dyn_cast<SecretType>().getPlaintextType();
  inferredReturnTypes.push_back(t);
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