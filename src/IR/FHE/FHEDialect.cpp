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

SecretType BatchedSecretType::getCorrespondingSecretType() const {
  return SecretType::get(getContext(), getPlaintextType());
}

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
  auto op = MultiplyOpAdaptor(operands, attributes, regions);
  auto plaintextType = Type();
  bool batched = false;
  for (auto operand: op.x()) {
    if (auto secret_type = operand.getType().dyn_cast_or_null<SecretType>()) {
      plaintextType = secret_type.getPlaintextType();
    }
    if (auto bst = operand.getType().dyn_cast_or_null<BatchedSecretType>()) {
      plaintextType = bst.getPlaintextType();
      batched = true;
    }
    //TODO: check things properly!
  }
  if (batched)
    inferredReturnTypes.push_back(BatchedSecretType::get(context, plaintextType));
  else
    inferredReturnTypes.push_back(SecretType::get(context, plaintextType));
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
  auto op = AddOpAdaptor(operands, attributes, regions);
  auto plaintextType = Type();
  bool batched = false;
  for (auto operand: op.x()) {
    if (auto secret_type = operand.getType().dyn_cast_or_null<SecretType>()) {
      plaintextType = secret_type.getPlaintextType();
    }
    if (auto bst = operand.getType().dyn_cast_or_null<BatchedSecretType>()) {
      plaintextType = bst.getPlaintextType();
      batched = true;
    }
    //TODO: check things properly!
  }
  if (batched)
    inferredReturnTypes.push_back(BatchedSecretType::get(context, plaintextType));
  else
    inferredReturnTypes.push_back(SecretType::get(context, plaintextType));
  return ::mlir::success();
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
  auto op = SubOpAdaptor(operands, attributes, regions);
  auto plaintextType = Type();
  bool batched = false;
  for (auto operand: op.x()) {
    if (auto secret_type = operand.getType().dyn_cast_or_null<SecretType>()) {
      plaintextType = secret_type.getPlaintextType();
    }
    if (auto bst = operand.getType().dyn_cast_or_null<BatchedSecretType>()) {
      plaintextType = bst.getPlaintextType();
      batched = true;
    }
    //TODO: check things properly!
  }
  if (batched)
    inferredReturnTypes.push_back(BatchedSecretType::get(context, plaintextType));
  else
    inferredReturnTypes.push_back(SecretType::get(context, plaintextType));
  return ::mlir::success();
}

::mlir::LogicalResult fhe::ConstOp::inferReturnTypes(::mlir::MLIRContext *context,
                                                     ::llvm::Optional<::mlir::Location> location,
                                                     ::mlir::ValueRange operands,
                                                     ::mlir::DictionaryAttr attributes,
                                                     ::mlir::RegionRange regions,
                                                     ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access operands
  // when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely "packaged" inside the operation class.
  auto op = ConstOpAdaptor(operands, attributes, regions);
  if (auto da = op.value().dyn_cast_or_null<DenseElementsAttr>()) {
    inferredReturnTypes.push_back(fhe::BatchedSecretType::get(context, da.getElementType()));
  } else {
    inferredReturnTypes.push_back(fhe::SecretType::get(context, op.value().getType()));
  }
  return ::mlir::success();
}

void fhe::ConstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto type = Type();
  if (getType().isa<SecretType>())
    type = getType().cast<SecretType>().getPlaintextType();
  else
    type = getType().cast<BatchedSecretType>().getPlaintextType();

  if (auto intCst = value().dyn_cast<IntegerAttr>()) {
    auto intType = type.dyn_cast<IntegerType>();

    // Sugar i1 constants with 'true' and 'false'.
    if (intType && intType.getWidth()==1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, build a complex name with the value and type.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << "c" << intCst.getInt();
    if (intType)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
  } else if (auto fCst = value().dyn_cast<FloatAttr>()) {
    auto floatType = type.dyn_cast<FloatType>();
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << "c" << (int) fCst.getValueAsDouble();
    if (floatType)
      specialName << "_s" << type;
    setNameFn(getResult(), specialName.str());
  } else if (auto arrayCst = value().dyn_cast<ArrayAttr>()) {
    //TODO: Somehow support array stuff better?
    setNameFn(getResult(), "vcst");
  } else {
    setNameFn(getResult(), "cst");
  }
}

::mlir::OpFoldResult fhe::ConstOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  return value();
}

::mlir::OpFoldResult fhe::MaterializeOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  if (auto m_op = input().getDefiningOp<fhe::MaterializeOp>())
    if (m_op.input().getType()==result().getType())
      return m_op.input();
  return {};
}

::mlir::OpFoldResult fhe::RotateOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands) {
  if (i()==0)
    return x();
  return {};
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