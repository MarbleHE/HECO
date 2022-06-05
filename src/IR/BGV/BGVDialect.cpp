#include "heco/IR/BGV/BGVDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace bgv;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heco/IR/BGV/BGVTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "heco/IR/BGV/BGV.cpp.inc"

::mlir::LogicalResult bgv::MultiplyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = MultiplyOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.y().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bgv.multiply must be of type bgv.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bgv.multiply must have same elementType.");
    auto new_size = (type_x.getSize() - 1) + (type_y.getSize() - 1) + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::SubOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = SubOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.y().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bgv.sub must be of type bgv.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bgv.sub must have same elementType.");
    auto new_size = std::max(type_x.getSize(), type_y.getSize());
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = AddOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.y().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bgv.add must be of type bgv.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bgv.add must have same elementType.");
    auto new_size = std::max(type_x.getSize(), type_y.getSize());
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::AddPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = AddPlainOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.y().getType().dyn_cast<PlaintextType>();
    assert(type_x && type_y && "Inputs to bgv.multiply_plain must be of type bgv.ctxt & bgv.ptxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bgv.add_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::SubPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = SubPlainOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.y().getType().dyn_cast<PlaintextType>();
    assert(type_x && type_y && "Inputs to bgv.multiply_plain must be of type bgv.ctxt & bgv.ptxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bgv.sub_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::RelinearizeOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = RelinearizeOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    assert(type_x && "Input to bgv.relinearize must be of type bgv.ctxt."); // Should never trigger
    assert(type_x.getSize() == 3 && "Size of input to bgv.relinearize must be three!");
    inferredReturnTypes.push_back(CiphertextType::get(context, 2, type_x.getElementType()));
    return ::mlir::success();
}

/// simplifies away materialization(materialization(x)) to x if the types work
::mlir::OpFoldResult bgv::MaterializeOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto m_op = input().getDefiningOp<bgv::MaterializeOp>())
        if (m_op.input().getType() == result().getType())
            return m_op.input();
    return {};
}

//===----------------------------------------------------------------------===//
// BGV dialect definitions
//===----------------------------------------------------------------------===//
#include "heco/IR/BGV/BGVDialect.cpp.inc"
void BGVDialect::initialize()
{
    // Registers all the Types into the BGVDialect class
    addTypes<
#define GET_TYPEDEF_LIST
#include "heco/IR/BGV/BGVTypes.cpp.inc"
        >();

    // Registers all the Operations into the BGVDialect class
    addOperations<
#define GET_OP_LIST
#include "heco/IR/BGV/BGV.cpp.inc"
        >();
}