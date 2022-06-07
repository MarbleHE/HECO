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

::mlir::LogicalResult bgv::MultiplyManyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = MultiplyManyOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x()[0].getType().dyn_cast<CiphertextType>();
    auto new_size = type_x.getSize();
    for (auto xx : op.x())
    {
        CiphertextType type_xx = xx.getType().dyn_cast<CiphertextType>();
        assert(type_x && type_xx && "Inputs to bgv.add must be of type bgv.ctxt."); // Should never trigger
        assert(type_x.getElementType() == type_xx.getElementType() && "Inputs to bgv.add_many must have same elementType.");
        new_size = std::max(new_size, type_xx.getSize());
    }
    new_size = new_size + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::SubOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
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
    auto op = AddOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.y().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bgv.add must be of type bgv.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bgv.add must have same elementType.");
    auto new_size = std::max(type_x.getSize(), type_y.getSize());
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::AddManyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddManyOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x()[0].getType().dyn_cast<CiphertextType>();
    auto new_size = type_x.getSize();
    for (auto xx : op.x())
    {
        CiphertextType type_xx = xx.getType().dyn_cast<CiphertextType>();
        assert(type_x && type_xx && "Inputs to bgv.add must be of type bgv.ctxt."); // Should never trigger
        assert(type_x.getElementType() == type_xx.getElementType() && "Inputs to bgv.add_many must have same elementType.");
        new_size = std::max(new_size, type_xx.getSize());
    }
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::MultiplyPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = MultiplyPlainOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.y().getType().dyn_cast<PlaintextType>();
    assert(type_x && type_y && "Inputs to bgv.multiply_plain must be of type bgv.ctxt & bgv.ptxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bgv.multiply_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::AddPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
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
    auto op = SubPlainOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.y().getType().dyn_cast<PlaintextType>();
    assert(type_x && type_y && "Inputs to bgv.multiply_plain must be of type bgv.ctxt & bgv.ptxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bgv.sub_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bgv::ExponentiateOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ExponentiateOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    assert(type_x && "First input to bgv.exponentiate must be of type bgv.ctxt."); // Should never trigger
    auto new_size = type_x.getSize() + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
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

::mlir::LogicalResult bgv::RotateOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = RotateOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    assert(type_x && "Input to bgv.rotate must be of type bgv.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult bgv::ModswitchOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    assert(type_x && "Input to bgv.modswitch must be of type bgv.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult bgv::ModswitchPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchPlainOpAdaptor(operands, attributes, regions);
    PlaintextType type_x = op.x().getType().dyn_cast<PlaintextType>();
    assert(type_x && "Input to bgv.modswitch_plain must be of type bgv.ptxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult bgv::ModswitchToOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchToOpAdaptor(operands, attributes, regions);
    CiphertextType type_x = op.x().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.y().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bgv.modswitch_plain must be of type bgv.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_y);
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