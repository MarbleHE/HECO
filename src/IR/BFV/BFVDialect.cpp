#include "heco/IR/BFV/BFVDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace heco;
using namespace bfv;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heco/IR/BFV/BFVTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "heco/IR/BFV/BFV.cpp.inc"

::mlir::LogicalResult bfv::MultiplyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = MultiplyOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bfv.multiply must be of type bfv.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bfv.multiply must have same elementType.");
    auto new_size = (type_x.getSize() - 1) + (type_y.getSize() - 1) + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::MultiplyManyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = MultiplyManyOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX()[0].getType().dyn_cast<CiphertextType>();
    auto new_size = type_x.getSize();
    for (auto xx : op.getX())
    {
        CiphertextType type_xx = xx.getType().dyn_cast<CiphertextType>();
        assert(type_x && type_xx && "Inputs to bfv.add must be of type bfv.ctxt."); // Should never trigger
        assert(
            type_x.getElementType() == type_xx.getElementType() &&
            "Inputs to bfv.add_many must have same elementType.");
        new_size = std::max(new_size, type_xx.getSize());
    }
    new_size = new_size + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::SubOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = SubOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bfv.sub must be of type bfv.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bfv.sub must have same elementType.");
    auto new_size = std::max(type_x.getSize(), type_y.getSize());
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bfv.add must be of type bfv.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bfv.add must have same elementType.");
    auto new_size = std::max(type_x.getSize(), type_y.getSize());
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::AddManyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddManyOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX()[0].getType().dyn_cast<CiphertextType>();
    auto new_size = type_x.getSize();
    for (auto xx : op.getX())
    {
        CiphertextType type_xx = xx.getType().dyn_cast<CiphertextType>();
        assert(type_x && type_xx && "Inputs to bfv.add must be of type bfv.ctxt."); // Should never trigger
        assert(
            type_x.getElementType() == type_xx.getElementType() &&
            "Inputs to bfv.add_many must have same elementType.");
        new_size = std::max(new_size, type_xx.getSize());
    }
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::MultiplyPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = MultiplyPlainOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.getY().getType().dyn_cast<PlaintextType>();
    assert(
        type_x && type_y &&
        "Inputs to bfv.multiply_plain must be of type bfv.ctxt & bfv.ptxt."); // Should never trigger
    assert(
        type_x.getElementType() == type_y.getElementType() &&
        "Inputs to bfv.multiply_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::AddPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddPlainOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.getY().getType().dyn_cast<PlaintextType>();
    assert(
        type_x && type_y &&
        "Inputs to bfv.multiply_plain must be of type bfv.ctxt & bfv.ptxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bfv.add_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::SubPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = SubPlainOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.getY().getType().dyn_cast<PlaintextType>();
    assert(
        type_x && type_y &&
        "Inputs to bfv.multiply_plain must be of type bfv.ctxt & bfv.ptxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to bfv.sub_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::ExponentiateOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ExponentiateOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    assert(type_x && "First input to bfv.exponentiate must be of type !bfv.ctxt."); // Should never trigger
    auto new_size = type_x.getSize() + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::RelinearizeOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = RelinearizeOpAdaptor(operands, attributes, properties, regions);
    auto type_x = op.getX().getType().dyn_cast<CiphertextType>();
    assert(type_x && "First input to bfv.relinearize must be of type !bfv.ctxt."); // Should never trigger
    assert(type_x.getSize() == 3 && "Size of input to bfv.relinearize must be three!");
    inferredReturnTypes.push_back(CiphertextType::get(context, 2, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult bfv::RotateOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = RotateOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    assert(type_x && "Input to bfv.rotate must be of type bfv.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult bfv::ModswitchOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    assert(type_x && "Input to bfv.modswitch must be of type bfv.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult bfv::ModswitchPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchPlainOpAdaptor(operands, attributes, properties, regions);
    PlaintextType type_x = op.getX().getType().dyn_cast<PlaintextType>();
    assert(type_x && "Input to bfv.modswitch_plain must be of type bfv.ptxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult bfv::ModswitchToOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchToOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to bfv.modswitch_plain must be of type bfv.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_y);
    return ::mlir::success();
}

/// simplifies a constant operation to its value (used for constant folding?)
::mlir::OpFoldResult bfv::ConstOp::fold(FoldAdaptor adaptor)
{
    return getValue();
}

/// simplifies away materialization(materialization(x)) to x if the types work
::mlir::OpFoldResult bfv::MaterializeOp::fold(FoldAdaptor adaptor)
{
    if (auto m_op = getInput().getDefiningOp<bfv::MaterializeOp>())
        if (m_op.getInput().getType() == getResult().getType())
            return m_op.getInput();
    return {};
}

/// simplifies away extract(v, 0) as scalars are simply "ctxt where we only care about slot 0"
::mlir::OpFoldResult bfv::ExtractOp::fold(FoldAdaptor adaptor)
{
    if (adaptor.getI().isZero())
        return getVector();
    else
        return nullptr;
}

//===----------------------------------------------------------------------===//
// BFV dialect definitions
//===----------------------------------------------------------------------===//
#include "heco/IR/BFV/BFVDialect.cpp.inc"
void BFVDialect::initialize()
{
    // Registers all the Types into the BFVDialect class
    addTypes<
#define GET_TYPEDEF_LIST
#include "heco/IR/BFV/BFVTypes.cpp.inc"
        >();

    // Registers all the Operations into the BFVDialect class
    addOperations<
#define GET_OP_LIST
#include "heco/IR/BFV/BFV.cpp.inc"
        >();
}