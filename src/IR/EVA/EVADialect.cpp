#include "heco/IR/EVA/EVADialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace heco;
using namespace eva;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heco/IR/EVA/EVATypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "heco/IR/EVA/EVA.cpp.inc"

::mlir::LogicalResult eva::MultiplyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = MultiplyOpAdaptor(operands, attributes, properties, regions);
    int size = -173;
    for (auto operand : {op.getX(), op.getY()})
    {
        if (auto ctxt = operand.getType().dyn_cast_or_null<CipherType>()) {
            size = ctxt.getSize();
        }
        if (auto ptxt = operand.getType().dyn_cast_or_null<VectorType>()) {
            size = ptxt.getSize();
        }
    }

    // it's always a ciphertext (according to the EVA paper)
    inferredReturnTypes.push_back(CipherType::get(context, size));
    return ::mlir::success();
}

::mlir::LogicalResult eva::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    int size = -173;
    for (auto operand : {op.getX(), op.getY()})
    {
        if (auto ctxt = operand.getType().dyn_cast_or_null<CipherType>()) {
            size = ctxt.getSize();
        }
        if (auto ptxt = operand.getType().dyn_cast_or_null<VectorType>()) {
            size = ptxt.getSize();
        }

        // TODO: check things properly! (including encryption parameters)
    }

    // it's always a ciphertext (according to the EVA paper)
    inferredReturnTypes.push_back(CipherType::get(context, size));
    return ::mlir::success();
}

::mlir::LogicalResult eva::SubOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = SubOpAdaptor(operands, attributes, properties, regions);
    int size = -173;
    for (auto operand : {op.getX(), op.getY()})
    {
        if (auto ctxt = operand.getType().dyn_cast_or_null<CipherType>()) {
            size = ctxt.getSize();
        }
        if (auto ptxt = operand.getType().dyn_cast_or_null<VectorType>()) {
            size = ptxt.getSize();
        }

        // TODO: check things properly! (including encryption parameters)
    }

    // it's always a ciphertext (according to the EVA paper)
    inferredReturnTypes.push_back(CipherType::get(context, size));
    return ::mlir::success();
}

/// simplifies away negation(negation(x)) to x if the types work
::mlir::OpFoldResult eva::NegateOp::fold(FoldAdaptor adaptor)
{
    if (auto m_op = getInput().getDefiningOp<eva::NegateOp>())
        if (m_op.getInput().getType() == getResult().getType())
            return m_op.getInput();
    return {};
}


/// simplifies a constant operation to its value (used for constant folding?)
::mlir::OpFoldResult eva::ConstOp::fold(FoldAdaptor adaptor)
{
    return getValue();
}

/// simplifies away materialization(materialization(x)) to x if the types work
::mlir::OpFoldResult eva::MaterializeOp::fold(FoldAdaptor adaptor)
{
    if (auto m_op = getInput().getDefiningOp<eva::MaterializeOp>())
        if (m_op.getInput().getType() == getResult().getType())
            return m_op.getInput();
    return {};
}

/// simplifies rotate(x,0) to x
::mlir::OpFoldResult eva::RotateOp::fold(FoldAdaptor adaptor)
{
    bool ASSUME_VECTOR_CYCLICAL = true; // TODO: introduce a flag for this!!

    // Simplify
    //   %op = rotate(%x) by 0
    // to
    //   %x
    if (getI() == 0)
        return getX();

    // Simplify
    //   %op = rotate(%x) by -1
    // to
    //   %op = rotate(%x) by k where k == x.size() - 1;
    // I.e. we wrap around rotations to de-duplicate them for later phases
    if (ASSUME_VECTOR_CYCLICAL)
    {
        if (getI() < 0)
        { // wrap around to positive values. Technically not always the best choice,
            // but in theory we could always revert that again later when generating code,
            // when we know what rotations are natively available
            auto vec_size = getX().getType().dyn_cast<CipherType>().getSize();
            if (vec_size > 0)
            {
                getOperation()->setAttr(
                    "i",
                    IntegerAttr::get(IntegerType::get(getContext(), 32, mlir::IntegerType::Signed), vec_size + getI()));
            }
        }
    }

    return {};
}

//===----------------------------------------------------------------------===//
// EVA dialect definitions
//===----------------------------------------------------------------------===//
#include "heco/IR/EVA/EVADialect.cpp.inc"

void EVADialect::initialize()
{
    // Registers all the Types into the EVADialect class
    addTypes<
#define GET_TYPEDEF_LIST
#include "heco/IR/EVA/EVATypes.cpp.inc"
        >();

    // Registers all the Operations into the EVADialect class
    addOperations<
#define GET_OP_LIST
#include "heco/IR/EVA/EVA.cpp.inc"
        >();
}