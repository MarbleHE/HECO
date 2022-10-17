#include "heco/IR/CKKS/CKKSDialect.h"
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
using namespace ckks;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heco/IR/CKKS/CKKSTypes.cpp.inc"

ScalarCipherType VectorCipherType::getCorrespondingScalarCipherType() const
{
    return ScalarCipherType::get(getContext(), getPlaintextType());
}

VectorCipherType VectorCipherType::get(::mlir::MLIRContext *context, ::mlir::Type plaintextType)
{
    return get(context, plaintextType, -24);
}

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "heco/IR/CKKS/CKKS.cpp.inc"

::mlir::LogicalResult ckks::MultiplyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = MultiplyOpAdaptor(operands, attributes, regions);
    auto plaintextType = Type();
    int size = -173;
    bool batched = false;
    for (auto operand : op.x())
    {
        if (auto secret_type = operand.getType().dyn_cast_or_null<ScalarCipherType>())
        {
            plaintextType = secret_type.getPlaintextType();
        }
        if (auto bst = operand.getType().dyn_cast_or_null<VectorCipherType>())
        {
            plaintextType = bst.getPlaintextType();
            size = bst.getSize() < 0 ? size : bst.getSize();
            batched = true;
        }
        // TODO: check things properly!
    }
    if (batched)
        inferredReturnTypes.push_back(VectorCipherType::get(context, plaintextType, size));
    else
        inferredReturnTypes.push_back(ScalarCipherType::get(context, plaintextType));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = AddOpAdaptor(operands, attributes, regions);
    auto plaintextType = Type();
    int size = -1;
    bool batched = false;
    for (auto operand : op.x())
    {
        if (auto secret_type = operand.getType().dyn_cast_or_null<ScalarCipherType>())
        {
            plaintextType = secret_type.getPlaintextType();
        }
        if (auto bst = operand.getType().dyn_cast_or_null<VectorCipherType>())
        {
            plaintextType = bst.getPlaintextType();
            size = bst.getSize() < 0 ? size : bst.getSize();
            batched = true;
        }
        // TODO: check things properly!
    }
    if (batched)
        inferredReturnTypes.push_back(VectorCipherType::get(context, plaintextType, size));
    else
        inferredReturnTypes.push_back(ScalarCipherType::get(context, plaintextType));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::SubOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = SubOpAdaptor(operands, attributes, regions);
    auto plaintextType = Type();
    int size = -1;
    bool batched = false;
    for (auto operand : op.x())
    {
        if (auto secret_type = operand.getType().dyn_cast_or_null<ScalarCipherType>())
        {
            plaintextType = secret_type.getPlaintextType();
        }
        if (auto bst = operand.getType().dyn_cast_or_null<VectorCipherType>())
        {
            plaintextType = bst.getPlaintextType();
            size = bst.getSize() < 0 ? size : bst.getSize();
            batched = true;
        }
        // TODO: check things properly!
    }
    if (batched)
        inferredReturnTypes.push_back(VectorCipherType::get(context, plaintextType, size));
    else
        inferredReturnTypes.push_back(ScalarCipherType::get(context, plaintextType));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::ConstOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = ConstOpAdaptor(operands, attributes, regions);
    if (auto da = op.value().dyn_cast_or_null<DenseElementsAttr>())
    {
        inferredReturnTypes.push_back(ckks::VectorCipherType::get(context, da.getElementType()));
    }
    else
    {
        inferredReturnTypes.push_back(ckks::ScalarCipherType::get(context, op.value().getType()));
    }
    return ::mlir::success();
}

void ckks::ConstOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
{
    auto type = Type();
    if (getType().isa<ScalarCipherType>())
        type = getType().cast<ScalarCipherType>().getPlaintextType();
    else
        type = getType().cast<VectorCipherType>().getPlaintextType();

    if (auto intCst = value().dyn_cast<IntegerAttr>())
    {
        auto intType = type.dyn_cast<IntegerType>();

        // Sugar i1 constants with 'true' and 'false'.
        if (intType && intType.getWidth() == 1)
            return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

        // Otherwise, build a complex name with the value and type.
        SmallString<32> specialNameBuffer;
        llvm::raw_svector_ostream specialName(specialNameBuffer);
        specialName << "c" << intCst.getInt();
        if (intType)
            specialName << '_' << type;
        setNameFn(getResult(), specialName.str());
    }
    else if (auto fCst = value().dyn_cast<FloatAttr>())
    {
        auto floatType = type.dyn_cast<FloatType>();
        SmallString<32> specialNameBuffer;
        llvm::raw_svector_ostream specialName(specialNameBuffer);
        specialName << "c" << (int)fCst.getValueAsDouble();
        if (floatType)
            specialName << "_s" << type;
        setNameFn(getResult(), specialName.str());
    }
    else if (auto arrayCst = value().dyn_cast<ArrayAttr>())
    {
        // TODO: Somehow support array stuff better?
        setNameFn(getResult(), "vcst");
    }
    else
    {
        setNameFn(getResult(), "cst");
    }
}

/// simplifies a constant operation to its value (used for constant folding?)
::mlir::OpFoldResult ckks::ConstOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return value();
}

/// simplifies away materialization(materialization(x)) to x if the types work
::mlir::OpFoldResult ckks::MaterializeOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto m_op = input().getDefiningOp<ckks::MaterializeOp>())
        if (m_op.input().getType() == result().getType())
            return m_op.input();
    return {};
}

/// simplifies rotate(x,0) to x
::mlir::OpFoldResult ckks::RotateOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    bool ASSUME_VECTOR_CYCLICAL = true; // TODO: introduce a flag for this!!

    // Simplify
    //   %op = rotate(%x) by 0
    // to
    //   %x
    if (i() == 0)
        return x();

    // Simplify
    //   %op = rotate(%x) by -1
    // to
    //   %op = rotate(%x) by k where k == x.size() - 1;
    // I.e. we wrap around rotations to de-duplicate them for later phases
    if (ASSUME_VECTOR_CYCLICAL)
    {
        if (i() < 0)
        { // wrap around to positive values. Technically not always the best choice,
            // but in theory we could always revert that again later when generating code,
            // when we know what rotations are natively available
            auto vec_size = x().getType().dyn_cast<VectorCipherType>().getSize();
            if (vec_size > 0)
            {
                getOperation()->setAttr(
                    "i",
                    IntegerAttr::get(IntegerType::get(getContext(), 32, mlir::IntegerType::Signed), vec_size + i()));
            }
        }
    }

    return {};
}
/// simplifies add(x,0) and add(x) to x
::mlir::OpFoldResult ckks::AddOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto neutral_element = 0;
    SmallVector<Value> new_operands;
    for (auto v : x())
    {
        bool omit = false;
        if (auto cst_op = v.getDefiningOp<ckks::ConstOp>())
        {
            if (auto dea = cst_op.value().dyn_cast_or_null<DenseElementsAttr>())
            {
                if (dea.size() == 1)
                {
                    if (dea.getElementType().isIntOrIndex())
                    {
                        if (dea.value_begin<const IntegerAttr>()->getInt() == neutral_element)
                            omit = true;
                    }
                    else if (dea.getElementType().isIntOrFloat())
                    {
                        // because we've already excluded IntOrIndex, it must be float
                        if (dea.value_begin<const FloatAttr>()->getValueAsDouble() == neutral_element)
                            omit = true;
                    }
                }
            }
            else if (auto ia = cst_op.value().dyn_cast_or_null<IntegerAttr>())
            {
                if (ia.getInt() == neutral_element)
                    omit = true;
            }
            else if (auto fa = cst_op.value().dyn_cast_or_null<FloatAttr>())
            {
                if (fa.getValueAsDouble() == neutral_element)
                    omit = true;
            }
        }
        if (!omit)
            new_operands.push_back(v);
    }
    xMutable().assign(new_operands);
    if (x().size() > 1)
        return getResult();
    else
        return x().front();
}
/// simplifies sub(x,0) and sub(x) to x
::mlir::OpFoldResult ckks::SubOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto neutral_element = 0;
    SmallVector<Value> new_operands;
    for (auto v : x())
    {
        bool omit = false;
        if (auto cst_op = v.getDefiningOp<ckks::ConstOp>())
        {
            if (auto dea = cst_op.value().dyn_cast_or_null<DenseElementsAttr>())
            {
                if (dea.size() == 1)
                {
                    if (dea.getElementType().isIntOrIndex())
                    {
                        if (dea.value_begin<const IntegerAttr>()->getInt() == neutral_element)
                            omit = true;
                    }
                    else if (dea.getElementType().isIntOrFloat())
                    {
                        // because we've already excluded IntOrIndex, it must be float
                        if (dea.value_begin<const FloatAttr>()->getValueAsDouble() == neutral_element)
                            omit = true;
                    }
                }
            }
            else if (auto ia = cst_op.value().dyn_cast_or_null<IntegerAttr>())
            {
                if (ia.getInt() == neutral_element)
                    omit = true;
            }
            else if (auto fa = cst_op.value().dyn_cast_or_null<FloatAttr>())
            {
                if (fa.getValueAsDouble() == neutral_element)
                    omit = true;
            }
        }
        if (!omit)
            new_operands.push_back(v);
    }
    xMutable().assign(new_operands);
    if (x().size() > 1)
        return getResult();
    else
        return x().front();
}
/// simplifies mul(x,1) and mul(x) to x
::mlir::OpFoldResult ckks::MultiplyOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    auto neutral_element = 1;
    SmallVector<Value> new_operands;
    for (auto v : x())
    {
        bool omit = false;
        if (auto cst_op = v.getDefiningOp<ckks::ConstOp>())
        {
            if (auto dea = cst_op.value().dyn_cast_or_null<DenseElementsAttr>())
            {
                if (dea.size() == 1)
                {
                    if (dea.getElementType().isIntOrIndex())
                    {
                        if (dea.value_begin<const IntegerAttr>()->getInt() == neutral_element)
                            omit = true;
                    }
                    else if (dea.getElementType().isIntOrFloat())
                    {
                        // because we've already excluded IntOrIndex, it must be float
                        if (dea.value_begin<const FloatAttr>()->getValueAsDouble() == neutral_element)
                            omit = true;
                    }
                }
            }
            else if (auto ia = cst_op.value().dyn_cast_or_null<IntegerAttr>())
            {
                if (ia.getInt() == neutral_element)
                    omit = true;
            }
            else if (auto fa = cst_op.value().dyn_cast_or_null<FloatAttr>())
            {
                if (fa.getValueAsDouble() == neutral_element)
                    omit = true;
            }
        }
        if (!omit)
            new_operands.push_back(v);
    }
    xMutable().assign(new_operands);
    if (x().size() > 1)
        return getResult();
    else
        return x().front();
}

//===----------------------------------------------------------------------===//
// CKKS dialect definitions
//===----------------------------------------------------------------------===//
#include "heco/IR/CKKS/CKKSDialect.cpp.inc"
void CKKSDialect::initialize()
{
    // Registers all the Types into the CKKSDialect class
    addTypes<
#define GET_TYPEDEF_LIST
#include "heco/IR/CKKS/CKKSTypes.cpp.inc"
        >();

    // Registers all the Operations into the CKKSDialect class
    addOperations<
#define GET_OP_LIST
#include "heco/IR/CKKS/CKKS.cpp.inc"
        >();
}