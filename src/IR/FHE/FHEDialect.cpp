#include "heco/IR/FHE/FHEDialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace heco;
using namespace fhe;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heco/IR/FHE/FHETypes.cpp.inc"

SecretType BatchedSecretType::getCorrespondingSecretType() const
{
    return SecretType::get(getContext(), getPlaintextType());
}

BatchedSecretType BatchedSecretType::get(::mlir::MLIRContext *context, ::mlir::Type plaintextType)
{
    assert(false && "This really should not be used!");
    return get(context, plaintextType, -24);
}

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
::mlir::ParseResult fhe::CombineOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
{
    parser.parseLParen();

    llvm::SmallVector<std::pair<mlir::OpAsmParser::UnresolvedOperand, llvm::SmallVector<Attribute>>> inputs;
    mlir::OpAsmParser::UnresolvedOperand remaining_inputs;
    bool done = false;
    while (!done)
    {
        // Get the operand
        mlir::OpAsmParser::UnresolvedOperand result;
        parser.parseOperand(result);

        // Get the indices
        if (parser.parseOptionalLSquare().succeeded())
        {
            inputs.push_back({ result, {} });
            // TODO: Add support for multiple things, i.e. [i:j, k, l:m] appearing inside one set of square brackets
            int a;
            parser.parseInteger(a);
            int b = a;
            if (parser.parseOptionalColon().succeeded())
                parser.parseInteger(b);
            for (int i = a; i <= b; ++i)
                inputs.back().second.push_back(parser.getBuilder().getIndexAttr(i));

            parser.parseRSquare();
        }
        else
        {
            assert(remaining_inputs.name.empty() && "must not have multiple remaining_inputs in fhe.combine op");
            remaining_inputs = result;
        }

        // We've parsed the current operand, check if there is another one:
        if (parser.parseOptionalComma().failed())
        {
            done = true;
        }
    }

    parser.parseRParen();

    // Parse type at the end
    Type type;
    parser.parseColonType(type);

    // Resolve the operands
    llvm::SmallVector<Value> operands;
    SmallVector<Attribute> indices;
    llvm::SmallVector<Type> types;
    for (auto p : inputs)
    {
        llvm::SmallVector<Value> found_values;
        parser.resolveOperand(p.first, type, found_values);
        if (found_values.size() != 1)
            return failure();
        operands.push_back(found_values[0]);

        if (p.second.size() == 1)
            indices.push_back(p.second.front());
        else
            indices.push_back(parser.getBuilder().getArrayAttr(p.second));

        types.push_back(found_values[0].getType());
    }
    if (!remaining_inputs.name.empty())
    {
        llvm::SmallVector<Value> found_values;
        parser.resolveOperand(remaining_inputs, type, found_values);
        if (found_values.size() != 1)
            return failure();
        operands.push_back(found_values[0]);
        indices.push_back(parser.getBuilder().getStringAttr("all"));
        types.push_back(found_values[0].getType());
    }

    // build the actual op/op state
    result.addAttribute("indices", parser.getBuilder().getArrayAttr(indices));
    result.addOperands(operands);
    result.addTypes(type);

    return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void fhe::CombineOp::print(::mlir::OpAsmPrinter &printer)
{
    auto &op = *this;
    printer << "(";
    assert(op.getVectors().size() == op.getIndices().size() && "combine op must have indices entry for each operand");
    auto indices = op.getIndices().getValue();
    for (size_t i = 0; i < op.getVectors().size(); ++i)
    {
        if (i != 0)
            printer << ", ";
        printer.printOperand(op.getOperand(i));
        // print the index, if it exists
        if (auto aa = indices[i].dyn_cast_or_null<ArrayAttr>())
        {
            bool continuous = aa.size() > 1;
            for (size_t j = 1; j < aa.size(); ++j)
                continuous &= aa[j - 1].dyn_cast<IntegerAttr>().getInt() + 1 == aa[j].dyn_cast<IntegerAttr>().getInt();

            if (continuous)
            {
                // TODO: Update this to always print stuff continuously if possible, gobbling input until non-continuous
                //  and only then emitting a comma and the next value.
                //  Requires writing to a sstream first and later wrapping "["/"]" iff a comma was emitted.
                auto start = aa[0].dyn_cast<IntegerAttr>().getInt();
                auto end = aa[aa.size() - 1].dyn_cast<IntegerAttr>().getInt();
                printer << "[" << start << ":" << end << "]";
            }
            else if (aa.size() > 1)
            {
                printer << "[";
                for (size_t j = 0; j < aa.size(); ++j)
                {
                    if (j != 0)
                        printer << ", ";
                    printer << aa[j].dyn_cast<IntegerAttr>().getInt();
                }
                printer << "]";
            }
            else
            { // single value inside an array...weird but OK
                printer << "[" << aa[0].dyn_cast<IntegerAttr>().getInt() << "]";
            }
        }
        else if (auto ia = indices[i].dyn_cast_or_null<IntegerAttr>())
        {
            printer << "[" << ia.getInt() << "]";
        } // else -> do not print implicit "all"
    }
    printer << ") : ";
    printer.printType(op.getType());
}

#define GET_OP_CLASSES
#include "heco/IR/FHE/FHE.cpp.inc"

::mlir::LogicalResult fhe::MultiplyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = MultiplyOpAdaptor(operands, attributes, properties, regions);
    auto plaintextType = Type();
    int size = -173;
    bool batched = false;
    for (auto operand : op.getX())
    {
        if (auto secret_type = operand.getType().dyn_cast_or_null<SecretType>())
        {
            plaintextType = secret_type.getPlaintextType();
        }
        if (auto bst = operand.getType().dyn_cast_or_null<BatchedSecretType>())
        {
            plaintextType = bst.getPlaintextType();
            size = bst.getSize() < 0 ? size : bst.getSize();
            batched = true;
        }
        // TODO: check things properly!
    }
    if (batched)
        inferredReturnTypes.push_back(BatchedSecretType::get(context, plaintextType, size));
    else
        inferredReturnTypes.push_back(SecretType::get(context, plaintextType));
    return ::mlir::success();
}

::mlir::LogicalResult fhe::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    auto plaintextType = Type();
    int size = -1;
    bool batched = false;
    for (auto operand : op.getX())
    {
        if (auto secret_type = operand.getType().dyn_cast_or_null<SecretType>())
        {
            plaintextType = secret_type.getPlaintextType();
        }
        if (auto bst = operand.getType().dyn_cast_or_null<BatchedSecretType>())
        {
            plaintextType = bst.getPlaintextType();
            size = bst.getSize() < 0 ? size : bst.getSize();
            batched = true;
        }
        // TODO: check things properly!
    }
    if (batched)
        inferredReturnTypes.push_back(BatchedSecretType::get(context, plaintextType, size));
    else
        inferredReturnTypes.push_back(SecretType::get(context, plaintextType));
    return ::mlir::success();
}

::mlir::LogicalResult fhe::SubOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = SubOpAdaptor(operands, attributes, properties, regions);
    auto plaintextType = Type();
    int size = -1;
    bool batched = false;
    for (auto operand : op.getX())
    {
        if (auto secret_type = operand.getType().dyn_cast_or_null<SecretType>())
        {
            plaintextType = secret_type.getPlaintextType();
        }
        if (auto bst = operand.getType().dyn_cast_or_null<BatchedSecretType>())
        {
            plaintextType = bst.getPlaintextType();
            size = bst.getSize() < 0 ? size : bst.getSize();
            batched = true;
        }
        // TODO: check things properly!
    }
    if (batched)
        inferredReturnTypes.push_back(BatchedSecretType::get(context, plaintextType, size));
    else
        inferredReturnTypes.push_back(SecretType::get(context, plaintextType));
    return ::mlir::success();
}

::mlir::LogicalResult fhe::ConstOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = ConstOpAdaptor(operands, attributes, properties, regions);
    if (DenseElementsAttr da = op.getValue().dyn_cast_or_null<DenseElementsAttr>())
    {
        inferredReturnTypes.push_back(fhe::BatchedSecretType::get(context, da.getElementType(), da.getNumElements()));
    }
    else
    {
        inferredReturnTypes.push_back(fhe::SecretType::get(context, op.getValue().getType()));
    }
    return ::mlir::success();
}

void fhe::ConstOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
{
    auto type = Type();
    if (getType().isa<SecretType>())
        type = getType().cast<SecretType>().getPlaintextType();
    else
        type = getType().cast<BatchedSecretType>().getPlaintextType();

    if (auto intCst = getValue().dyn_cast<IntegerAttr>())
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
    else if (auto fCst = getValue().dyn_cast<FloatAttr>())
    {
        auto floatType = type.dyn_cast<FloatType>();
        SmallString<32> specialNameBuffer;
        llvm::raw_svector_ostream specialName(specialNameBuffer);
        specialName << "c" << (int)fCst.getValueAsDouble();
        if (floatType)
            specialName << "_s" << type;
        setNameFn(getResult(), specialName.str());
    }
    else if (auto arrayCst = getValue().dyn_cast<ArrayAttr>())
    {
        // TODO: Somehow support array stuff better?
        setNameFn(getResult(), "vcst");
    }
    else
    {
        setNameFn(getResult(), "cst");
    }
}

/// Simplifies /during EmitC processing/
///  %os = materialize(%ctxt)->bst
///  %ex_op = extract(%os, i)
///  %op = materialize(%ex_op) -> ctxt
/// to
///  %op = rotate(%ctxt, -i)
::mlir::LogicalResult fhe::MaterializeOp::canonicalize(MaterializeOp op, ::mlir::PatternRewriter &rewriter)
{
    if (auto ot = op.getType().dyn_cast_or_null<emitc::OpaqueType>())
    {
        if (ot.getValue() == "seal::Ciphertext")
        {
            if (auto ex_op = op.getInput().getDefiningOp<fhe::ExtractOp>())
            {
                if (auto original_source = ex_op.getVector().getDefiningOp<MaterializeOp>())
                {
                    if (auto original_ot = original_source.getInput().getType().dyn_cast_or_null<emitc::OpaqueType>())
                    {
                        if (original_ot.getValue() == "seal::Ciphertext")
                        {
                            // we now have something like this
                            // %os = materialize(%ctxt)->bst
                            // %ex_op = extract(%os, i)
                            // %op = materialize(%ex_op) -> ctxt
                            //
                            // Instead of doing all of that, we can just change this to
                            // %op = rotate(%ctxt, -i)
                            //
                            // This works because in ctxt land, there are no more scalars (result of extract)
                            // and so "scalar" just means "what's in position 0 of the ctxt"
                            //
                            // Note that we don't actually remove the first materialize and ex_op,
                            // since they'll be canonicalized away anyway as dead code if appropriate

                            // rewriter.replaceOpWithNewOp<emitc::CallOp>(op, ex_op.getVector(),
                            // -ex_op.getI().getLimitedValue(INT32_MAX));
                            auto i = (int)ex_op.getI().getLimitedValue(INT32_MAX);
                            auto a0 = rewriter.getIndexAttr(0); // stands for "first operand"
                            auto a1 = rewriter.getSI32IntegerAttr(i);
                            auto aa = ArrayAttr::get(rewriter.getContext(), { a0, a1 });
                            rewriter.replaceOpWithNewOp<emitc::CallOp>(
                                op, TypeRange(ot), llvm::StringRef("evaluator.rotate"), aa, ArrayAttr(),
                                ValueRange(original_source.getInput()));
                            return success();
                        }
                    }
                }
            }
        }
    }
    return failure();
}

// replaces insert (extract %v1, i) into %v2, i  (note: i must match!) with combine (v1,v2) ([i], [-i])
// where [-i] means "everything except i"
::mlir::LogicalResult fhe::InsertOp::canonicalize(InsertOp op, ::mlir::PatternRewriter &rewriter)
{
    if (auto ex_op = op.getScalar().getDefiningOp<fhe::ExtractOp>())
    {
        auto i = (int)ex_op.getI().getLimitedValue(INT32_MAX);
        auto v1 = ex_op.getVector();
        auto bst = ex_op.getVector().getType().dyn_cast<fhe::BatchedSecretType>();
        if (bst == op.getDest().getType())
        {
            if (i == (int)op.getI().getLimitedValue(INT32_MAX))
            {
                auto ai = rewriter.getIndexAttr(i);
                auto ami = rewriter.getStringAttr("all");
                auto aa = rewriter.getArrayAttr({ ai, ami });
                rewriter.replaceOpWithNewOp<fhe::CombineOp>(op, bst, ValueRange({ v1, op.getDest() }), aa);
                return success();
            }
        }
    }
    return failure();
}

/// simplifies a constant operation to its value (used for constant folding?)
::mlir::OpFoldResult fhe::ConstOp::fold(FoldAdaptor adaptor)
{
    return getValue();
}

/// simplifies away materialization(materialization(x)) to x if the types work
::mlir::OpFoldResult fhe::MaterializeOp::fold(FoldAdaptor adaptor)
{
    if (auto m_op = getInput().getDefiningOp<fhe::MaterializeOp>())
        if (m_op.getInput().getType() == getResult().getType())
            return m_op.getInput();
    return {};
}

/// simplifies rotate(x,0) to x
::mlir::OpFoldResult fhe::RotateOp::fold(FoldAdaptor adaptor)
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
            auto vec_size = getX().getType().dyn_cast<BatchedSecretType>().getSize();
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
/// simplifies add(x,0) and add(x) to x
::mlir::OpFoldResult fhe::AddOp::fold(FoldAdaptor adaptor)
{
    // add(x) -> x
    if (getX().size() == 1)
        return getX().front();

    // add(x,0,y) -> add(x,y)
    auto neutral_element = 0;
    SmallVector<Value> new_operands;
    bool any_change = false;
    for (auto v : getX())
    {
        bool omit = false;
        if (auto cst_op = v.getDefiningOp<fhe::ConstOp>())
        {
            if (auto dea = cst_op.getValue().dyn_cast_or_null<DenseElementsAttr>())
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
            else if (auto ia = cst_op.getValue().dyn_cast_or_null<IntegerAttr>())
            {
                if (ia.getInt() == neutral_element)
                    omit = true;
            }
            else if (auto fa = cst_op.getValue().dyn_cast_or_null<FloatAttr>())
            {
                if (fa.getValueAsDouble() == neutral_element)
                    omit = true;
            }
        }
        if (!omit)
            new_operands.push_back(v);
        else
            any_change = true;
    }

    if (any_change)
    {
        getXMutable().assign(new_operands);
        if (getX().size() > 1)
            return getResult();
        else
            return getX().front();
    }
    else
    {
        return nullptr;
    }
}
/// simplifies sub(x,0) and sub(x) to x
::mlir::OpFoldResult fhe::SubOp::fold(FoldAdaptor adaptor)
{
    // sub(x) -> x
    if (getX().size() == 1)
        return getX().front();

    // sub(x,0) -> x
    auto neutral_element = 0;
    SmallVector<Value> new_operands;
    bool any_change = false;
    for (auto v : getX())
    {
        bool omit = false;
        if (auto cst_op = v.getDefiningOp<fhe::ConstOp>())
        {
            if (auto dea = cst_op.getValue().dyn_cast_or_null<DenseElementsAttr>())
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
            else if (auto ia = cst_op.getValue().dyn_cast_or_null<IntegerAttr>())
            {
                if (ia.getInt() == neutral_element)
                    omit = true;
            }
            else if (auto fa = cst_op.getValue().dyn_cast_or_null<FloatAttr>())
            {
                if (fa.getValueAsDouble() == neutral_element)
                    omit = true;
            }
        }
        if (!omit)
            new_operands.push_back(v);
        else
            any_change = true;
    }

    if (any_change)
    {
        getXMutable().assign(new_operands);
        if (getX().size() > 1)
            return getResult();
        else
            return getX().front();
    }
    else
        return nullptr;
}

/// simplifies mul(x,1) and mul(x) to x, simplifies mul(x,0,y) to plaintext(0)
::mlir::OpFoldResult fhe::MultiplyOp::fold(FoldAdaptor adaptor)
{
    // mul(x) --> x
    if (getX().size() == 1)
        return getX().front();

    // mul(x,1,y) --> mul(x,y) and mul(x,0,y) --> 0
    SmallVector<Value> new_operands;
    bool any_change = false;
    for (auto v : getX())
    {
        bool omit = false;
        if (auto cst_op = v.getDefiningOp<fhe::ConstOp>())
        {
            if (auto dea = cst_op.getValue().dyn_cast_or_null<DenseElementsAttr>())
            {
                if (dea.size() == 1)
                {
                    if (dea.getElementType().isIntOrIndex())
                    {
                        if (dea.value_begin<const IntegerAttr>()->getInt() == 1)
                            omit = true;
                        else if (dea.value_begin<const IntegerAttr>()->getInt() == 0)
                        {
                            return *dea.value_begin<const IntegerAttr>();
                        }
                    }
                    else if (dea.getElementType().isIntOrFloat())
                    {
                        // because we've already excluded IntOrIndex, it must be float
                        if (dea.value_begin<const FloatAttr>()->getValueAsDouble() == 1)
                            omit = true;
                        else if (dea.value_begin<const FloatAttr>()->getValueAsDouble() == 0)
                        {
                            return *dea.value_begin<const FloatAttr>();
                        }
                    }
                }
            }
            else if (auto ia = cst_op.getValue().dyn_cast_or_null<IntegerAttr>())
            {
                if (ia.getInt() == 1)
                    omit = true;
                else if (ia.getInt() == 0)
                {
                    return ia;
                }
            }
            else if (auto fa = cst_op.getValue().dyn_cast_or_null<FloatAttr>())
            {
                if (fa.getValueAsDouble() == 1)
                    omit = true;
                else if (fa.getValueAsDouble() == 0)
                {
                    return fa;
                }
            }
        }
        else if (auto acst_op = v.getDefiningOp<arith::ConstantIntOp>())
        {
            if (acst_op.value() == 1)
                omit = true;
            else if (acst_op.value() == 0)
            {
                return acst_op.getValueAttr();
            }
        }
        else if (auto acst_op = v.getDefiningOp<arith::ConstantIndexOp>())
        {
            if (acst_op.value() == 1)
                omit = true;
            else if (acst_op.value() == 0)
            {
                return acst_op.getValueAttr();
            }
        }
        else if (auto acst_op = v.getDefiningOp<arith::ConstantFloatOp>())
        {
            if (acst_op.value().convertToDouble() == 1)
                omit = true;
            else if (acst_op.value().convertToDouble() == 0)
            {
                return acst_op.getValueAttr();
            }
        }
        if (!omit)
            new_operands.push_back(v);
        else
            any_change = true;
    }

    if (any_change)
    {
        getXMutable().assign(new_operands);
        if (getX().size() > 1)
            return getResult();
        else
            return getX().front();
    }
    else
    {
        return nullptr;
    }
}

/// Removes a combine if one of the operands completely covers the vector already:
///  e.g., fhe.combine(%8[0:63], %1) : !fhe.batched_secret<64 x f64>
///  can be simplified to %0 assuming the types match
::mlir::OpFoldResult fhe::CombineOp::fold(FoldAdaptor adaptor)
{
    for (size_t i = 0; i < getVectors().size(); ++i)
    {
        auto size = getType().dyn_cast<fhe::BatchedSecretType>().getSize();
        auto v = getVectors()[i];
        if (v.getType().dyn_cast<fhe::BatchedSecretType>().getSize() == size)
            // Check if the indices for this one cover everything
            if (auto iaa = getIndices()[i].dyn_cast_or_null<ArrayAttr>())
                // check it matches size
                if ((int)iaa.size() == size)
                    // Check there first one is 0
                    if (iaa[0].dyn_cast<IntegerAttr>().getValue() == 0)
                        return getVectors()[i];
    }
    return Value(); // Unsuccessful, could not fold
}

//===----------------------------------------------------------------------===//
// FHE dialect definitions
//===----------------------------------------------------------------------===//
#include "heco/IR/FHE/FHEDialect.cpp.inc"
void FHEDialect::initialize()
{
    // Registers all the Types into the FHEDialect class
    addTypes<
#define GET_TYPEDEF_LIST
#include "heco/IR/FHE/FHETypes.cpp.inc"
        >();

    // Registers all the Operations into the FHEDialect class
    addOperations<
#define GET_OP_LIST
#include "heco/IR/FHE/FHE.cpp.inc"
        >();
}