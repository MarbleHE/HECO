#include "heco/Passes/fhe2bfv/LowerFHEToBFV.h"
#include <iostream>
#include "heco/IR/BFV/BFVDialect.h"
#include "heco/IR/FHE/FHEDialect.h"
#include "heco/IR/Poly/PolyDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void LowerFHEToBFVPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<bfv::BFVDialect>();
    registry.insert<poly::PolyDialect>();
}

class BFVRotatePattern final : public OpConversionPattern<fhe::RotateOp>
{
protected:
    using OpConversionPattern<fhe::RotateOp>::typeConverter;

public:
    using OpConversionPattern<fhe::RotateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::RotateOp op, typename fhe::RotateOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        Value o = op.getOperand();
        Value materialized_operand;
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
            materialized_operand =
                typeConverter->materializeArgumentConversion(rewriter, op.getLoc(), operandDstType, o);
        else
            materialized_operand = o;

        rewriter.setInsertionPoint(op);
        auto poly_type = dstType.dyn_cast<bfv::CiphertextType>().getElementType();
        // TODO: MATCH PARAMETERS PROPERLY OR GET ACTUAL KEY FROM SOMEWHERE
        // auto key_type = bfv::GaloisKeysType::get(rewriter.getContext(), 0, 0, 0, poly_type);
        // auto keys = rewriter.create<bfv::LoadGaloisKeysOp>(op.getLoc(), key_type, "foo.glk", "glk.parms");
        rewriter.replaceOpWithNewOp<bfv::RotateOp>(op, dstType, materialized_operand, op.getI());
        return success();
    };
};

class BFVCombinePattern final : public OpConversionPattern<fhe::CombineOp>
{
protected:
    using OpConversionPattern<fhe::CombineOp>::typeConverter;

public:
    using OpConversionPattern<fhe::CombineOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::CombineOp op, typename fhe::CombineOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // TODO: Handle combining!
        return failure();
    };
};

class BFVConstPattern final : public OpConversionPattern<fhe::ConstOp>
{
protected:
    using OpConversionPattern<fhe::ConstOp>::typeConverter;

public:
    using OpConversionPattern<fhe::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::ConstOp op, typename fhe::ConstOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // TODO: Handle const ops!
        return failure();
    };
};

class BFVMaterializePattern final : public OpConversionPattern<fhe::MaterializeOp>
{
protected:
    using OpConversionPattern<fhe::MaterializeOp>::typeConverter;

public:
    using OpConversionPattern<fhe::MaterializeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::MaterializeOp op, typename fhe::MaterializeOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        llvm::SmallVector<Value> materialized_operands;
        auto o = op.getOperand();
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            assert(new_operand && "Type Conversion must not fail");
            materialized_operands.push_back(new_operand);
        }
        else
        {
            materialized_operands.push_back(o);
        }

        rewriter.replaceOpWithNewOp<bfv::MaterializeOp>(op, TypeRange(dstType), materialized_operands);
        return success();
    }
};

class BFVInsertPattern final : public OpConversionPattern<fhe::InsertOp>
{
protected:
    using OpConversionPattern<fhe::InsertOp>::typeConverter;

public:
    using OpConversionPattern<fhe::InsertOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::InsertOp op, typename fhe::InsertOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        rewriter.replaceOpWithNewOp<bfv::InsertOp>(
            op, TypeRange(dstType), materialized_operands[0], materialized_operands[1], op.getIAttr());
        return success();
    }
};

class BFVExtractPattern final : public OpConversionPattern<fhe::ExtractOp>
{
protected:
    using OpConversionPattern<fhe::ExtractOp>::typeConverter;

public:
    using OpConversionPattern<fhe::ExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::ExtractOp op, typename fhe::ExtractOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        llvm::SmallVector<Value> materialized_operands;
        auto o = op.getOperand();
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            assert(new_operand && "Type Conversion must not fail");
            materialized_operands.push_back(new_operand);
        }
        else
        {
            materialized_operands.push_back(o);
        }

        rewriter.replaceOpWithNewOp<bfv::ExtractOp>(op, TypeRange(dstType), materialized_operands[0], op.getIAttr());
        return success();
    }
};

/// Basic Pattern for operations without attributes.
template <typename OpType>
class BFVBasicPattern final : public OpConversionPattern<OpType>
{
protected:
    using OpConversionPattern<OpType>::typeConverter;

public:
    using OpConversionPattern<OpType>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        OpType op, typename OpType::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }
        // TODO: Handle ctxt-ptxt versions for all operations!

        // Additions
        if (std::is_same<OpType, fhe::AddOp>())
        {
            if (op.getNumOperands() > 2)
                rewriter.replaceOpWithNewOp<bfv::AddManyOp>(op, TypeRange(dstType), materialized_operands);
            else
                rewriter.replaceOpWithNewOp<bfv::AddOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Subtractions
        else if (std::is_same<OpType, fhe::SubOp>())
        {
            rewriter.replaceOpWithNewOp<bfv::SubOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Multiplications
        else if (std::is_same<OpType, fhe::MultiplyOp>())
        {
            if (op.getNumOperands() > 2)
            {
                // TODO: Handle ptxt in multiplies with more than two arguments
                rewriter.setInsertionPoint(op);
                auto poly_type = op.getType().template dyn_cast<bfv::CiphertextType>().getElementType();
                // TODO: MATCH PARAMETERS PROPERLY OR GET ACTUAL KEY FROM SOMEWHERE
                // auto key_type = bfv::RelinKeysType::get(rewriter.getContext(), 0, 0, 0, poly_type);
                // auto keys = rewriter.create<bfv::LoadRelinKeysOp>(op.getLoc(), key_type, "foo.rlk", "rlk.parms");
                auto new_op =
                    rewriter.replaceOpWithNewOp<bfv::MultiplyManyOp>(op, TypeRange(dstType), materialized_operands);
                rewriter.setInsertionPointAfter(new_op);
            }
            else
            {
                // TODO: Handle ptxt in first position, too!
                if (materialized_operands.size() == 2 && materialized_operands[1].getType().isa<bfv::PlaintextType>())
                    rewriter.replaceOpWithNewOp<bfv::MultiplyPlainOp>(op, TypeRange(dstType), materialized_operands);
                else
                    rewriter.replaceOpWithNewOp<bfv::MultiplyOp>(op, TypeRange(dstType), materialized_operands);
            }
            return success();
        }

        return failure();
    };
};

/// This is basically just boiler-plate code,
/// nothing here actually depends on the current dialect thats being converted.
class BFVFunctionConversionPattern final : public OpConversionPattern<func::FuncOp>
{
public:
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::FuncOp op, typename func::FuncOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // Compute the new signature of the function.
        TypeConverter::SignatureConversion signatureConversion(op.getFunctionType().getNumInputs());
        SmallVector<Type> newResultTypes;
        if (failed(typeConverter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return failure();
        if (typeConverter->convertSignatureArgs(op.getFunctionType().getInputs(), signatureConversion).failed())
            return failure();
        auto new_functype = FunctionType::get(getContext(), signatureConversion.getConvertedTypes(), newResultTypes);

        rewriter.startRootUpdate(op);
        op.setType(new_functype);
        for (auto it = op.getRegion().args_begin(); it != op.getRegion().args_end(); ++it)
        {
            auto arg = *it;
            auto oldType = arg.getType();
            auto newType = typeConverter->convertType(oldType);
            arg.setType(newType);
            if (newType != oldType)
            {
                rewriter.setInsertionPointToStart(&op.getBody().getBlocks().front());
                auto m_op = typeConverter->materializeSourceConversion(rewriter, arg.getLoc(), oldType, arg);
                arg.replaceAllUsesExcept(m_op, m_op.getDefiningOp());
            }
        }
        rewriter.finalizeRootUpdate(op);

        return success();
    }
};

/// More boiler-plate code that isn't dialect specific
class BFVReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        if (op->getNumOperands() != 1)
        {
            emitError(op->getLoc(), "Only single value returns support for now.");
            return failure();
        }
        auto dstType = this->getTypeConverter()->convertType(op->getOperandTypes().front());
        if (!dstType)
            return failure();

        rewriter.setInsertionPoint(op);
        auto materialized =
            typeConverter->materializeTargetConversion(rewriter, op.getLoc(), dstType, op.getOperands());
        // build a new return op
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized);

        return success();
    }
};

void LowerFHEToBFVPass::runOnOperation()
{
    // TODO: We still need to emit a pre-amble with an include statement
    //  this should refer to some "magic file" that also sets up keys/etc and our custom evaluator wrapper for now

    auto type_converter = TypeConverter();

    type_converter.addConversion([&](Type t) {
        if (t.isa<fhe::BatchedSecretType>())
            // TODO: How to find the correct type here?
            return std::optional<Type>(bfv::CiphertextType::get(
                &getContext(), 1, poly::PolynomialType::get(&getContext(), 2, true, 17, "parms.txt")));
        else if (t.isa<fhe::SecretType>())
            // TODO: How to find the correct type here?
            return std::optional<Type>(bfv::CiphertextType::get(
                &getContext(), 1, poly::PolynomialType::get(&getContext(), 2, true, 17, "parms.txt")));
        else if (t.isIntOrIndexOrFloat())
            // TODO: How to find the correct type here?
            return std::optional<Type>(bfv::PlaintextType::get(
                &getContext(), poly::PolynomialType::get(&getContext(), 2, true, 17, "parms.txt")));
        else
            return std::optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<bfv::CiphertextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>() || old_type.dyn_cast_or_null<fhe::SecretType>())
            {
                return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<bfv::PlaintextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.isIntOrIndexOrFloat())
            {
                return std::optional<Value>(builder.create<bfv::EncodeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<bfv::CiphertextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>() || old_type.dyn_cast_or_null<fhe::SecretType>())
            {
                return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<bfv::PlaintextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.isIntOrIndexOrFloat())
            {
                return std::optional<Value>(builder.create<bfv::EncodeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<bfv::CiphertextType>())
                return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, bst, vs));
        }
        else if (auto st = t.dyn_cast_or_null<fhe::SecretType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<bfv::CiphertextType>())
                return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, st, vs));
        }
        else if (t.isIntOrIndexOrFloat())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<bfv::PlaintextType>())
                return std::optional<Value>(builder.create<bfv::EncodeOp>(loc, t, vs));
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });

    ConversionTarget target(getContext());
    target.addIllegalDialect<fhe::FHEDialect>();
    target.addLegalDialect<bfv::BFVDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<fhe::ConstOp>();
    target.addLegalOp<fhe::CombineOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op) {
        auto fop = llvm::dyn_cast<func::FuncOp>(op);
        for (auto t : op->getOperandTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : op->getResultTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getInputs())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getResults())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        return true;
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](Operation *op) { return type_converter.isLegal(op->getOperandTypes()); });
    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<
        BFVFunctionConversionPattern, BFVReturnPattern, BFVBasicPattern<fhe::SubOp>, BFVBasicPattern<fhe::AddOp>,
        BFVBasicPattern<fhe::SubOp>, BFVBasicPattern<fhe::MultiplyOp>, BFVRotatePattern, BFVConstPattern,
        BFVMaterializePattern, BFVInsertPattern, BFVExtractPattern>(type_converter, patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}