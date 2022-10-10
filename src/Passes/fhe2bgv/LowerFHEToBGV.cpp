#include "heco/Passes/fhe2bgv/LowerFHEToBGV.h"
#include <iostream>
#include "heco/IR/BGV/BGVDialect.h"
#include "heco/IR/FHE/FHEDialect.h"
#include "heco/IR/Poly/PolyDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void LowerFHEToBGVPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<bgv::BGVDialect>();
    registry.insert<poly::PolyDialect>();
}

class BGVRotatePattern final : public OpConversionPattern<fhe::RotateOp>
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
        auto poly_type = dstType.dyn_cast<bgv::CiphertextType>().getElementType();
        // TODO: MATCH PARAMETERS PROPERLY OR GET ACTUAL KEY FROM SOMEWHERE
        auto key_type = bgv::GaloisKeysType::get(rewriter.getContext(), 0, 0, 0, poly_type);
        auto keys = rewriter.create<bgv::LoadGaloisKeysOp>(op.getLoc(), key_type, "foo.glk", "glk.parms");
        rewriter.replaceOpWithNewOp<bgv::RotateOp>(op, dstType, materialized_operand, keys, op.i());
        return success();
    };
};

class BGVCombinePattern final : public OpConversionPattern<fhe::CombineOp>
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

class BGVConstPattern final : public OpConversionPattern<fhe::ConstOp>
{
protected:
    using OpConversionPattern<fhe::ConstOp>::typeConverter;

public:
    using OpConversionPattern<fhe::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::ConstOp op, typename fhe::ConstOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // TODO: Handle combining!
        return failure();
    };
};

/// Basic Pattern for operations without attributes.
template <typename OpType>
class BGVBasicPattern final : public OpConversionPattern<OpType>
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

        // Additions
        if (std::is_same<OpType, fhe::AddOp>())
        {
            if (op.getNumOperands() > 2)
                rewriter.replaceOpWithNewOp<bgv::AddManyOp>(op, TypeRange(dstType), materialized_operands);
            else
                rewriter.replaceOpWithNewOp<bgv::AddOp>(op, TypeRange(dstType), materialized_operands[0]);
            return success();
        }
        // Subtractions
        else if (std::is_same<OpType, fhe::SubOp>())
        {
            rewriter.replaceOpWithNewOp<bgv::SubOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Multiplications
        else if (std::is_same<OpType, fhe::MultiplyOp>())
        {
            if (op.getNumOperands() > 2)
            {
                rewriter.setInsertionPoint(op);
                auto poly_type = op.getType().template dyn_cast<bgv::CiphertextType>().getElementType();
                // TODO: MATCH PARAMETERS PROPERLY OR GET ACTUAL KEY FROM SOMEWHERE
                auto key_type = bgv::RelinKeysType::get(rewriter.getContext(), 0, 0, 0, poly_type);
                auto keys = rewriter.create<bgv::LoadRelinKeysOp>(op.getLoc(), key_type, "foo.rlk", "rlk.parms");
                auto new_op = rewriter.replaceOpWithNewOp<bgv::MultiplyManyOp>(
                    op, TypeRange(dstType), materialized_operands, keys);
                rewriter.setInsertionPointAfter(new_op);
            }
            else
                rewriter.replaceOpWithNewOp<bgv::MultiplyOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }

        return failure();
    };
};

/// This is basically just boiler-plate code,
/// nothing here actually depends on the current dialect thats being converted.
class BGVFunctionConversionPattern final : public OpConversionPattern<func::FuncOp>
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
class BGVReturnPattern final : public OpConversionPattern<func::ReturnOp>
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
        auto materialized = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), dstType, op.operands());
        // build a new return op
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized);

        return success();
    }
};

void LowerFHEToBGVPass::runOnOperation()
{
    // TODO: We still need to emit a pre-amble with an include statement
    //  this should refer to some "magic file" that also sets up keys/etc and our custom evaluator wrapper for now

    auto type_converter = TypeConverter();

    type_converter.addConversion([&](Type t) {
        if (t.isa<fhe::BatchedSecretType>())
            // TODO: How to find the correct type here?
            return llvm::Optional<Type>(bgv::CiphertextType::get(
                &getContext(), 1, poly::PolynomialType::get(&getContext(), 2, true, 17, "parms.txt")));
        else
            return llvm::Optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<bgv::CiphertextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>())
            {
                return llvm::Optional<Value>(builder.create<bgv::MaterializeOp>(loc, ot, vs));
            }
        }
        return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<bgv::CiphertextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>())
            {
                return llvm::Optional<Value>(builder.create<bgv::MaterializeOp>(loc, ot, vs));
            }
        }
        return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<bgv::CiphertextType>())
                return llvm::Optional<Value>(builder.create<bgv::MaterializeOp>(loc, bst, vs));
        }
        return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */
    });

    ConversionTarget target(getContext());
    target.addIllegalDialect<fhe::FHEDialect>();
    // target.addLegalOp<fhe::ConstOp>(); // TODO: Remove this temp override and actually deal with const!
    // target.addLegalOp<fhe::MaterializeOp>(); // TODO: Remove this temp override and actually deal with materialize!
    target.addLegalDialect<bgv::BGVDialect>();
    target.addLegalOp<ModuleOp>();
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
        BGVFunctionConversionPattern, BGVReturnPattern, BGVBasicPattern<fhe::SubOp>, BGVBasicPattern<fhe::AddOp>,
        BGVBasicPattern<fhe::SubOp>, BGVBasicPattern<fhe::MultiplyOp>, BGVRotatePattern, BGVConstPattern>(
        type_converter, patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}