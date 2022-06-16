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

/// More boiler-plate code that isn't really dialect specific (except for one mention of the target type)
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
        if (auto bst = dstType.dyn_cast_or_null<fhe::BatchedSecretType>())
        {
            rewriter.setInsertionPoint(op);
            auto materialized =
                typeConverter->materializeTargetConversion(rewriter, op.getLoc(), dstType, op.operands());
            // build a new return op
            rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized);

        } // else do nothing
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
        if (auto ot = t.dyn_cast_or_null<fhe::BatchedSecretType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<bgv::CiphertextType>())
            {
                return llvm::Optional<Value>(builder.create<bgv::MaterializeOp>(loc, ot, vs));
            }
        }
        return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<fhe::BatchedSecretType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<bgv::CiphertextType>())
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

    patterns.add<BGVFunctionConversionPattern, BGVReturnPattern>
        //     EmitCBasicPattern<bgv::SubOp>, EmitCBasicPattern<bgv::SubPlainOp>, EmitCBasicPattern<bgv::AddOp>,
        //     EmitCBasicPattern<bgv::AddPlainOp>, EmitCBasicPattern<bgv::AddManyOp>,
        //     EmitCBasicPattern<bgv::MultiplyOp>, EmitCBasicPattern<bgv::MultiplyPlainOp>,
        //     EmitCBasicPattern<bgv::MultiplyManyOp>, EmitCBasicPattern<bgv::RelinearizeOp>,
        //     EmitCBasicPattern<bgv::ModswitchToOp>, EmitCLoadPattern<bgv::LoadCtxtOp>,
        //     EmitCLoadPattern<bgv::LoadPtxtOp>, EmitCLoadPattern<bgv::LoadPublicKeyOp>,
        //     EmitCLoadPattern<bgv::LoadRelinKeysOp>, EmitCLoadPattern<bgv::LoadGaloisKeysOp>, EmitCRotatePattern,
        //     EmitCSinkPattern>
        (type_converter, patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}