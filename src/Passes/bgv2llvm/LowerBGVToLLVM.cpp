#include "heco/Passes/bgv2llvm/LowerBGVToLLVM.h"
#include "heco/IR/BGV/BGVDialect.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void LowerBGVToLLVMPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<emitc::EmitCDialect, mlir::AffineDialect, func::FuncDialect, scf::SCFDialect>();
}

void LowerBGVToLLVMPass::runOnOperation()
{
    // TODO: We still need to emit a pre-amble with an include statement
    //  this should refer to some "magic file" that also sets up keys/etc and our custom evaluator wrapper for now

    auto type_converter = TypeConverter();

    type_converter.addConversion([&](Type t) {
        if (t.isa<bgv::CiphertextType>())
            return llvm::Optional<Type>(emitc::OpaqueType::get(&getContext(), "seal::Ciphertext"));
        else
            return llvm::Optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<bgv::CiphertextType>())
                if (ot.getValue().str() == "seal::Ciphertext")
                    return llvm::Optional<Value>(builder.create<bgv::MaterializeOp>(loc, ot, vs));
        }
        return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<bgv::CiphertextType>())
                if (ot.getValue().str() == "seal::Ciphertext")
                    return llvm::Optional<Value>(builder.create<bgv::MaterializeOp>(loc, ot, vs));
        }
        return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<bgv::CiphertextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "seal::Ciphertext")
                    return llvm::Optional<Value>(builder.create<bgv::MaterializeOp>(loc, bst, vs));
        }
        return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */
    });

    ConversionTarget target(getContext());
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();
    target.addLegalOp<bgv::MaterializeOp>();
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
    // target.addDynamicallyLegalOp<bgv::SubOp>([&](Operation *op)
    //                                          { return type_converter.isLegal(op->getOperandTypes()) &&
    //                                          type_converter.isLegal(op->getResultTypes()); });
    // target.addDynamicallyLegalOp<bgv::AddOp>([&](Operation *op)
    //                                          { return type_converter.isLegal(op->getOperandTypes()) &&
    //                                          type_converter.isLegal(op->getResultTypes()); });
    // target.addDynamicallyLegalOp<bgv::MultiplyOp>([&](Operation *op)
    //                                               { return type_converter.isLegal(op->getOperandTypes()) &&
    //                                               type_converter.isLegal(op->getResultTypes()); });

    mlir::RewritePatternSet patterns(&getContext());
    // patterns.add<
    //     EmitCReturnPattern,
    //     EmitCArithmeticPattern<fhe::SubOp>,
    //     EmitCArithmeticPattern<fhe::AddOp>,
    //     EmitCArithmeticPattern<fhe::MultiplyOp>,
    //     EmitCRotatePattern,
    //     EmitCCombinePattern,
    //     EmitCFunctionPattern>(type_converter, patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}