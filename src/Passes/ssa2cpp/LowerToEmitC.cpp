#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "llvm/ADT/APSInt.h"

#include "abc/IR/FHE/FHEDialect.h"
#include "abc/Passes/ssa2cpp/LowerToEmitC.h"

using namespace mlir;

void LowerToEmitCPass::getDependentDialects(mlir::DialectRegistry &registry) const {
  registry.insert<mlir::emitc::EmitCDialect,
                  fhe::FHEDialect,
                  mlir::AffineDialect,
                  mlir::StandardOpsDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
}

class EmitCFunctionPattern final : public OpConversionPattern<FuncOp> {
 public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, typename FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Compute the new signature of the function.
    TypeConverter::SignatureConversion signatureConversion(op.getType().getNumInputs());
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter->convertTypes(op.getType().getResults(),
                                           newResultTypes)))
      return failure();
    if (typeConverter->convertSignatureArgs(op.getType().getInputs(), signatureConversion).failed())
      return failure();
    auto new_functype = FunctionType::get(getContext(), signatureConversion.getConvertedTypes(), newResultTypes);

    rewriter.startRootUpdate(op);
    op.setType(new_functype);
    for (auto it = op.getRegion().args_begin(); it!=op.getRegion().args_end(); ++it) {
      auto arg = *it;
      auto oldType = arg.getType();
      auto newType = typeConverter->convertType(oldType);
      arg.setType(newType);
      if (newType!=oldType) {
        rewriter.setInsertionPointToStart(&op.getBody().getBlocks().front());
        auto m_op = typeConverter->materializeSourceConversion(rewriter, arg.getLoc(), oldType, arg);
        arg.replaceAllUsesExcept(m_op, m_op.getDefiningOp());
      }
    }
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

class EmitCReturnPattern final : public OpConversionPattern<ReturnOp> {
 public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, typename ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands()!=1) {
      emitError(op->getLoc(), "Only single value returns support for now.");
      return failure();
    }
    auto dstType = this->getTypeConverter()->convertType(op->getOperandTypes().front());
    if (!dstType)
      return failure();
    if (auto bst = dstType.dyn_cast_or_null<emitc::OpaqueType>()) {
      rewriter.setInsertionPoint(op);
      auto materialized = typeConverter->materializeTargetConversion(rewriter,
                                                                     op.getLoc(),
                                                                     dstType,
                                                                     op.operands());
      // build a new return op
      rewriter.replaceOpWithNewOp<ReturnOp>(op, materialized);

    } // else do nothing
    return success();
  }
};

void LowerToEmitCPass::runOnOperation() {

  auto type_converter = TypeConverter();

  type_converter.addConversion([&](Type t) {
    if (t.isa<fhe::BatchedSecretType>() || t.isa<fhe::SecretType>())
      return llvm::Optional<Type>(emitc::OpaqueType::get(&getContext(), "seal::Ciphertext"));
    else
      return llvm::Optional<Type>(t);
  });
  type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
    if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>()) {
      assert(!vs.empty() && ++vs.begin()==vs.end() && "currently can only materalize single values");
      auto old_type = vs.front().getType();
      if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>() || old_type.dyn_cast_or_null<fhe::SecretType>())
        if (ot.getValue().str()=="seal::Ciphertext")
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, ot, vs));
    }
    return llvm::Optional<Value>(llvm::None); // would instead like to signal NO other conversions can be tried
  });
  type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
    if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>()) {
      assert(!vs.empty() && ++vs.begin()==vs.end() && "currently can only materalize single values");
      auto old_type = vs.front().getType();
      if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>() || old_type.dyn_cast_or_null<fhe::SecretType>())
        if (ot.getValue().str()=="seal::Ciphertext")
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, ot, vs));
    }
    return llvm::Optional<Value>(llvm::None); // would instead like to signal NO other conversions can be tried
  });
  type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
    if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>()) {
      assert(!vs.empty() && ++vs.begin()==vs.end() && "currently can only materalize single values");
      auto old_type = vs.front().getType();
      if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
        if (ot.getValue().str()=="seal::Ciphertext")
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, bst, vs));
    }
    return llvm::Optional<Value>(llvm::None); // would instead like to signal NO other conversions can be tried
  });

  ConversionTarget target(getContext());
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addLegalOp<fhe::MaterializeOp>();
  target.addDynamicallyLegalOp<FuncOp>([&](Operation *op) {
    auto fop = llvm::dyn_cast<FuncOp>(op);
    for (auto t: op->getOperandTypes()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    for (auto t: op->getResultTypes()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    for (auto t: fop.getType().getInputs()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    for (auto t: fop.getType().getResults()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<ReturnOp>([&](Operation *op) {
    for (auto t: op->getOperandTypes()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<
      EmitCReturnPattern,
      EmitCFunctionPattern
  >(type_converter, patterns.getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

}