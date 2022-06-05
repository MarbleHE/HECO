#include <iostream>
#include <memory>
#include "heco/Passes/ssa2ssa/Tensor2BatchedSecret.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "heco/IR/FHE/FHEDialect.h"

using namespace mlir;

void Tensor2BatchedSecretPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
  registry.insert<fhe::FHEDialect,
                  mlir::AffineDialect,
                  mlir::func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
}

class ExtractPattern final : public OpConversionPattern<tensor::ExtractOp>
{
public:
  using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, typename tensor::ExtractOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    int size = -28;
    auto tt = op.tensor().getType().dyn_cast<TensorType>();
    if (tt.hasStaticShape() && tt.getShape().size() == 1)
      size = tt.getShape().front();

    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return failure();
    if (auto st = dstType.dyn_cast_or_null<fhe::SecretType>())
    {
      auto batchedSecret = typeConverter->materializeTargetConversion(rewriter,
                                                                      op.tensor().getLoc(),
                                                                      fhe::BatchedSecretType::get(getContext(),
                                                                                                  st.getPlaintextType(),
                                                                                                  size),
                                                                      op.tensor());
      // TODO: Support  tensor.extract indices format in fhe.extract,too
      auto cOp = op.indices().front().getDefiningOp<arith::ConstantOp>();
      if (!cOp)
      {
        emitError(op.getLoc(),
                  "tensor2fhe requires all tensor.extract indices used with tensors of fhe.secret to be constant!");
        return failure();
      }
      auto indexAttr = cOp.getValue().cast<IntegerAttr>();
      rewriter.template replaceOpWithNewOp<fhe::ExtractOp>(op, dstType, batchedSecret, indexAttr);
    } // else do nothing
    return success();
  }
};

class InsertPattern final : public OpConversionPattern<tensor::InsertOp>
{
public:
  using OpConversionPattern<tensor::InsertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertOp op, typename tensor::InsertOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return failure();
    if (auto bst = dstType.dyn_cast_or_null<fhe::BatchedSecretType>())
    {
      auto batchedSecret = typeConverter->materializeTargetConversion(rewriter,
                                                                      op.dest().getLoc(),
                                                                      bst,
                                                                      op.dest());
      // TODO: Support  tensor.extract indices format in fhe.extract,too
      auto cOp = op.indices().front().getDefiningOp<arith::ConstantOp>();
      if (!cOp)
      {
        emitError(op.getLoc(),
                  "tensor2fhe requires all tensor.insert indices used with tensors of fhe.secret to be constant!");
        return failure();
      }
      auto indexAttr = cOp.getValue().cast<IntegerAttr>();
      rewriter.template replaceOpWithNewOp<fhe::InsertOp>(op, dstType, op.scalar(), batchedSecret, indexAttr);
    } // else do nothing
    return success();
  }
};

class ReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    if (op.getNumOperands() != 1)
    {
      emitError(op.getLoc(), "Currently only single value return operations are supported.");
      return failure();
    }
    auto dstType = this->getTypeConverter()->convertType(op.getOperandTypes().front());
    if (!dstType)
      return failure();
    if (auto bst = dstType.dyn_cast_or_null<fhe::BatchedSecretType>())
    {
      rewriter.setInsertionPoint(op);
      auto batchedSecret = typeConverter->materializeTargetConversion(rewriter,
                                                                      op.getLoc(),
                                                                      dstType,
                                                                      op.operands().front());
      rewriter.template replaceOpWithNewOp<func::ReturnOp>(op, batchedSecret);

    } // else do nothing
    return success();
  }
};

class FunctionPattern final : public OpConversionPattern<FuncOp>
{
public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, typename FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {

    // Compute the new signature of the function.
    TypeConverter::SignatureConversion signatureConversion(op.getFunctionType().getNumInputs());
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter->convertTypes(op.getFunctionType().getResults(),
                                           newResultTypes)))
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

void Tensor2BatchedSecretPass::runOnOperation()
{

  auto type_converter = TypeConverter();

  type_converter.addConversion([&](TensorType t)
                               {
    int size = -155;
    if (t.hasStaticShape() && t.getShape().size()==1) {
      size = t.getShape().front();
    }
    if (auto st = t.getElementType().dyn_cast_or_null<fhe::SecretType>()) {
      return llvm::Optional<Type>(fhe::BatchedSecretType::get(&getContext(), st.getPlaintextType(), size));
    } else {
      return llvm::Optional<Type>(t);
    } });
  type_converter.addConversion([](Type t)
                               {
    if (t.isa<TensorType>())
      return llvm::Optional<Type>(llvm::None);
    else
      return llvm::Optional<Type>(t); });
  type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc)
                                          {
    if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>()) {
      assert(!vs.empty() && ++vs.begin()==vs.end() && "currently can only materalize single values");
      auto old_type = vs.front().getType();
      if (auto tt = old_type.dyn_cast_or_null<TensorType>())
        if (auto st = tt.getElementType().dyn_cast_or_null<fhe::SecretType>())
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, bst, vs));
    }
    return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */ });
  type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc)
                                            {
    if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>()) {
      assert(!vs.empty() && ++vs.begin()==vs.end() && "currently can only materalize single values");
      auto old_type = vs.front().getType();
      if (auto tt = old_type.dyn_cast_or_null<TensorType>())
        if (auto st = tt.getElementType().dyn_cast_or_null<fhe::SecretType>())
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, bst, vs));
    }
    return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */ });
  type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc)
                                          {
    if (auto tt = t.dyn_cast_or_null<TensorType>()) {
      assert(!vs.empty() && ++vs.begin()==vs.end() && "currently can only materalize single values");
      auto old_type = vs.front().getType();
      if (auto bst = old_type.dyn_cast_or_null<fhe::BatchedSecretType>())
        if (tt.getElementType()==bst.getCorrespondingSecretType())
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, tt, vs));
    }
    return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */ });

  ConversionTarget target(getContext());
  target.addLegalDialect<fhe::FHEDialect>();
  target.addDynamicallyLegalDialect<tensor::TensorDialect>([&](Operation *op)
                                                           { return type_converter.isLegal(op); });
  target.addDynamicallyLegalOp<FuncOp>([&](Operation *op)
                                       {
    auto fop = llvm::dyn_cast<FuncOp>(op);
    for (auto t: op->getOperandTypes()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    for (auto t: op->getResultTypes()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    for (auto t: fop.getFunctionType().getInputs()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    for (auto t: fop.getFunctionType().getResults()) {
      if (!type_converter.isLegal(t))
        return false;
    }
    return true; });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](Operation *op)
                                               { return type_converter.isLegal(op->getOperandTypes()); });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<
      ExtractPattern,
      InsertPattern,
      ReturnPattern,
      FunctionPattern>(type_converter, patterns.getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}