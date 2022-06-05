#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "llvm/ADT/APSInt.h"

#include "heco/IR/FHE/FHEDialect.h"
#include "heco/Passes/bgv2emitc/LowerBGVToEmitC.h"

using namespace mlir;

void LowerToEmitCPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
  registry.insert<mlir::emitc::EmitCDialect,
                  fhe::FHEDialect,
                  mlir::AffineDialect,
                  func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
}

class EmitCFunctionPattern final : public OpConversionPattern<func::FuncOp>
{
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, typename func::FuncOp::Adaptor adaptor,
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

class EmitCRotatePattern final : public OpConversionPattern<fhe::RotateOp>
{
public:
  using OpConversionPattern<fhe::RotateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(fhe::RotateOp op, typename fhe::RotateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {

    rewriter.setInsertionPoint(op);

    auto dstType = typeConverter->convertType(op.getType());
    if (!dstType)
      return failure();

    // Materialize the operands if necessary
    Value x = op.x();
    auto xDstType = typeConverter->convertType(x.getType());
    if (!xDstType)
      return failure();
    if (x.getType() != xDstType)
    {
      auto new_operand = typeConverter->materializeTargetConversion(rewriter,
                                                                    op.getLoc(),
                                                                    xDstType,
                                                                    x);
      x = new_operand;
    }

    // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with seal's API)
    auto aa = ArrayAttr::get(getContext(), {IntegerAttr::get(IndexType::get(getContext()),
                                                             0), // means "first operand"
                                            rewriter.getSI32IntegerAttr(op.i())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, TypeRange(dstType),
                                               llvm::StringRef("evaluator.rotate"),
                                               aa,
                                               ArrayAttr(),
                                               ValueRange(x));

    return success();
  }
};

class EmitCCombinePattern final : public OpConversionPattern<fhe::CombineOp>
{
public:
  using OpConversionPattern<fhe::CombineOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(fhe::CombineOp op, typename fhe::CombineOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {

    rewriter.setInsertionPoint(op);

    auto dstType = typeConverter->convertType(op.getType());
    if (!dstType)
      return failure();

    // Materialize the operands if necessary
    llvm::SmallVector<Value> new_operands;
    for (auto x : op.vectors())
    {
      auto xDstType = typeConverter->convertType(x.getType());
      if (!xDstType)
        return failure();
      if (x.getType() != xDstType)
      {
        auto new_operand = typeConverter->materializeTargetConversion(rewriter,
                                                                      op.getLoc(),
                                                                      xDstType,
                                                                      x);
        new_operands.push_back(new_operand);
      }
      else
      {
        new_operands.push_back(x);
      }
    }

    // TODO: "REAL" lowering of fhe.combine should happen through some type of BGV dialect
    //  For now we just call a "combine" function in our custom evaluator and let it handle the "lowering"

    // We pass the indices as initializer lists, so we need to build those first
    std::vector<std::string> init_lists;
    for (auto a : op.indices())
    {
      if (auto ia = a.dyn_cast<IntegerAttr>())
      {
        init_lists.emplace_back("{" + std::to_string(ia.getInt()) + "}");
      }
      else if (auto aa = a.dyn_cast<ArrayAttr>())
      {
        std::string s = "{";
        bool omit_comma = true;
        for (auto ia : aa.getAsRange<IntegerAttr>())
        {
          if (!omit_comma)
          {
            s += ", ";
          }
          else
          {
            omit_comma = false;
          }
          s += std::to_string(ia.getInt());
        }
        s += "}";
        init_lists.push_back(s);
      }
      else if (auto sa = a.dyn_cast<StringAttr>())
      {
        assert(sa.getValue() == "all");
        init_lists.emplace_back("{}");
      }
    }

    // Now interleave operands and initializer lists
    SmallVector<Attribute> params;
    for (size_t i = 0; i < op.vectors().size(); ++i)
    {
      params.push_back(rewriter.getIndexAttr(i));
      params.push_back(emitc::OpaqueAttr::get(rewriter.getContext(), init_lists[i]));
    }

    auto aa = ArrayAttr::get(getContext(), params);
    rewriter.replaceOpWithNewOp<emitc::CallOp>(op,
                                               dstType,
                                               llvm::StringRef("evaluator.combine"),
                                               aa,
                                               ArrayAttr(), // no template args
                                               new_operands);

    return success();
  }
};

template <typename OpType>
class EmitCArithmeticPattern final : public OpConversionPattern<OpType>
{
protected:
  using OpConversionPattern<OpType>::typeConverter;

public:
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
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

        auto new_operand = typeConverter->materializeTargetConversion(rewriter,
                                                                      op.getLoc(),
                                                                      operandDstType,
                                                                      o);
        materialized_operands.push_back(new_operand);
      }
      else
      {
        materialized_operands.push_back(o);
      }
    }

    // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with seal's API)
    std::string op_str;
    if (std::is_same<OpType, fhe::SubOp>())
      op_str = "sub";
    if (std::is_same<OpType, fhe::AddOp>())
      op_str = "add";
    if (std::is_same<OpType, fhe::MultiplyOp>())
      op_str = "multiply";

    if (op.getNumOperands() > 2)
      op_str = op_str + "_many";

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, TypeRange(dstType),
                                               llvm::StringRef("evaluator." + op_str),
                                               ArrayAttr(),
                                               ArrayAttr(),
                                               materialized_operands);

    return success();
  }
};

class EmitCReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    if (op->getNumOperands() != 1)
    {
      emitError(op->getLoc(), "Only single value returns support for now.");
      return failure();
    }
    auto dstType = this->getTypeConverter()->convertType(op->getOperandTypes().front());
    if (!dstType)
      return failure();
    if (auto bst = dstType.dyn_cast_or_null<emitc::OpaqueType>())
    {
      rewriter.setInsertionPoint(op);
      auto materialized = typeConverter->materializeTargetConversion(rewriter,
                                                                     op.getLoc(),
                                                                     dstType,
                                                                     op.operands());
      // build a new return op
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized);

    } // else do nothing
    return success();
  }
};

void LowerToEmitCPass::runOnOperation()
{

  // TODO: We still need to emit a pre-amble with an include statement
  //  this should refer to some "magic file" that also sets up keys/etc and our custom evaluator wrapper for now

  auto type_converter = TypeConverter();

  type_converter.addConversion([&](Type t)
                               {
    if (t.isa<fhe::BatchedSecretType>() || t.isa<fhe::SecretType>())
      return llvm::Optional<Type>(emitc::OpaqueType::get(&getContext(), "seal::Ciphertext"));
    else
      return llvm::Optional<Type>(t); });
  type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc)
                                          {
    if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>()) {
      assert(!vs.empty() && ++vs.begin()==vs.end() && "currently can only materalize single values");
      auto old_type = vs.front().getType();
      if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>() || old_type.dyn_cast_or_null<fhe::SecretType>())
        if (ot.getValue().str()=="seal::Ciphertext")
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, ot, vs));
    }
    return llvm::Optional<Value>(llvm::None);  /* would instead like to signal NO other conversions can be tried */ });
  type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc)
                                            {
    if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>())
    {
      assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
      auto old_type = vs.front().getType();
      if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>() || old_type.dyn_cast_or_null<fhe::SecretType>())
        if (ot.getValue().str() == "seal::Ciphertext")
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, ot, vs));
    }
    return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */ });
  type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc)
                                          {
    if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>())
    {
      assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
      auto old_type = vs.front().getType();
      if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
        if (ot.getValue().str() == "seal::Ciphertext")
          return llvm::Optional<Value>(builder.create<fhe::MaterializeOp>(loc, bst, vs));
    }
    return llvm::Optional<Value>(llvm::None); /* would instead like to signal NO other conversions can be tried */ });

  ConversionTarget target(getContext());
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addLegalOp<fhe::MaterializeOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op)
                                             {
    auto fop = llvm::dyn_cast<func::FuncOp>(op);
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
  target.addDynamicallyLegalOp<fhe::SubOp>([&](Operation *op)
                                           { return type_converter.isLegal(op->getOperandTypes()) && type_converter.isLegal(op->getResultTypes()); });
  target.addDynamicallyLegalOp<fhe::AddOp>([&](Operation *op)
                                           { return type_converter.isLegal(op->getOperandTypes()) && type_converter.isLegal(op->getResultTypes()); });
  target.addDynamicallyLegalOp<fhe::MultiplyOp>([&](Operation *op)
                                                { return type_converter.isLegal(op->getOperandTypes()) && type_converter.isLegal(op->getResultTypes()); });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<
      EmitCReturnPattern,
      EmitCArithmeticPattern<fhe::SubOp>,
      EmitCArithmeticPattern<fhe::AddOp>,
      EmitCArithmeticPattern<fhe::MultiplyOp>,
      EmitCRotatePattern,
      EmitCCombinePattern,
      EmitCFunctionPattern>(type_converter, patterns.getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}