#include "heco/Passes/bfv2emitc/LowerBFVToEmitC.h"
#include "heco/IR/BFV/BFVDialect.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void LowerBFVToEmitCPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<mlir::emitc::EmitCDialect>();
}

/// Convert Rotate
class EmitCRotatePattern final : public OpConversionPattern<bfv::RotateOp>
{
public:
    using OpConversionPattern<bfv::RotateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        bfv::RotateOp op, typename bfv::RotateOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands if necessary
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op->getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with
        // seal's API)
        auto aa = ArrayAttr::get(
            getContext(), { IntegerAttr::get(
                                IndexType::get(getContext()),
                                0), // means "first operand"
                            rewriter.getSI32IntegerAttr(op.getOffset()) });

        rewriter.replaceOpWithNewOp<emitc::CallOp>(
            op, TypeRange(dstType), llvm::StringRef("evaluator_rotate"), aa, ArrayAttr(), materialized_operands);

        return success();
    }
};

/// Convert Relin
class EmitCRelinPattern final : public OpConversionPattern<bfv::RelinearizeOp>
{
public:
    using OpConversionPattern<bfv::RelinearizeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        bfv::RelinearizeOp op, typename bfv::RelinearizeOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands if necessary
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op->getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with
        // seal's API)
        rewriter.replaceOpWithNewOp<emitc::CallOp>(
            op, TypeRange(dstType), llvm::StringRef("evaluator_relinearize"), ArrayAttr(), ArrayAttr(),
            materialized_operands);

        return success();
    }
};

/// Convert Encode
class EmitCEncodePattern final : public OpConversionPattern<bfv::EncodeOp>
{
public:
    using OpConversionPattern<bfv::EncodeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        bfv::EncodeOp op, typename bfv::EncodeOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands if necessary
        llvm::SmallVector<Value> materialized_operands;
        Value o = op->getOperand(0);
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            materialized_operands.push_back(new_operand);
        }
        else
        {
            materialized_operands.push_back(o);
        }

        // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with
        // seal's API)
        rewriter.replaceOpWithNewOp<emitc::CallOp>(
            op, TypeRange(dstType), llvm::StringRef("evaluator_encode"), ArrayAttr(), ArrayAttr(),
            materialized_operands);

        return success();
    }
};

/// Convert Decode
class EmitCDecodePattern final : public OpConversionPattern<bfv::DecodeOp>
{
public:
    using OpConversionPattern<bfv::DecodeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        bfv::DecodeOp op, typename bfv::DecodeOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands if necessary
        llvm::SmallVector<Value> materialized_operands;
        Value o = op->getOperand(0);
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            materialized_operands.push_back(new_operand);
        }
        else
        {
            materialized_operands.push_back(o);
        }

        // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with
        // seal's API)
        rewriter.replaceOpWithNewOp<emitc::CallOp>(
            op, TypeRange(dstType), llvm::StringRef("evaluator_decode"), ArrayAttr(), ArrayAttr(),
            materialized_operands);

        return success();
    }
};

/// Convert Sink
class EmitCSinkPattern final : public OpConversionPattern<bfv::SinkOp>
{
public:
    using OpConversionPattern<bfv::SinkOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        bfv::SinkOp op, typename bfv::SinkOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        // Materialize the operands where necessary
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op->getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized_operands);

        return success();
    }
};

/// Basic Pattern for operations without attributes.
template <typename OpType>
class EmitCBasicPattern final : public OpConversionPattern<OpType>
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
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with
        // seal's API)
        std::string op_str = "";
        if (std::is_same<OpType, bfv::SubOp>())
            op_str = "sub";
        else if (std::is_same<OpType, bfv::SubPlainOp>())
            op_str = "sub_plain";
        else if (std::is_same<OpType, bfv::AddOp>())
            op_str = "add";
        else if (std::is_same<OpType, bfv::AddPlainOp>())
            op_str = "add_plain";
        else if (std::is_same<OpType, bfv::AddManyOp>())
            op_str = "add_many";
        else if (std::is_same<OpType, bfv::MultiplyOp>())
            op_str = "multiply";
        else if (std::is_same<OpType, bfv::MultiplyPlainOp>())
            op_str = "multiply_plain";
        else if (std::is_same<OpType, bfv::MultiplyManyOp>())
            op_str = "multiply_plain";
        else if (std::is_same<OpType, bfv::ModswitchToOp>())
            op_str = "modswitch_to";
        else
            return failure();

        // For the _many ops, we need to build a vector of the arguments!
        if (std::is_same<OpType, bfv::AddManyOp>() || std::is_same<OpType, bfv::MultiplyManyOp>())
        {
            auto template_array = ArrayAttr::get(
                rewriter.getContext(), { emitc::OpaqueAttr::get(rewriter.getContext(), "seal::Ciphertext") });
            emitc::CallOp v = rewriter.create<emitc::CallOp>(
                op.getLoc(), TypeRange(emitc::OpaqueType::get(rewriter.getContext(), "std::vector<seal::Ciphertext>")),
                llvm::StringRef("std::vector"), ArrayAttr(), template_array, ValueRange());

            size_t num_operands =
                std::is_same<OpType, bfv::AddManyOp>() ? op->getNumOperands() : op->getNumOperands() - 1;
            for (size_t i = 0; i < num_operands; ++i)
            {
                rewriter.create<emitc::CallOp>(
                    op.getLoc(), TypeRange(), llvm::StringRef("insert"), ArrayAttr(), ArrayAttr(),
                    ValueRange({ v.getResult(0), materialized_operands[i] }));
            }

            if (std::is_same<OpType, bfv::AddManyOp>())
            {
                rewriter.replaceOpWithNewOp<emitc::CallOp>(
                    op, TypeRange(dstType), llvm::StringRef("evaluator_" + op_str), ArrayAttr(), ArrayAttr(),
                    ValueRange{ (v.getResult(0)) });
            }
            else
            {
                rewriter.replaceOpWithNewOp<emitc::CallOp>(
                    op, TypeRange(dstType), llvm::StringRef("evaluator_" + op_str), ArrayAttr(), ArrayAttr(),
                    ValueRange{ (v.getResult(0)), materialized_operands.back() });
            }
        }
        else
        {
            rewriter.replaceOpWithNewOp<emitc::CallOp>(
                op, TypeRange(dstType), llvm::StringRef("evaluator_" + op_str), ArrayAttr(), ArrayAttr(),
                materialized_operands);
        }

        return success();
    }
};

/// Pattern for operations with file and parms attributes.
template <typename OpType>
class EmitCLoadPattern final : public OpConversionPattern<OpType>
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

        // Create constant strings for the attributes
        TypeRange str_type = emitc::OpaqueType::get(rewriter.getContext(), "std::string");
        auto file = rewriter.create<emitc::ConstantOp>(
            op.getLoc(), str_type, emitc::OpaqueAttr::get(rewriter.getContext(), "\"" + op.getFile().str() + "\""));
        auto parms = rewriter.create<emitc::ConstantOp>(
            op.getLoc(), str_type, emitc::OpaqueAttr::get(rewriter.getContext(), "\"" + op.getParms().str() + "\""));
        SmallVector<Value> operands = { file, parms };

        // build a series of calls to our custom evaluator wrapper (for now, because it's faster than dealing with
        // seal's API)
        std::string op_str = "";
        if (std::is_same<OpType, bfv::LoadCtxtOp>())
            op_str = "load_ctxt";
        else if (std::is_same<OpType, bfv::LoadPublicKeyOp>())
            op_str = "load_public_key";
        else if (std::is_same<OpType, bfv::LoadRelinKeysOp>())
            op_str = "load_relin_keys";
        else if (std::is_same<OpType, bfv::LoadGaloisKeysOp>())
            op_str = "load_galois_keys";
        else
            return failure();

        rewriter.replaceOpWithNewOp<emitc::CallOp>(
            op, TypeRange(dstType), llvm::StringRef("evaluator_" + op_str), ArrayAttr(), ArrayAttr(), operands);

        return success();
    }
};

/// This is basically just boiler-plate code,
/// nothing here actually depends on the current dialect thats being converted.
class FunctionConversionPattern final : public OpConversionPattern<func::FuncOp>
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
class EmitCReturnPattern final : public OpConversionPattern<func::ReturnOp>
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
        if (auto bst = dstType.dyn_cast_or_null<emitc::OpaqueType>())
        {
            rewriter.setInsertionPoint(op);
            auto materialized =
                typeConverter->materializeTargetConversion(rewriter, op.getLoc(), dstType, op.getOperands());
            // build a new return op
            rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized);

        } // else do nothing
        return success();
    }
};

void LowerBFVToEmitCPass::runOnOperation()
{
    // TODO: We still need to emit a pre-amble with an include statement
    //  this should refer to some "magic file" that also sets up keys/etc and our custom evaluator wrapper for now

    auto type_converter = TypeConverter();

    type_converter.addConversion([&](Type t) {
        if (t.isa<bfv::CiphertextType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "seal::Ciphertext"));
        else if (t.isa<bfv::PlaintextType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "seal::Plaintext"));
        else if (t.isa<bfv::PublicKeyType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "seal::PublicKey"));
        else if (t.isa<bfv::RelinKeysType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "seal::RelinKeys"));
        else if (t.isa<bfv::GaloisKeysType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "seal::GaloisKeys"));
        else
            return std::optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<bfv::CiphertextType>())
            {
                if (ot.getValue().str() == "seal::Ciphertext")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<bfv::PlaintextType>())
            {
                if (ot.getValue().str() == "seal::Plaintext")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<bfv::PublicKeyType>())
            {
                if (ot.getValue().str() == "seal::PublicKey")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<bfv::RelinKeysType>())
            {
                if (ot.getValue().str() == "seal::RelinKeys")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<bfv::GaloisKeysType>())
            {
                if (ot.getValue().str() == "seal::GaloisKeys")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<bfv::CiphertextType>())
            {
                if (ot.getValue().str() == "seal::Ciphertext")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<bfv::PlaintextType>())
            {
                if (ot.getValue().str() == "seal::Plaintext")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<bfv::PublicKeyType>())
            {
                if (ot.getValue().str() == "seal::PublicKey")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<bfv::RelinKeysType>())
            {
                if (ot.getValue().str() == "seal::RelinKeys")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<bfv::GaloisKeysType>())
            {
                if (ot.getValue().str() == "seal::GaloisKeys")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });
    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<bfv::CiphertextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "seal::Ciphertext")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<bfv::PlaintextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "seal::Plaintext")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<bfv::PublicKeyType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "seal::PublicKey")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<bfv::RelinKeysType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "seal::RelinKeys")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<bfv::GaloisKeysType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "seal::GaloisKeys")
                    return std::optional<Value>(builder.create<bfv::MaterializeOp>(loc, bst, vs));
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });

    ConversionTarget target(getContext());
    target.addIllegalDialect<bfv::BFVDialect>();
    target.addLegalOp<bfv::MaterializeOp>();
    target.addLegalDialect<emitc::EmitCDialect>();
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

    // TODO: Emit the emitc.include operation!

    patterns.add<
        EmitCBasicPattern<bfv::SubOp>, EmitCBasicPattern<bfv::SubPlainOp>, EmitCBasicPattern<bfv::AddOp>,
        EmitCBasicPattern<bfv::AddPlainOp>, EmitCBasicPattern<bfv::AddManyOp>, EmitCBasicPattern<bfv::MultiplyOp>,
        EmitCBasicPattern<bfv::MultiplyPlainOp>, EmitCBasicPattern<bfv::MultiplyManyOp>,
        EmitCBasicPattern<bfv::ModswitchToOp>, EmitCLoadPattern<bfv::LoadCtxtOp>, EmitCLoadPattern<bfv::LoadPtxtOp>,
        EmitCLoadPattern<bfv::LoadPublicKeyOp>, EmitCLoadPattern<bfv::LoadRelinKeysOp>,
        EmitCLoadPattern<bfv::LoadGaloisKeysOp>, EmitCRelinPattern, EmitCRotatePattern, EmitCEncodePattern,
        EmitCDecodePattern, EmitCSinkPattern, FunctionConversionPattern, EmitCReturnPattern>(
        type_converter, patterns.getContext());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}