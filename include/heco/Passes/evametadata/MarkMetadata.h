#ifndef HECO_PASSES_EVAMARKMETADATA_MARKMETADATA_H_
#define HECO_PASSES_EVAMARKMETADATA_MARKMETADATA_H_

#include "mlir/Pass/Pass.h"

struct MarkMetadataPass : public mlir::PassWrapper<MarkMetadataPass, mlir::OperationPass<>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    MarkMetadataPass() = default;
    MarkMetadataPass(const MarkMetadataPass &){}; // Necessary to make Options work

    mlir::StringRef getArgument() const final
    {
        return "evametadata";
    }

    Option<int> source_modulo{*this, "source_modulo", llvm::cl::desc("Length of the modulo chain used to encrypt the source nodes (arguments)."),
                            llvm::cl::init(7)};

    Option<int> source_scale{*this, "source_scale", llvm::cl::desc("Binary exponent of the scale"
                            "of the fixed point representation value in the source nodes"),
                            llvm::cl::init(30)};
    
    Option<int> waterline{*this, "waterline", llvm::cl::desc("Binary exponent of the treshold scale for rescaling"),
                            llvm::cl::init(60)};
    
    Option<int> scale_drop{*this, "scale_drop", llvm::cl::desc("Binary exponent of the scale drop after rescaling"),
                            llvm::cl::init(60)};
};

#endif // HECO_PASSES_EVAMARKMETADATA_MARKMETADATA_H_
