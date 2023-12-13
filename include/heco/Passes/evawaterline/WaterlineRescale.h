#ifndef HECO_PASSES_EVAWATERLINE_WATERLINERESCALE_H_
#define HECO_PASSES_EVAWATERLINE_WATERLINERESCALE_H_

#include "mlir/Pass/Pass.h"

struct WaterlineRescalePass : public mlir::PassWrapper<WaterlineRescalePass, mlir::OperationPass<>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    WaterlineRescalePass() = default;
    WaterlineRescalePass(const WaterlineRescalePass &){}; // Necessary to make Options work

    mlir::StringRef getArgument() const final
    {
        return "evawaterline";
    }

    Option<int> source_scale{*this, "source_scale", llvm::cl::desc("Binary exponent of the scale"
                            "of the fixed point representation value in the source nodes"),
                            llvm::cl::init(30)};
    
    Option<int> waterline{*this, "waterline", llvm::cl::desc("Binary exponent of the treshold scale for rescaling"),
                            llvm::cl::init(60)};
    
    Option<int> scale_drop{*this, "scale_drop", llvm::cl::desc("Binary exponent of the scale drop after rescaling"),
                            llvm::cl::init(60)};
};

#endif // HECO_PASSES_EVAWATERLINE_WATERLINERESCALE_H_
