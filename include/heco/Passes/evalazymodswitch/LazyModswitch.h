#ifndef HECO_PASSES_EVALAZYMODSWITCH_LAZYMODSWITCH_H_
#define HECO_PASSES_EVALAZYMODSWITCH_LAZYMODSWITCH_H_

#include "mlir/Pass/Pass.h"

struct LazyModswitchPass : public mlir::PassWrapper<LazyModswitchPass, mlir::OperationPass<>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    LazyModswitchPass() = default;
    LazyModswitchPass(const LazyModswitchPass &){}; // Necessary to make Options work

    mlir::StringRef getArgument() const final
    {
        return "evalazymodswitch";
    }

    Option<int> source_modulo{*this, "source_mod", llvm::cl::desc("Length of the modulo chain used for encrypting the source nodes (arguments)."),
                            llvm::cl::init(7)};    
};

#endif // HECO_PASSES_EVALAZYMODSWITCH_LAZYMODSWITCH_H_
