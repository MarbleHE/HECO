#ifndef HECO_PASSES_AST2HIR_LOWERASTTOHIR_H_
#define HECO_PASSES_AST2HIR_LOWERASTTOHIR_H_

#include "mlir/Pass/Pass.h"

namespace heco
{

    /// Lowering from the AST-style ABC dialect to SSA representation
    struct LowerASTtoHIRPass : public mlir::PassWrapper<LowerASTtoHIRPass, mlir::OperationPass<mlir::ModuleOp>>
    {
        void getDependentDialects(mlir::DialectRegistry &registry) const override;
        void runOnOperation() override;

        mlir::StringRef getArgument() const final
        {
            return "ast2hir";
        }
    };

} // namespace heco

#endif // HECO_PASSES_AST2HIR_LOWERASTTOHIR_H_
