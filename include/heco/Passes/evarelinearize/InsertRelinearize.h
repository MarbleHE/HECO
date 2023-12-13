#ifndef HECO_PASSES_EVARELINEARIZE_INSERTRELINEARIZE_H_
#define HECO_PASSES_EVARELINEARIZE_INSERTRELINEARIZE_H_

#include "mlir/Pass/Pass.h"

struct InsertRelinearizePass : public mlir::PassWrapper<InsertRelinearizePass, mlir::OperationPass<>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    InsertRelinearizePass() = default;
    InsertRelinearizePass(const InsertRelinearizePass &){}; // Necessary to make Options work

    mlir::StringRef getArgument() const final
    {
        return "evarelinearize";
    }

    Option<int> poly_mod_degree{ *this, "poly_mod_degree",
                                 llvm::cl::desc("Polynomial Degree of the Ciphertexts to assume."),
                                 llvm::cl::init(1024) };

    Option<std::string> params_file{
        *this, "params_file",
        llvm::cl::desc("Name of the paramter file defining, among other things, the ciphertext moduli"),
        llvm::cl::init("parms.parms")
    };
};

#endif // HECO_PASSES_EVARELINEARIZE_INSERTRELINEARIZE_H_
