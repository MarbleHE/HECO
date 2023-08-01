#ifndef HECO_PASSES_FHE2BFV_LOWERFHETOBFV_H_
#define HECO_PASSES_FHE2BFV_LOWERFHETOBFV_H_

#include "mlir/Pass/Pass.h"

struct LowerFHEToBFVPass : public mlir::PassWrapper<LowerFHEToBFVPass, mlir::OperationPass<>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    LowerFHEToBFVPass() = default;
    LowerFHEToBFVPass(const LowerFHEToBFVPass &){}; // Necessary to make Options work

    mlir::StringRef getArgument() const final
    {
        return "fhe2bfv";
    }

    Option<int> poly_mod_degree{ *this, "poly_mod_degree",
                                 llvm::cl::desc("Polynomial Degree of the Ciphertexts to assume."),
                                 llvm::cl::init(1024) };

    Option<std::string> params_file{
        *this, "params_file",
        llvm::cl::desc("Name of the paramter file defining, among other things, the ciphertext moduli"),
        llvm::cl::init("parms.parms")
    };

    Option<std::string> relin_keys_file{ *this, "relin_keys_file",
                                         llvm::cl::desc("Name of the file containing the relineriaztion keys."),
                                         llvm::cl::init("relin_keys.rk") };

    Option<std::string> galois_keys_file{ *this, "galois_keys_file",
                                          llvm::cl::desc("Name of the file containing the Galois keys."),
                                          llvm::cl::init("galois_keys.gk") };
};

#endif // HECO_PASSES_FHE2BFV_LOWERFHETOBFV_H_
