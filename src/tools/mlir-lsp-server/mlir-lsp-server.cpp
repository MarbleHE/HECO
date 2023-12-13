#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>
#include "heco/IR/BFV/BFVDialect.h"
#include "heco/IR/FHE/FHEDialect.h"
#include "heco/IR/EVA/EVADialect.h"
#include "heco/IR/Poly/PolyDialect.h"

// Based on https://mlir.llvm.org/docs/Tools/MLIRLSP/#supporting-custom-dialects-and-passes
int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;
    // TODO (Q&A): Is there a nice way to automate registering dialects to our custom lsp server?
    //  (e.g., via some CMake and/or TableGen magic)
    registry.insert<heco::fhe::FHEDialect>();
    registry.insert<heco::bfv::BFVDialect>();
    registry.insert<heco::eva::EVADialect>();
    registry.insert<heco::poly::PolyDialect>();
    registerAllDialects(registry);

    // TODO (Q&A): Even when registering our dialects with MlirLspServerMain, the created server seems broken
    return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}