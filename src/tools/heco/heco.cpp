//===- heco.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "heco/IR/BFV/BFVDialect.h"
#include "heco/IR/EVA/EVADialect.h"
#include "heco/IR/FHE/FHEDialect.h"
#include "heco/IR/Poly/PolyDialect.h"
#include "heco/Passes/evalazymodswitch/LazyModswitch.h"
#include "heco/Passes/bfv2emitc/LowerBFVToEmitC.h"
#include "heco/Passes/bfv2llvm/LowerBFVToLLVM.h"
#include "heco/Passes/fhe2bfv/LowerFHEToBFV.h"
#include "heco/Passes/fhe2emitc/LowerFHEToEmitC.h"
#include "heco/Passes/hir2hir/Batching.h"
#include "heco/Passes/hir2hir/CombineSimplify.h"
#include "heco/Passes/hir2hir/InternalOperandBatching.h"
#include "heco/Passes/hir2hir/LowerVirtual.h"
#include "heco/Passes/hir2hir/Nary.h"
#include "heco/Passes/hir2hir/ScalarBatching.h"
#include "heco/Passes/hir2hir/Tensor2BatchedSecret.h"
#include "heco/Passes/hir2hir/UnrollLoops.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
using namespace heco;
using namespace fhe;
using namespace bfv;
using namespace eva;
using namespace poly;

void fullPipelineBuilder(OpPassManager &manager)
{
    manager.addPass(std::make_unique<UnrollLoopsPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass()); // this can greatly reduce the number of operations after unrolling
    manager.addPass(std::make_unique<NaryPass>());

    // Must canonicalize before Tensor2BatchedSecretPass, since it only handles constant indices in tensor.extract
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<Tensor2BatchedSecretPass>());
    manager.addPass(createCanonicalizerPass()); // necessary to remove redundant fhe.materialize
    manager.addPass(createCSEPass()); // necessary to remove duplicate fhe.extract

    manager.addPass(std::make_unique<BatchingPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(
        createCSEPass()); // try and remove all the redundant rotates, in the hope it also gives us less combine ops?
    manager.addPass(std::make_unique<CombineSimplifyPass>());
    manager.addPass(createCSEPass()); // otherwise, the internal batching pass has no "same origin" things to find!
    manager.addPass(createCanonicalizerPass());

    manager.addPass(std::make_unique<InternalOperandBatchingPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());

    manager.addPass(std::make_unique<LowerFHEToBFVPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());

    manager.addPass(std::make_unique<LowerBFVToEmitCPass>());
    manager.addPass(createCanonicalizerPass()); // necessary to remove redundant fhe.materialize
    manager.addPass(createCSEPass());
}

void fhePipelineBuilder(OpPassManager &manager)
{
    manager.addPass(std::make_unique<UnrollLoopsPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass()); // this can greatly reduce the number of operations after unrolling
    manager.addPass(std::make_unique<NaryPass>());

    // Must canonicalize before Tensor2BatchedSecretPass, since it only handles constant indices in tensor.extract
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<Tensor2BatchedSecretPass>());
    manager.addPass(createCanonicalizerPass()); // necessary to remove redundant fhe.materialize
    manager.addPass(createCSEPass()); // necessary to remove duplicate fhe.extract

    manager.addPass(std::make_unique<BatchingPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(
        createCSEPass()); // try and remove all the redundant rotates, in the hope it also gives us less combine ops?
    manager.addPass(std::make_unique<CombineSimplifyPass>());
    manager.addPass(createCSEPass()); // otherwise, the internal batching pass has no "same origin" things to find!
    manager.addPass(createCanonicalizerPass());

    manager.addPass(std::make_unique<InternalOperandBatchingPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());
}

int main(int argc, char **argv)
{
    mlir::MLIRContext context;
    context.enableMultithreading();

    mlir::DialectRegistry registry;
    registry.insert<FHEDialect>();
    registry.insert<EVADialect>();
    registry.insert<BFVDialect>();
    registry.insert<PolyDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<emitc::EmitCDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<linalg::LinalgDialect>();
    context.loadDialect<FHEDialect>();
    context.loadDialect<EVADialect>();
    context.loadDialect<BFVDialect>();
    context.loadDialect<PolyDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<affine::AffineDialect>();
    context.loadDialect<tensor::TensorDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<emitc::EmitCDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<linalg::LinalgDialect>();
    // Uncomment the following to include *all* MLIR Core dialects, or selectively
    // include what you need like above. You only need to register dialects that
    // will be *parsed* by the tool, not the one generated
    registerAllDialects(registry);
    context.loadAllAvailableDialects();


    // Uncomment the following to make *all* MLIR core passes available.
    // This is only useful for experimenting with the command line to compose
    registerAllPasses();

    registerCanonicalizerPass();
    affine::registerAffineLoopUnrollPass();
    registerCSEPass();
    PassRegistration<UnrollLoopsPass>();
    PassRegistration<NaryPass>();
    PassRegistration<Tensor2BatchedSecretPass>();
    PassRegistration<BatchingPass>();
    PassRegistration<CombineSimplifyPass>();
    PassRegistration<InternalOperandBatchingPass>();
    PassRegistration<ScalarBatchingPass>();
    PassRegistration<LowerVirtualPass>();
    PassRegistration<LowerFHEToBFVPass>();
    PassRegistration<LowerBFVToEmitCPass>();
    PassRegistration<LowerBFVToLLVMPass>();
    PassRegistration<LowerFHEToEmitCPass>();
    PassRegistration<LazyModswitchPass>();

    PassPipelineRegistration<>("full-pass", "Run all passes", fullPipelineBuilder);
    PassPipelineRegistration<>("fhe-pass", "Run FHE-level passes", fhePipelineBuilder);

    return asMainReturnCode(MlirOptMain(argc, argv, "HECO optimizer\n", registry));
}
