//===- abc-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv)
{
    mlir::MLIRContext context;

    mlir::DialectRegistry registry;
    registry.insert<emitc::EmitCDialect>();
    context.loadDialect<emitc::EmitCDialect>();
    // Uncomment the following to include *all* MLIR Core dialects, or selectively
    // include what you need like above. You only need to register dialects that
    // will be *parsed* by the tool, not the one generated
    // registerAllDialects(registry);

    registerAllTranslations();

    return failed(mlir::mlirTranslateMain(argc, argv, "EmitC Translation Tool"));
}
