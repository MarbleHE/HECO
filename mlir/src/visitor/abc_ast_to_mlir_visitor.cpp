
#include <ast_opt_mlir/visitor/abc_ast_to_mlir_visitor.h>

using namespace abc;

SpecialAbcAstToMlirVisitor::SpecialAbcAstToMlirVisitor(mlir::MLIRContext &ctx) : builder(&ctx) {};

void SpecialAbcAstToMlirVisitor::visit(Block &) {
  std::cout << "Visited block" << std::endl;
  auto op = builder.create<VariableDeclarationOp>(builder.getUnknownLoc(),  "test", builder.getF32Type(), 1);

  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  module.push_back(op);
  module.dump();
}
