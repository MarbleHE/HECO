#ifndef AST_OPTIMIZER_MLIR_PYTHON_PYABC_PYABC_VISITOR_ABC_AST_TO_MLIR_VISITOR_H_
#define AST_OPTIMIZER_MLIR_PYTHON_PYABC_PYABC_VISITOR_ABC_AST_TO_MLIR_VISITOR_H_

#include <ast_opt/utilities/PlainVisitor.h>
#include <ast_opt/utilities/Visitor.h>

#include <ABC/ABCDialect.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

/// Forward declaration of the class that will actually implement the AbcAstToMlirVisitor's logic
class SpecialAbcAstToMlirVisitor;

/// AbcAstToMlirVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialAbcAstToMlirVisitor, PlainVisitor> AbcAstToMlirVisitor;

class SpecialAbcAstToMlirVisitor : public PlainVisitor {
 private:
  mlir::OpBuilder builder;

 public:
  SpecialAbcAstToMlirVisitor(mlir::MLIRContext &ctx);

  void visit(Block&);

};

#endif //AST_OPTIMIZER_MLIR_PYTHON_PYABC_PYABC_VISITOR_ABC_AST_TO_MLIR_VISITOR_H_
