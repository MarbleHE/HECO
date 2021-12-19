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
  mlir::ModuleOp module;

 public:
  SpecialAbcAstToMlirVisitor(mlir::MLIRContext &ctx);

  void visit(AbstractExpression &expr);

//  void visit(BinaryExpression &elem);

  void visit(Block &elem);

//  void visit(Call &elem);
//
//  void visit(ExpressionList &elem);
//
//  void visit(For &elem);
//
//  void visit(Function &elem);
//
//  void visit(FunctionParameter &elem);
//
//  void visit(If &elem);
//
//  void visit(IndexAccess &elem);
//
//  void visit(LiteralBool &elem);
//
//  void visit(LiteralChar &elem);
//
//  void visit(LiteralInt &elem);
//
//  void visit(LiteralFloat &elem);
//
//  void visit(LiteralDouble &elem);
//
//  void visit(LiteralString &elem);
//
//  void visit(OperatorExpression &elem);

  void visit(Return &elem);

//  void visit(TernaryOperator &elem);
//
//  void visit(UnaryExpression &elem);
//
//  void visit(Assignment &elem);

  void visit(VariableDeclaration &elem);

//  void visit(Variable &elem);

};

#endif //AST_OPTIMIZER_MLIR_PYTHON_PYABC_PYABC_VISITOR_ABC_AST_TO_MLIR_VISITOR_H_
