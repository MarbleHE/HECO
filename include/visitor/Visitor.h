#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_H
#define AST_OPTIMIZER_INCLUDE_VISITOR_H

#include <string>

class AbstractExpr;

class AbstractStatement;

class Ast;

class BinaryExpr;

class Block;

class Call;

class CallExternal;

class Class;

class Function;

class FunctionParameter;

class Group;

class If;

class Literal;

class LiteralBool;

class LiteralInt;

class LiteralString;

class LiteralFloat;

class LogicalExpr;

class Operator;

class Return;

class UnaryExpr;

class VarAssignm;

class VarDecl;

class Variable;

class While;

class Scope;

class Visitor {
 public:
  virtual void visit(BinaryExpr &elem);

  virtual void visit(Block &elem);

  virtual void visit(Call &elem);

  virtual void visit(CallExternal &elem);

  virtual void visit(Function &elem);

  virtual void visit(FunctionParameter &elem);

  virtual void visit(Group &elem);

  virtual void visit(If &elem);

  virtual void visit(LiteralBool &elem);

  virtual void visit(LiteralInt &elem);

  virtual void visit(LiteralString &elem);

  virtual void visit(LiteralFloat &elem);

  virtual void visit(LogicalExpr &elem);

  virtual void visit(Operator &elem);

  virtual void visit(Return &elem);

  virtual void visit(UnaryExpr &elem);

  virtual void visit(VarAssignm &elem);

  virtual void visit(VarDecl &elem);

  virtual void visit(Variable &elem);

  virtual void visit(While &elem);

  Scope* curScope;

  void changeToOuterScope();

  void changeToInnerScope(const std::string &nodeId);

  Visitor();

  /// This and only this method should be used to traverse an AST.
  /// \param elem A reference to the Abstract Syntax Tree (AST).
  virtual void visit(Ast &elem);
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_H
