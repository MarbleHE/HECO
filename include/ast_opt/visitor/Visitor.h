#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VISITOR_H_

#include <string>
#include <unordered_map>

class AbstractNode;
class AbstractExpr;
class AbstractStatement;
class Ast;
class ArithmeticExpr;
class Block;
class Call;
class CallExternal;
class Datatype;
class Function;
class For;
class FunctionParameter;
class If;
class AbstractLiteral;
class LiteralBool;
class LiteralInt;
class LiteralString;
class LiteralFloat;
class LogicalExpr;
class Operator;
class ParameterList;
class Return;
class UnaryExpr;
class VarAssignm;
class VarDecl;
class Variable;
class While;
class Scope;
class Rotate;
class Transpose;
class MatrixElementRef;
class AbstractMatrix;
class OperatorExpr;
class MatrixAssignm;
class GetMatrixSize;

class Visitor {
 protected:
  bool ignoreScope{false};
  std::unordered_map<AbstractStatement *, Scope *> stmtToScopeMapper;

 public:
  virtual void visit(AbstractNode &elem);

  virtual void visit(AbstractExpr &elem);

  virtual void visit(AbstractStatement &elem);

  virtual void visit(ArithmeticExpr &elem);

  virtual void visit(Block &elem);

  virtual void visit(Call &elem);

  virtual void visit(CallExternal &elem);

  virtual void visit(Datatype &elem);

  virtual void visit(For &elem);

  virtual void visit(Function &elem);

  virtual void visit(FunctionParameter &elem);

  virtual void visit(If &elem);

  virtual void visit(LiteralBool &elem);

  virtual void visit(LiteralInt &elem);

  virtual void visit(LiteralString &elem);

  virtual void visit(LiteralFloat &elem);

  virtual void visit(LogicalExpr &elem);

  virtual void visit(Operator &elem);

  virtual void visit(ParameterList &elem);

  virtual void visit(Return &elem);

  virtual void visit(Rotate &elem);

  virtual void visit(Transpose &elem);

  virtual void visit(UnaryExpr &elem);

  virtual void visit(VarAssignm &elem);

  virtual void visit(VarDecl &elem);

  virtual void visit(Variable &elem);

  virtual void visit(While &elem);

  virtual void visit(MatrixElementRef &elem);

  virtual void visit(AbstractMatrix &elem);

  virtual void visit(OperatorExpr &elem);

  virtual void visit(MatrixAssignm &elem);

  virtual void visit(GetMatrixSize &elem);

  Scope *curScope;

  void setIgnoreScope(bool ignScope);

  void changeToOuterScope();

  void changeToInnerScope(const std::string &nodeId, AbstractStatement *statement);

  Visitor();

  /// This and only this method should be used to traverse an AST.
  /// \param elem A reference to the Abstract Syntax Tree (AST).
  virtual void visit(Ast &elem);

  void addStatementToScope(AbstractStatement &stat);

  void removeStatementFromScope(AbstractStatement &stat);

  /// Hack to force-overwrite the current scope information in a visitor
  /// Useful when you need to call a visitor on a subtree from another visitor:
  /// Simply pass the calling visitor's scope information to the callee visitor
  void forceScope(std::unordered_map<AbstractStatement *, Scope *> stmtToScopeMapper, Scope * curScope);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VISITOR_H_
