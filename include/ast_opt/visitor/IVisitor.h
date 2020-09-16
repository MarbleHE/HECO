#ifndef AST_OPTIMIZER_VISITOR_IVISITOR_H
#define AST_OPTIMIZER_VISITOR_IVISITOR_H

#include <string>

class BinaryExpression;

class Block;

//TODO: Define and implement AST node for Call
// Might want to introduce "FunctionArguments" or something similar
// that links an AbstractExpression to each FunctionParameter
class Call;

class ExpressionList;

class Function;

class FunctionParameter;

class For;

class If;

class IndexAccess;

template<typename T>
class Literal;

typedef Literal<bool> LiteralBool;
typedef Literal<char> LiteralChar;
typedef Literal<int> LiteralInt;
typedef Literal<float> LiteralFloat;
typedef Literal<double> LiteralDouble;
typedef Literal<std::string> LiteralString;

class OperatorExpression;

class Return;

class UnaryExpression;

class Assignment;

class VariableDeclaration;

class Variable;

// TODO: Implement Scope
class Scope;

class IVisitor {
 public:
  virtual void visit(BinaryExpression &elem) = 0;

  virtual void visit(Block &elem) = 0;

  virtual void visit(ExpressionList &elem) = 0;

  virtual void visit(For &elem) = 0;

  virtual void visit(Function &elem) = 0;

  virtual void visit(FunctionParameter &elem) = 0;

  virtual void visit(If &elem) = 0;

  virtual void visit(IndexAccess &elem) = 0;

  virtual void visit(LiteralBool &elem) = 0;

  virtual void visit(LiteralChar &elem) = 0;

  virtual void visit(LiteralInt &elem) = 0;

  virtual void visit(LiteralFloat &elem) = 0;

  virtual void visit(LiteralDouble &elem) = 0;

  virtual void visit(LiteralString &elem) = 0;

  virtual void visit(OperatorExpression &elem) = 0;

  virtual void visit(Return &elem) = 0;

  virtual void visit(UnaryExpression &elem) = 0;

  virtual void visit(Assignment &elem) = 0;

  virtual void visit(VariableDeclaration &elem) = 0;

  virtual void visit(Variable &elem) = 0;

};

#endif //AST_OPTIMIZER_VISITOR_IVISITOR_H