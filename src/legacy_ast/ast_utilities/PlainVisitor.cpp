#include "heco/ast_utilities/PlainVisitor.h"

#include "heco/ast/AbstractExpression.h"
#include "heco/ast/AbstractNode.h"
#include "heco/ast/AbstractStatement.h"
#include "heco/ast/AbstractTarget.h"
#include "heco/ast/Assignment.h"
#include "heco/ast/BinaryExpression.h"
#include "heco/ast/Block.h"
#include "heco/ast/Call.h"
#include "heco/ast/ExpressionList.h"
#include "heco/ast/For.h"
#include "heco/ast/Function.h"
#include "heco/ast/FunctionParameter.h"
#include "heco/ast/If.h"
#include "heco/ast/IndexAccess.h"
#include "heco/ast/Literal.h"
#include "heco/ast/OperatorExpression.h"
#include "heco/ast/Return.h"
#include "heco/ast/TernaryOperator.h"
#include "heco/ast/UnaryExpression.h"
#include "heco/ast/Variable.h"
#include "heco/ast/VariableDeclaration.h"

void PlainVisitor::visit(BinaryExpression &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(Block &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(Call &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(ExpressionList &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(For &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(Function &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(FunctionParameter &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(If &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(IndexAccess &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralBool &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralChar &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralInt &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralFloat &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralDouble &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralString &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(OperatorExpression &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(Return &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(TernaryOperator &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(UnaryExpression &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(Assignment &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(VariableDeclaration &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visit(Variable &elem)
{
  visitChildren(elem);
}

void PlainVisitor::visitChildren(AbstractNode &elem)
{
  for (auto &c : elem)
  {
    c.accept(*this);
  }
}
