#include "heco/legacy_ast/ast_utilities/PlainVisitor.h"
#include "heco/legacy_ast/ast/AbstractExpression.h"
#include "heco/legacy_ast/ast/AbstractNode.h"
#include "heco/legacy_ast/ast/AbstractStatement.h"
#include "heco/legacy_ast/ast/AbstractTarget.h"
#include "heco/legacy_ast/ast/Assignment.h"
#include "heco/legacy_ast/ast/BinaryExpression.h"
#include "heco/legacy_ast/ast/Block.h"
#include "heco/legacy_ast/ast/Call.h"
#include "heco/legacy_ast/ast/ExpressionList.h"
#include "heco/legacy_ast/ast/For.h"
#include "heco/legacy_ast/ast/Function.h"
#include "heco/legacy_ast/ast/FunctionParameter.h"
#include "heco/legacy_ast/ast/If.h"
#include "heco/legacy_ast/ast/IndexAccess.h"
#include "heco/legacy_ast/ast/Literal.h"
#include "heco/legacy_ast/ast/OperatorExpression.h"
#include "heco/legacy_ast/ast/Return.h"
#include "heco/legacy_ast/ast/TernaryOperator.h"
#include "heco/legacy_ast/ast/UnaryExpression.h"
#include "heco/legacy_ast/ast/Variable.h"
#include "heco/legacy_ast/ast/VariableDeclaration.h"

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
