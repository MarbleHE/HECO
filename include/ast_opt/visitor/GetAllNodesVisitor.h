#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_GETALLNODESVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_GETALLNODESVISITOR_H_

#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/AbstractTarget.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/TernaryOperator.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/visitor/ScopedVisitor.h"
#include "ast_opt/utilities/Scope.h"

class GetAllNodesVisitor : public ScopedVisitor {

 private:

 public:

  std::vector<AbstractNode *> v;

  explicit GetAllNodesVisitor();

  void visit(AbstractNode &elem);

};


#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_GETALLNODESVISITOR_H_
