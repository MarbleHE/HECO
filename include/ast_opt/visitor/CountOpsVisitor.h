#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_

#include <stack>
#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/Visitor.h"
#include "ast_opt/runtime/AbstractCiphertext.h"


/// Forward declaration of the class that will actually implement the PrintVisitor's logic
class SpecialCountOpsVisitor;

/// PrintVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialCountOpsVisitor> CountOpsVisitor;

class SpecialCountOpsVisitor : public ScopedVisitor {
 private:

  int _number_ops = 0;
  int _number_adds = 0;
  int _number_mult = 0;

 public:

  void visit(BinaryExpression &elem);

  int getNumberOps();
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_
