#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_

#include <stack>
#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/visitor/ScopedVisitor.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"


/// Forward declaration of the class that will actually implement the PrintVisitor's logic
class SpecialCountOpsVisitor;



/// PrintVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialCountOpsVisitor> CountOpsVisitor;

class SpecialCountOpsVisitor : public ScopedVisitor {
 private:

  /// A stack that keeps track of intermediate results. Each visit(..) of an expression (node that inherits from
  /// AbstractExpression) pushes its evaluation result on the stack so that the parent node can acccess the result.
  std::stack<std::unique_ptr<AbstractValue>> intermedResult;

  int _number_ops = 0;
  int _number_adds = 0;

 public:
  SpecialCountOpsVisitor(AbstractNode &inputs);

  void visit(AbstractNode&);

  void visit(BinaryExpression &elem);

  int getNumberOps();

  /// Executes an input program given as AST. *NOTE*: As our RuntimeVisitor does not handle Functions yet, the visitor
  /// must be called on the function's Block. The missing function signature (input and outputs args) are derived from
  /// the input and output AST given in the constructor and the getOutput method, respectively.
  /// \param rootNode The root node of the input program.
  void executeAst(AbstractNode &rootNode);
  std::unique_ptr<AbstractValue> getNextStackElement();
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_
