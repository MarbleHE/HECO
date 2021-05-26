#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_

#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/visitor/ScopedVisitor.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"


/// Forward declaration of the class that will actually implement the PrintVisitor's logic
class SpecialCountOpsVisitor;


/// A custom exception class to be used to break out of the RuntimeVisitor in case that we visited a Return statement.
/// This is important for programs that use Return statements to prematurely exit a program, e.g., in the body of a
/// For loop. The exception must be caught (and ignored) by the caller.
struct ReturnStatementReached : public std::exception {
  [[nodiscard]] const char *what() const noexcept override {
    return "Program reached Return statement. Exception raised to break out of RuntimeVisitor. "
           "This exception must be caught but can be ignored.";
  }
};

/// PrintVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialCountOpsVisitor> CountOpsVisitor;

class SpecialCountOpsVisitor : public ScopedVisitor {
 private:

  /// Current indentation level
  int _number_ops = 0;

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
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COUNTOPSVISITOR_H_
