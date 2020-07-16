#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_PRINTVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_PRINTVISITOR_H_

#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/visitor/Visitor.h"


/// Forward declaration of the class that will actually implement the PrintVisitor's logic
class SpecialPrintVisitor;

/// PrintVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialPrintVisitor> PrintVisitor;

class SpecialPrintVisitor : public ScopedVisitor {
 private:
  /// Reference to the stream to which we write the output
  std::ostream& os;

  /// Current indentation level
  int indentation_level = 0;

  /// Compute the current required indentation string
  /// from the current indentation_level
  std::string getIndentation();

 public:
  explicit SpecialPrintVisitor(std::ostream& os);

  /// We only need one behaviour, therefore we provide a function only for the top level of the class hierarchy
  void visit(AbstractNode&);

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_PRINTVISITOR_H_