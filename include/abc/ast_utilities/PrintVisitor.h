#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_PRINTVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_PRINTVISITOR_H_

#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "abc/ast/AbstractNode.h"
#include "abc/ast_utilities/Visitor.h"
#include "abc/ast_utilities/PlainVisitor.h"


/// Forward declaration of the class that will actually implement the PrintVisitor's logic
class SpecialPrintVisitor;

/// PrintVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialPrintVisitor, PlainVisitor> PrintVisitor;

class SpecialPrintVisitor : public PlainVisitor {
 private:
  /// Reference to the stream to which we write the output
  std::ostream& os;

  /// Current indentation level
  int indentation_level = 0;

  /// Compute the current required indentation string
  /// from the current indentation_level
  [[nodiscard]] std::string getIndentation() const;

 public:
  explicit SpecialPrintVisitor(std::ostream& os);

#include "abc/ast_utilities/warning_suggestOverride_prologue.h"

  void visit(AbstractNode&);

  void visit(LiteralBool&);

#include "abc/ast_utilities/warning_epilogue.h"

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_PRINTVISITOR_H_
