#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_NOISEPRINTVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_NOISEPRINTVISITOR_H_

#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/Visitor.h"


/// Forward declaration of the class that will actually implement the PrintVisitor's logic
class SpecialNoisePrintVisitor;

/// PrintVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialNoisePrintVisitor> NoisePrintVisitor;

class SpecialNoisePrintVisitor : public ScopedVisitor {
 private:
  /// Reference to the stream to which we write the output
  std::ostream& os;

  /// Current indentation level
  int indentation_level = 0;

  /// Compute the current required indentation string
  /// from the current indentation_level
  [[nodiscard]] std::string getIndentation() const;

  std::unordered_map<std::string, int> noise_map;
  std::unordered_map<std::string, double> rel_noise_map;

 public:
  explicit SpecialNoisePrintVisitor(std::ostream& os, std::unordered_map<std::string, int> noise_map,
                                    std::unordered_map<std::string, double> rel_noise_map);

  void visit(AbstractNode&);

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_NOISEPRINTVISITOR_H_
