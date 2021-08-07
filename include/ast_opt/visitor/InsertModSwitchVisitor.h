
#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_INSERTMODSWITCHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_INSERTMODSWITCHVISITOR_H_



#include <list>
#include <sstream>
#include <string>
#include <utility>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/Visitor.h"


/// Forward declaration of the class that will actually implement the PrintVisitor's logic
class SpecialInsertModSwitchVisitor;

/// PrintVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialInsertModSwitchVisitor> InsertModSwitchVisitor;

class SpecialInsertModSwitchVisitor : public ScopedVisitor {
 private:
  /// Reference to the stream to which we write the output
  std::ostream& os;

  int encNoiseBudget;
  std::unordered_map<std::string, int> noise_map;
  std::unordered_map<std::string, double> rel_noise_map;

 public:
  explicit SpecialInsertModSwitchVisitor(std::ostream& os, std::unordered_map<std::string, int> noise_map,
                                              std::unordered_map<std::string, double> rel_noise_map, int encNoiseBudget);

  void visit(BinaryExpression& node);

  void visit(BinaryExpression& node, BinaryExpression& tail);



#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_INSERTMODSWITCHVISITOR_H_
