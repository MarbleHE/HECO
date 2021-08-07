
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

  /// Visits an AST and based on noise heuristics and the bitlengths of the primes in the coeff modulus identifies regions
  /// where modswitch ops are appropriate
  /// \param node
  //TODO: implement
  void visit(BinaryExpression& node);

  /// Takes ownership of an AST,insert modSwitch Ops at appropriate places and returns (potentially a different) rewritten AST
  /// \param ast AST to be rewritten (function takes ownership)
  /// \return a new AST that has been rewritten
  //TODO: implement
  static std::unique_ptr<AbstractNode> rewriteAst(std::unique_ptr<AbstractNode> &&ast);



};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_INSERTMODSWITCHVISITOR_H_
