
#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_INSERTMODSWITCHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_INSERTMODSWITCHVISITOR_H_



#include <list>
#include <sstream>
#include <string>
#include <utility>
#include <seal/seal.h>

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
  std::unique_ptr<BinaryExpression> modSwitchNode; // bin expr after which modswitch is determined to be possible
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap; // map of current coeff moduli


 public:
  explicit SpecialInsertModSwitchVisitor(std::ostream& os, std::unordered_map<std::string, int> noise_map,
                                              std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap, int encNoiseBudget);

  /// Visits an AST and based on noise heuristics and  identifies a binary expression
  /// where a modswitch op is appropriate based on the bitlengths of the primes in the coeff modulus and the remaining noise budget.
  /// pointer to resultiong binary expression will be stored in modwitchNode
  /// \param node
  //TODO: implement
  void visit(BinaryExpression& node);

  /// Takes ownership of an AST,insert modSwitch Ops at appropriate places and returns (potentially a different) rewritten AST
  /// \param ast AST to be rewritten (function takes ownership)
  /// \return a new AST that has been rewritten
  //TODO: implement
  static std::unique_ptr<AbstractNode> rewriteAst(std::unique_ptr<AbstractNode> &&ast);

  /// Returns the variable modSwitchNode, i.e. a unique pointer to a binary op that is eligible for insertion of modSwitch after it.
  /// \return modSwitchNode
  //TODO: implement
  std::unique_ptr<BinaryExpression> getModSwitchNode() const;

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_INSERTMODSWITCHVISITOR_H_
