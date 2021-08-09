
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
  std::vector<BinaryExpression *> modSwitchNodes = {}; // vector of binary ops before which modswitch insertion is deemed possible
                                                        // (note this does not entail checking the final noisebudget yet)
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap; // map of current coeff moduli


 public:
  explicit SpecialInsertModSwitchVisitor(std::ostream& os, std::unordered_map<std::string, int> noise_map,
                                              std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap, int encNoiseBudget);

  /// Visits an AST and based on noise heuristics and  identifies binary expressions
  /// where a modswitch op is appropriate based on the bitlengths of the primes in the coeff modulus and the remaining noise budget.
  /// pointer to resulting binary expressions will be stored in the vector modSwitchNodes
  /// \param node
  void visit(BinaryExpression& node);

  /// Takes ownership of an AST,insert modSwitch Ops at appropriate places and returns (potentially a different) rewritten AST
  /// \param ast AST to be rewritten (function takes ownership)
  /// \param binary expresseion before which modswitches are to be inserted
  /// \return a new AST that has been rewritten
  //TODO: implement
  static std::unique_ptr<AbstractNode> insertModSwitchInAst(std::unique_ptr<AbstractNode> *ast, BinaryExpression *binaryExpression = nullptr, std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap = {});

  /// Returns the variable modSwitchNode, i.e. a unique pointer to a binary op that is eligible for insertion of modSwitch before it.
  /// \return modSwitchNode
  std::vector<BinaryExpression *> getModSwitchNodes() const;

  /// Updates noise heuristics for the AST:
  /// Note: in BFV modSwitching introduces noise, therefore, we ust check that the noisebudget does not reach zero
  /// needed to decide whether an inserted modSwitch is indeed kept.
  /// \param ast (root node)
  /// \return noiseBudget of root node
  //TODO Implement
  void updateNoiseMap(AbstractNode& ast);

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_INSERTMODSWITCHVISITOR_H_
