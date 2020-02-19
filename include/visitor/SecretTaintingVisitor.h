#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_SECRETTAINTINGVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_SECRETTAINTINGVISITOR_H_

#include <unordered_map>
#include <vector>
#include <AbstractNode.h>
#include <set>
#include "Visitor.h"

/// The SecretTaintingVisitor traverses through a given AST and marks all Nodes as "tainted" that in some way deal
/// with an encrypted variable.
/// For example, if [a,b] are encrypted variables and "int result = a + 2;" is a statement, then the VarAssignm
/// object, the BinaryExpr object and both of the literals would be marked as tainted.
class SecretTaintingVisitor : public Visitor {
 protected:
  /// The list of tainted Nodes.
  std::set<std::string> taintedNodes;

  /// The list of tainted variables.
  std::set<std::string> taintedVariables;

 public:
  /// Returns the list of all tainted nodes in the AST.
  /// \return A list consisting of the node's name (first) and True if the node is tainted, otherwise False (second).
  [[nodiscard]] const std::set<std::string> &getSecretTaintingList() const;

  void visit(BinaryExpr &elem) override;
  void visit(Block &elem) override;
  void visit(Call &elem) override;
  void visit(CallExternal &elem) override;
  void visit(Function &elem) override;
  void visit(FunctionParameter &elem) override;
  void visit(If &elem) override;
  void visit(LiteralBool &elem) override;
  void visit(LiteralInt &elem) override;
  void visit(LiteralString &elem) override;
  void visit(LiteralFloat &elem) override;
  void visit(LogicalExpr &elem) override;
  void visit(Operator &elem) override;
  void visit(Return &elem) override;
  void visit(UnaryExpr &elem) override;
  void visit(VarAssignm &elem) override;
  void visit(VarDecl &elem) override;
  void visit(Variable &elem) override;
  void visit(While &elem) override;
  void visit(Ast &elem) override;

  bool nodeIsTainted(AbstractNode &node) const;
  void addTaintedNodes(std::vector<AbstractNode *> nodesToAdd);
  void addTaintedNode(AbstractNode *node);
  [[nodiscard]] bool anyNodesAreTainted(std::vector<AbstractNode *> nodes) const;
  void checkAndAddTaintedChildren(AbstractStatement *n, std::vector<std::string> varIdentifiersInRhs);

  template<class RandomAccessIterator>
  void addTaintedVariableIdentifiers(RandomAccessIterator first, RandomAccessIterator last) {
    std::copy(first, last, std::inserter(taintedVariables, taintedVariables.end()));
  }

  template<class RandomAccessIterator>
  bool anyVariableIdentifierIsTainted(RandomAccessIterator first, RandomAccessIterator last) {
    // make sure that the collection's template type T is a std::string
    // see https://stackoverflow.com/a/53716873/3017719
    static_assert(std::is_same_v<std::decay_t<decltype(*first)>, std::string>);
    // check if any of the variable identifiers of the given vector is in the vector of known tainted variables
    return std::any_of(first, last, [&](const std::string &identifier) {
      return std::find(taintedVariables.begin(), taintedVariables.end(), identifier)!=taintedVariables.end();
    });
  }
  void visit(Datatype &elem) override;
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_SECRETTAINTINGVISITOR_H_
