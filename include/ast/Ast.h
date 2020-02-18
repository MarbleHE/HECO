#ifndef AST_OPTIMIZER_INCLUDE_AST_H
#define AST_OPTIMIZER_INCLUDE_AST_H

#include <map>
#include <string>
#include <iostream>
#include <set>
#include "AbstractNode.h"
#include "Visitor.h"

class Ast {
 private:
  /// The root node of the AST. All other nodes of the AST must somehow be referenced by the rootNode.
  /// For example, an AST root node can be an object of Function class that references statements (i.e., objects derived
  /// from AbstractStatement) that represent the computations.
  AbstractNode *rootNode;

  /// This map stores the values passed to evaluateAst or evaluateCircuit for variables in the computation and serves
  /// as lookup table and central storage during the evaluation process.
  std::unordered_map<std::string, AbstractLiteral *> variableValuesForEvaluation;

 public:
  Ast();

  ~Ast();

  // copy constructors
  Ast(const Ast &otherAst);

  Ast(const Ast &otherAst, bool keepOriginalUniqueNodeId);

  /// Creates a new Abstract Syntax Tree (AST).
  /// \param rootNode The node to be defined as root for this AST.
  explicit Ast(AbstractNode *rootNode);

  /// Defines the root node of the AST.
  /// \param node The node to be defined as root node of this AST.
  AbstractNode *setRootNode(AbstractNode *node);

  [[nodiscard]] AbstractNode *getRootNode() const;

  virtual void accept(Visitor &v);

  bool hasVarValue(Variable *var);

  AbstractLiteral *getVarValue(const std::string &variableIdentifier);

  void updateVarValue(const std::string &variableIdentifier, AbstractLiteral *newValue);

  std::vector<AbstractLiteral *>
  evaluateAst(const std::unordered_map<std::string, AbstractLiteral *> &paramValues, bool printResult = false);

  std::vector<AbstractLiteral *>
  evaluateCircuit(const std::unordered_map<std::string, AbstractLiteral *> &paramValues, bool printResult = false);

  /// Checks whether the AST (more specifically, all of the AST's edges) are reversed.
  /// \return True iff all edges of the AST are reversed, otherwise false.
  [[nodiscard]] bool isReversed() const;

  /// Checks whether the current AST consists of nodes that are circuit-compatible, i.e., that define the child/parent
  /// nodes and can be looked at as a circuit.
  /// \return True iff the current AST consists of circuit-compatible nodes only.
  bool isValidCircuit();

  /// Reverses all edges by switching child and parent nodes of each reachable node within the AST.
  void reverseEdges();

  /// Traverses the tree in BFS-style and collects all the nodes of the AST.
  /// \return A list of all nodes reachable from the AST's root node.
  [[nodiscard]] std::set<AbstractNode *> getAllNodes() const;

  /// Traverses the tree in BFS-style and collects all the nodes of the AST for that the predicate returns True.
  /// \param predicate A function that takes a AbstractNode* and returns True if this node should be returned, otherwise False.
  /// \return A list of all nodes reachable from the AST's root node.
  std::set<AbstractNode *> getAllNodes(const std::function<bool(AbstractNode *)> &predicate) const;

  /// Deletes a node from the AST.
  /// \param node The node to delete from the AST.
  /// \param deleteSubtreeRecursively Determines whether children should be deleted recursively.
  void deleteNode(AbstractNode **node, bool deleteSubtreeRecursively = false);

  std::vector<AbstractLiteral *> evaluate(bool printResult, std::ostream &outputStream = std::cout);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_H



