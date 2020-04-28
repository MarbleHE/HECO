#ifndef AST_OPTIMIZER_INCLUDE_AST_H
#define AST_OPTIMIZER_INCLUDE_AST_H

#include <map>
#include <string>
#include <iostream>
#include <set>
#include "AbstractNode.h"
#include "ast_opt/visitor/Visitor.h"

class Ast {
 private:
  /// The root node of the AST. All other nodes of the AST must somehow be referenced by the rootNode.
  /// For example, an AST root node can be an object of Function class that references statements (i.e., objects derived
  /// from AbstractStatement) that represent the computations.
  AbstractNode *rootNode;

 public:
  Ast();

  ~Ast();

  /// Creates a (deep) copy of the given AST, i.e., including all of the root's children nodes.
  /// \param otherAst The AST to be copied.
  Ast(const Ast &otherAst);

  /// Similar as Ast(const Ast &otherAst), creates a (deep) copy of the given AST while additionally preserving the
  /// nodes unique IDs.
  /// \param otherAst The "original" AST to be copied.
  /// \param keepOriginalUniqueNodeId A flag indicating whether the node IDs from the original AST should be copied too.
  Ast(const Ast &otherAst, bool keepOriginalUniqueNodeId);

  /// Creates a new Abstract Syntax Tree (AST).
  /// \param rootNode The node to be defined as root for this AST.
  explicit Ast(AbstractNode *rootNode);

  /// Defines the root node of the AST.
  /// \param node The node to be defined as root node of this AST.
  AbstractNode *setRootNode(AbstractNode *node);

  [[nodiscard]] AbstractNode *getRootNode() const;

  virtual void accept(Visitor &v);

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
};

#endif //AST_OPTIMIZER_INCLUDE_AST_H



