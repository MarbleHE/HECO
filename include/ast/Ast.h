#ifndef AST_OPTIMIZER_INCLUDE_AST_H
#define AST_OPTIMIZER_INCLUDE_AST_H

#include <map>
#include <string>
#include "Node.h"
#include "Visitor.h"
#include <set>

class Ast {
 private:
  /// The root node of the AST. All other nodes of the AST must somehow be referenced by the rootNode.
  /// For example, an AST root node can be an object of Function class that references statements (i.e., objects derived
  /// from AbstractStatement) that represent the computations.
  Node* rootNode;

  /// This map stores the values passed to evaluateAst or evaluateCircuit for variables in the computation and serves
  /// as lookup table and central storage during the evaluation process.
  std::map<std::string, Literal*> variableValuesForEvaluation;

 public:
  Ast();

  // copy constructors
  Ast(const Ast &otherAst);
  Ast(const Ast &otherAst, bool keepOriginalUniqueNodeId);

  /// Creates a new Abstract Syntax Tree (AST).
  /// \param rootNode The node to be defined as root for this AST.
  explicit Ast(Node* rootNode);

  ~Ast();

  Node* setRootNode(Node* node);

  [[nodiscard]] Node* getRootNode() const;

  virtual void accept(Visitor &v);

  bool hasVarValue(Variable* var);

  Literal* getVarValue(const std::string &variableIdentifier);

  void updateVarValue(const std::string &variableIdentifier, Literal* newValue);

  Literal* evaluateAst(const std::map<std::string, Literal*> &paramValues, bool printResult);

  Literal* evaluateCircuit(const std::map<std::string, Literal*> &paramValues, bool printResult);

  /// Checks whether the AST (more specifically, all of the AST's edges) are reversed.
  /// \return True iff all edges of the AST are reversed, otherwise false.
  [[nodiscard]] bool isReversed() const;

  /// Checks whether the current AST consists of nodes that are circuit-compatible, i.e., that define the child/parent
  /// nodes and can be looked at as a circuit.
  /// \return True iff the current AST consists of circuit-compatible nodes only.
  bool isValidCircuit();

  /// Reverses all edges by switching child and parent nodes of each reachable node within the AST.
  void reverseEdges();

  /// Traverses through the tree in BFS-style and collects all the nodes of the AST.
  [[nodiscard]] std::set<Node*> getAllNodes() const;

  /// Deletes a node from the AST.
  /// \param node The node to delete from the AST.
  /// \param deleteSubtreeRecursively Determines whether children should be deleted recursively.
  void deleteNode(Node** node, bool deleteSubtreeRecursively = false);
  Literal* evaluate(bool printResult);
  std::set<Node*> getAllNodes(const std::function<bool(Node*)> &predicate) const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_H



