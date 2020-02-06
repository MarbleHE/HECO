#ifndef MASTER_THESIS_CODE_AST_H
#define MASTER_THESIS_CODE_AST_H

#include <map>
#include <string>
#include <set>
#include "../include/ast/Node.h"
#include "../visitor/Visitor.h"

class Ast {
 private:
  Node* rootNode;

  std::map<std::string, Literal*> variableValuesForEvaluation;

 public:
  Ast();

  // copy constructor
  Ast(const Ast &otherAst);

  explicit Ast(Node* rootNode);

  ~Ast();

  Node* setRootNode(Node* node);

  [[nodiscard]] Node* getRootNode() const;

  virtual void accept(Visitor &v);

  bool hasVarValue(Variable* var);

  Literal* getVarValue(const std::string &variableIdentifier);

  void updateVarValue(const std::string &variableIdentifier, Literal* newValue);

  Literal* evaluate(std::map<std::string, Literal*> &paramValues, bool printResult);

  [[nodiscard]] bool isReversed() const;

  /// Prints the AST in DOT language as used by graphviz, see https://www.graphviz.org/doc/info/lang.html.
  void printGraphviz();

  /// Checks whether the current AST consists of nodes that are circuit-compatible, i.e., that define the child/parent
  /// nodes and can be looked at as a circuit.
  /// \return True iff the current AST consists of circuit-compatible nodes only.
  bool isValidCircuit();

  /// Reverses all edges by switching child and parent nodes of each reachable node within the AST.
  void reverseEdges();

  /// Traverses through the tree in BFS-style and collects all the nodes of the AST.
  std::set<Node*> getAllNodes() const;
};

#endif //MASTER_THESIS_CODE_AST_H



