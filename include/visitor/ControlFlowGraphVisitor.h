#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_CONTROLFLOWGRAPHVISITOR_H_

#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <deque>
#include <stack>
#include <unordered_set>
#include "Visitor.h"
#include "AbstractNode.h"

struct GraphNode {
  AbstractNode *refToOriginalNode{nullptr};
  std::vector<GraphNode *> children;
  std::vector<GraphNode *> parents;

  GraphNode() = default;

  GraphNode(std::initializer_list<GraphNode *> parentsToBeAdded) {
    for (auto &c : parentsToBeAdded) addParent(c);
  }

  explicit GraphNode(AbstractNode *originalNode) {
    refToOriginalNode = originalNode;
  }

  void addChild(GraphNode *child) {
    children.push_back(child);
    child->parents.push_back(this);
  }

  void addParent(GraphNode *parent) {
    parents.push_back(parent);
    parent->children.push_back(this);
  }

  /// Traverses all nodes reachable from startNode (descendants) in a depth-first search style and prints the node's
  /// unique node ID.
  /// \param startNode The node where the traversal starts.
  /// \param outputStream The stream where to send the output string to.
  static void traverseAndPrintNodes(GraphNode &startNode, std::ostream &outputStream = std::cout) {
    // nodes that were already printed, helps to detect and bypass graph cycles
    std::unordered_set<std::string> printedNodes;
    // stack of nodes to be processed next
    std::stack<std::pair<GraphNode *, int>> q;
    q.push({std::make_pair(&startNode, 0)});

    while (!q.empty()) {
      auto[curNode, indentationLevel] = q.top();
      q.pop();
      outputStream << "(" << curNode->children.size() << ") "
                   << std::string(indentationLevel, '\t')
                   << curNode->refToOriginalNode->getUniqueNodeId()
                   << std::endl;
      // continue with next While-loop iteration if this node was already printed once - avoids iterating endless if
      // there is a cycle in the graph
      if (printedNodes.count(curNode->refToOriginalNode->getUniqueNodeId()) > 0) {
        if (!curNode->children.empty()) {
          outputStream << "    " << std::string(indentationLevel, '\t')
                       << "... see above, visiting an already visited node ..." << std::endl;
        }
        continue;
      }
      printedNodes.emplace(curNode->refToOriginalNode->getUniqueNodeId());
      // as we are using a stack, we need to add the children in reverse order
      for (auto it = curNode->children.rbegin(); it!=curNode->children.rend(); ++it) {
        q.push(std::make_pair(*it, indentationLevel + 1));
      }
    }
  }

  ///
  /// \param rootNodeOne
  /// \param rootNodeOther
  /// \return
  static bool areEqualGraphs(GraphNode *rootNodeOne, GraphNode *rootNodeOther) {
    // nodes that were already visited, helps to detect and bypass graph cycles
    std::unordered_set<GraphNode *> visitedNodes;
    //
    std::stack<GraphNode *> qOne{{rootNodeOne}};
    std::stack<GraphNode *> qOther{{rootNodeOther}};

    while (!qOne.empty()) {
      auto oneCur = qOne.top();
      auto otherCur = qOther.top();
      qOne.pop();
      qOther.pop();
      // check that the number of child and parent nodes is equal
      if (oneCur->children.size()!=otherCur->children.size() || oneCur->parents.size()!=otherCur->parents.size()) {
        return false;
      }
      if (visitedNodes.count(oneCur) > 0) {
        continue;
      }
      visitedNodes.insert(oneCur);
      for (int i = 0; i < oneCur->children.size(); ++i) {
        qOne.push(oneCur->children.at(i));
        qOther.push(otherCur->children.at(i));
      }
    }
    // ensure that qOne and qOther are empty (qOne is empty because while-loop ended)
    return qOther.empty();
  }
};

class ControlFlowGraphVisitor : public Visitor {
 private:
  /// The nodes that were created most recently. Those are the parent nodes of the next node to be created.
  std::vector<GraphNode *> lastCreatedNodes;

  /// The root node of the control flow graph.
  GraphNode *rootNode;

 public:
  void visit(ArithmeticExpr &elem) override;

  void visit(Block &elem) override;

  void visit(Call &elem) override;

  void visit(CallExternal &elem) override;

  void visit(Datatype &elem) override;

  void visit(For &elem) override;

  void visit(Function &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(If &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralString &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LogicalExpr &elem) override;

  void visit(Operator &elem) override;

  void visit(ParameterList &elem) override;

  void visit(Return &elem) override;

  void visit(Rotate &elem) override;

  void visit(Transpose &elem) override;

  void visit(UnaryExpr &elem) override;

  void visit(VarAssignm &elem) override;

  void visit(VarDecl &elem) override;

  void visit(Variable &elem) override;

  void visit(While &elem) override;

  void visit(GetMatrixElement &elem) override;

  void visit(Ast &elem) override;

  GraphNode *appendStatementToGraph(AbstractStatement &abstractStmt);

  GraphNode *appendStatementToGraph(AbstractStatement &abstractStmt, std::vector<GraphNode *> parentNodes);

  [[nodiscard]] GraphNode *getRootNode() const;
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
