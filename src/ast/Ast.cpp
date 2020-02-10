#include "Ast.h"

#include <variant>
#include <iostream>
#include <set>
#include <queue>
#include "Literal.h"
#include "Variable.h"
#include "Function.h"
#include "Return.h"
#include "LogicalExpr.h"

Ast::Ast(Node* rootNode) : rootNode(rootNode) {}

Ast::Ast() {
  rootNode = nullptr;
}

Node* Ast::setRootNode(Node* node) {
  this->rootNode = node;
  return node;
}

Node* Ast::getRootNode() const {
  return rootNode;
}

void Ast::accept(Visitor &v) {
  v.visit(*this);
}

Ast::~Ast() {
  delete rootNode;
}

Literal* Ast::evaluate(std::map<std::string, Literal*> &paramValues, bool printResult = false) {
  // store param values into a temporary map
  variableValuesForEvaluation.clear();
  for (const auto &[k, v] : paramValues) {
    variableValuesForEvaluation.emplace(std::pair(k, v));
  }

  // ensure that the root node is a Function
  auto func = dynamic_cast<Function*>(this->getRootNode());
  if (!func) throw std::logic_error("AST evaluation only supported for 'Function' root node!");

  // make sure that the provided number of parameters equals the required one
  if (paramValues.size() != func->getParams().size())
    throw std::invalid_argument("AST evaluation requires parameter value for all parameters!");

  // make sure that all parameters specified by the function have a defined value
  for (const auto &fp : func->getParams()) {
    auto var = dynamic_cast<Variable*>(fp.getValue());
    if (var != nullptr && !hasVarValue(var)) {
      throw std::invalid_argument(
          "AST evaluation requires parameter value for parameter " + var->getIdentifier());
    }
  }

  // perform evaluation recursively, starting at the root node
  auto result = getRootNode()->evaluate(*this);

  // print result if flag is set
  if (printResult)
    std::cout << (result == nullptr ? "void" : result->toString()) << std::endl;

  return result;
}

bool Ast::hasVarValue(Variable* var) {
  return getVarValue(var->getIdentifier()) != nullptr;
}

Literal* Ast::getVarValue(const std::string &variableIdentifier) {
  auto it = variableValuesForEvaluation.find(variableIdentifier);
  if (it == variableValuesForEvaluation.end())
    throw std::logic_error("Trying to retrieve value for variable not declared yet: " + variableIdentifier);
  return it->second;
}

void Ast::updateVarValue(const std::string &variableIdentifier, Literal* newValue) {
  variableValuesForEvaluation[variableIdentifier] = newValue;
}

bool Ast::isReversed() const {
  auto allNodes = getAllNodes();
  return std::all_of(allNodes.begin(), allNodes.end(), [](Node* n) { return n->isReversed; });
}

Ast::Ast(const Ast &otherAst, bool keepOriginalUniqueNodeId) {
  std::cout << "Copy constructor called!" << std::endl;
  this->setRootNode(otherAst.getRootNode()->cloneRecursiveDeep(keepOriginalUniqueNodeId));
}

Ast::Ast(const Ast &otherAst) : Ast(otherAst, false) {

}

bool Ast::isValidCircuit() {
  std::set<Node*> allAstNodes = getAllNodes();
  auto supportCircuitMode = [](Node* n) { return n->supportsCircuitMode(); };
  return std::all_of(allAstNodes.begin(), allAstNodes.end(), supportCircuitMode);
}

void Ast::reverseEdges() {
  std::queue<Node*> nodesToCheck;
  nodesToCheck.push(getRootNode());
  while (!nodesToCheck.empty()) {
    auto curNode = nodesToCheck.front();
    nodesToCheck.pop();
    auto nextNodesToAdd = curNode->isReversed ? curNode->getParents() : curNode->getChildren();
    // enqueue nodes to be processed next
    for (auto &n : nextNodesToAdd) nodesToCheck.push(n);
    curNode->swapChildrenParents();
  }
}

std::set<Node*> Ast::getAllNodes() const {
  std::set<Node*> allNodes{getRootNode()};
  std::queue<Node*> nodesToCheck{{getRootNode()}};
  while (!nodesToCheck.empty()) {
    auto curNode = nodesToCheck.front();
    nodesToCheck.pop();
    if (curNode == nullptr) continue;
    allNodes.insert(curNode);
    auto nextNodesToConsider = curNode->isReversed ? curNode->getParentsNonNull() : curNode->getChildrenNonNull();
    for (auto &c : nextNodesToConsider) nodesToCheck.push(c);
  }
  return allNodes;
}

void Ast::deleteNode(Node** node, bool deleteSubtreeRecursively) {
  Node* nodePtr = *node;
  nodePtr->getUniqueNodeId();
  // if deleteSubtreeRecursively is set, we need to delete all children first
  if (deleteSubtreeRecursively) {
    for (auto &c : nodePtr->getChildrenNonNull()) {
      c->getUniqueNodeId();
      deleteNode(&c, true);
    }
  }
    // if deleteSubtreesRecursively is not set but there are children, we cannot continue.
    // probably the user by mistake deleted the whole subtree.
  else if (!nodePtr->getChildrenNonNull().empty()) {
    throw std::logic_error("Cannot remove node (" + nodePtr->getUniqueNodeId()
                               + ") because node has children but deleteSubtreeRecursively is not set (false).");
  }
  // first isolate the node from its parents, then deallocate the heap memory
  nodePtr->isolateNode();
  delete nodePtr;
  // "clear" the pointer to avoid the further use of this deleted node
  *node = nullptr;
}

