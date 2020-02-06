#include "Ast.h"

#include <utility>
#include <variant>
#include <iostream>
#include <deque>
#include <sstream>
#include <ConeRewriter.h>
#include <set>
#include <queue>
#include <stack>
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
    if (auto var = dynamic_cast<Variable*>(fp.getValue())) {
      if (!hasVarValue(var)) {
        throw std::invalid_argument(
            "AST evaluation requires parameter value for parameter " + var->getIdentifier());
      }
    }
  }

  auto result = getRootNode()->evaluate(*this);

  // print result
  if (printResult) {
    if (result == nullptr) {
      std::cout << "void" << std::endl;
    } else {
      std::cout << *result << std::endl;
    }
  }

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

void Ast::toggleIsReversed() {
  hasReversedEdges = !hasReversedEdges;
}

bool Ast::isReversed() const {
  return hasReversedEdges;
}

void Ast::printGraphviz() {
  std::stringstream ss;
  ss << "digraph D {" << std::endl;
  std::deque<std::pair<Node*, int>> q;
  q.emplace_back(getRootNode(), 1);
  while (!q.empty()) {
    auto curNode = q.front().first;
    auto il = q.front().second;
    q.pop_front();
    std::string outStr;
    if (curNode == nullptr) continue;
    ss << curNode->getDotFormattedString(isReversed(), "\t", true);
    auto nodes = (isReversed()) ? curNode->getParents() : curNode->getChildren();
    for_each(nodes.rbegin(), nodes.rend(), [&q, il](Node* n) {
      q.emplace_front(n, il + 1);
    });
  }
  ss << "}" << std::endl;
  std::cout << ss.str();
}

Ast::Ast(const Ast &otherAst) {
  std::cout << "Copy constructor called!" << std::endl;
  Ast* clonedAst = new Ast;
  clonedAst->setRootNode(otherAst.getRootNode()->cloneRecursiveDeep());
  throw std::logic_error("Copy constructor for AST not implemented yet!");
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
    auto nextNodesToAdd = isReversed() ? curNode->getParents() : curNode->getChildren();
    std::for_each(nextNodesToAdd.begin(), nextNodesToAdd.end(), [&](Node* n) {
      nodesToCheck.push(n);
    });
    curNode->swapChildrenParents();
  }
  toggleIsReversed();
}

std::set<Node*> Ast::getAllNodes() {
  std::set<Node*> allNodes{getRootNode()};
  std::queue<Node*> nodesToCheck{{getRootNode()}};
  while (!nodesToCheck.empty()) {
    auto curNode = nodesToCheck.front();
    nodesToCheck.pop();
    if (curNode == nullptr) continue;
    allNodes.insert(curNode);
    for (auto &c : curNode->getChildren()) nodesToCheck.push(c);
  }
  return allNodes;
}

