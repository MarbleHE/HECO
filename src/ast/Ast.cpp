#include "Ast.h"

#include <utility>
#include <variant>
#include <iostream>
#include <deque>
#include <sstream>
#include "Literal.h"
#include "Variable.h"
#include "Function.h"
#include "Return.h"

Ast::Ast(Node *rootNode) : rootNode(rootNode) {}

Ast::Ast() {
    rootNode = nullptr;
}

Node *Ast::setRootNode(Node *node) {
    this->rootNode = node;
    return node;
}

Node *Ast::getRootNode() const {
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
  variablesValues.clear();
  for (const auto &[k, v] : paramValues) {
    variablesValues.emplace(std::pair(k, v));
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
        throw std::invalid_argument("AST evaluation requires parameter value for parameter " + var->getIdentifier());
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
    auto it = variablesValues.find(variableIdentifier);
    if (it == variablesValues.end())
        throw std::logic_error("Trying to retrieve value for variable not declared yet: " + variableIdentifier);
    return it->second;
}

void Ast::updateVarValue(const std::string &variableIdentifier, Literal *newValue) {
    variablesValues[variableIdentifier] = newValue;
}

void Ast::toggleIsReversed() {
    reversedEdges = !reversedEdges;
}

bool Ast::isReversed() const {
    return reversedEdges;
}

void Ast::printGraphviz() {
    std::stringstream ss;
    ss << "digraph D {" << std::endl;

    std::deque<std::pair<Node *, int>> q;
    q.emplace_back(getRootNode(), 1);
    while (!q.empty()) {
        auto curNode = q.front().first;
        auto il = q.front().second;
        q.pop_front();
        std::string outStr;
        ss << curNode->getDotFormattedString(isReversed(), "\t", true);
        auto nodes = (isReversed()) ? curNode->getParents() : curNode->getChildren();
        for_each(nodes.rbegin(), nodes.rend(), [&q, il](Node *n) {
            q.emplace_front(n, il + 1);
        });
    }

    ss << "}" << std::endl;
    std::cout << ss.str();
}

