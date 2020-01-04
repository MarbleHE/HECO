#include "Ast.h"

#include <utility>
#include <variant>
#include <iostream>
#include "Literal.h"
#include "Variable.h"
#include "Function.h"
#include "Return.h"

Ast::Ast(AbstractStatement* rootNode) : rootNode(rootNode) {}

Ast::Ast() {
  rootNode = nullptr;
}

AbstractStatement* Ast::setRootNode(AbstractStatement* node) {
  this->rootNode = node;
  return node;
}

AbstractStatement* Ast::getRootNode() const {
  return rootNode;
}

void Ast::accept(Visitor &v) {
  v.visit(*this);
}

Ast::~Ast() {
  delete rootNode;
}

Literal* Ast::evaluate(const std::map<std::string, Literal*> &paramValues, bool printResult = false) {
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

  // start evaluation traversal
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
  return it != variablesValues.end() ? it->second : nullptr;
}
void Ast::updateVarValue(const std::string &variableIdentifier, Literal* newValue) {
  variablesValues[variableIdentifier] = newValue;
}

