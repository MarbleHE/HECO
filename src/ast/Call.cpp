#include "../../include/ast/Call.h"

#include <utility>
#include <sstream>
#include <typeindex>
#include <iostream>
#include "Function.h"
#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"

json Call::toJson() const {
  json j = {{"type", getNodeName()},
            {"arguments", this->arguments},
            {"function", this->func->toJson()}};
  return j;
}

void Call::accept(Visitor &v) {
  v.visit(*this);
}

const std::vector<FunctionParameter*> &Call::getArguments() const {
  return arguments;
}

std::string Call::getNodeName() const {
  return "Call";
}

Call::~Call() {
  delete func;
}

Call::Call(Function* func) : func(func) {
}

Call::Call(std::vector<FunctionParameter*> arguments, Function* func) : func(func), arguments(std::move(arguments)) {
}

Function* Call::getFunc() const {
  return func;
}

Literal* Call::evaluate(Ast &ast) {
  // validation: make sure that both Call and Function have the same number of arguments
  if (this->getArguments().size() != this->getFunc()->getParams().size()) {
    std::stringstream ss;
    ss << "Number of arguments in Call and its called Function does not match (";
    ss << this->getArguments().size() << " vs. " << this->getFunc()->getParams().size();
    throw std::logic_error(ss.str());
  }

  // create vector to store parameter values for Function evaluation
  std::map<std::string, Literal*> paramValues;

  for (int i = 0; i < this->getFunc()->getParams().size(); i++) {
    // validation: make sure that datatypes in Call and Function are equal
    auto datatypeCall = *this->getArguments().at(i)->getDatatype();
    auto datatypeFunc = *this->getFunc()->getParams().at(i).getDatatype();
    if (datatypeCall != datatypeFunc)
      throw std::logic_error("Datatype in Call and Function mismatch! Cannot continue."
                             "Position of parameters in Call and Function must be equal.");

    // variable identifier: retrieve the variable identifier to bind the value to
    auto val = this->getFunc()->getParams().at(i).getValue();
    std::string varIdentifier;
    if (auto var = dynamic_cast<Variable*>(val)) {
      varIdentifier = var->getIdentifier();
    } else {
      throw std::logic_error("FunctionParameter in Call must have a Variable type as value.");
    }

    // variable value: retrieve the variable's value to be passed to the callee
    Literal* lit = this->getArguments().at(i)->getValue()->evaluate(ast);
    // make sure that evaluate returns a Literal
    if (lit == nullptr) throw std::logic_error("There's something wrong! Evaluate should return a Literal here.");

    // store value of lit in vector paramValues with its variable identifier
    // this is to be used to evaluate the Function called by Call
    lit->addLiteralValue(varIdentifier, paramValues);
  }

  // evaluate called function (returns nullptr if function is void)
  Ast subAst(this->getFunc());
  return subAst.evaluateAst(paramValues, false);
}
