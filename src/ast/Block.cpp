#include "../../include/ast/Block.h"

#include <iostream>
#include <VarDecl.h>

Block::Block() {
  statements = nullptr;
}

json Block::toJson() const {
  json j = {{"type", getNodeName()},
            {"statements", *this->statements}};
  return j;
}

Block::Block(AbstractStatement* stat) {
  auto* vec = new std::vector<AbstractStatement*>;
  vec->emplace_back(stat);
  this->statements = vec;
}

Block::Block(std::vector<AbstractStatement*>* statements) {
  this->statements = statements;
}

void Block::accept(Visitor &v) {
  v.visit(*this);
}

std::string Block::getNodeName() const {
  return "Block";
}

std::vector<AbstractStatement*>* Block::getStatements() const {
  return statements;
}

Block::~Block() {
  delete statements;
}

Literal* Block::evaluate(Ast &ast) {
  for (auto stmt : *getStatements()) {
    (void) stmt->evaluate(ast);
  }
  return nullptr;
}

