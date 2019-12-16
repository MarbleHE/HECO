#include "../../include/ast/Block.h"

#include <utility>
#include <iostream>
#include <VarDecl.h>


Block::Block() = default;

json Block::toJson() const {
    json j;
    j["type"] = getNodeName();
    //j["statements"] = *this->statements; // FIXME
    return j;
}


Block::Block(AbstractStatement *stat) {
    auto *vec = new std::vector<AbstractStatement *>;
    vec->emplace_back(stat);
    this->statements = vec;
}

Block::Block(std::vector<AbstractStatement *> *statements) {
    this->statements = statements;
}

void Block::accept(Visitor &v) {
    v.visit(*this);
}

std::string Block::getNodeName() const {
    return "Block";
}

std::vector<AbstractStatement *> *Block::getStatements() const {
    return statements;
}

