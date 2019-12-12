#include "../../include/ast/Block.h"

#include <utility>
#include <iostream>
#include <VarDecl.h>


Block::Block() = default;

void Block::addStatement(std::unique_ptr<AbstractStatement> &&statement) {
    blockStatements.push_back(std::move(statement));
}

json Block::toJson() const {
    json j;
    j["type"] = "Block";
    j["statements"] = this->blockStatements;
    return j;
}

Block::Block(AbstractStatement *stat) {
    blockStatements.emplace_back(stat);
}

Block::Block(std::vector<AbstractStatement *> *statements) : statements(statements) {

}

void Block::accept(Visitor &v) {
    v.visit(*this);
}

