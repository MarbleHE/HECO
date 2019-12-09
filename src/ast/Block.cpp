#include "../../include/ast/Block.h"

#include <utility>
#include <iostream>
#include <VarDecl.h>


Block::Block() = default;

void Block::addStatement(std::unique_ptr<AbstractStatement> &&statement) {
    blockStatements.push_back(std::move(statement));
}
