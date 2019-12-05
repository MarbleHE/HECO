#include "../../include/ast/Block.h"

#include <utility>


Block::Block(std::vector<std::unique_ptr<AbstractStatement>> *statements) {
    this->statements = statements;
}
