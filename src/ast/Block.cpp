#include <iostream>
#include <exception>
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/VarDecl.h"

json Block::toJson() const {
  std::vector<AbstractStatement *> stmts;
  stmts.reserve(countChildrenNonNull());
  for (auto c : getChildrenNonNull()) {
    stmts.push_back(dynamic_cast<AbstractStatement *>(c));
  }
  json j = {{"type", getNodeType()},
            {"statements", stmts}};
  return j;
}

Block::Block(AbstractStatement *stat) {
  this->addChild(stat);
}

Block::Block(std::vector<AbstractStatement *> statements) {
  if (statements.empty()) {
    throw std::logic_error("Block statement vector is empty!"
                           "If this is intended, use the parameter-less constructor instead.");
  }
  addChildren(std::vector<AbstractNode *>(statements.begin(), statements.end()), true);
}

void Block::accept(Visitor &v) {
  v.visit(*this);
}

std::string Block::getNodeType() const {
  return "Block";
}

std::vector<AbstractStatement *> Block::getStatements() const {
  std::vector<AbstractStatement *> stmts;
  stmts.reserve(countChildrenNonNull());
  for (auto c : getChildrenNonNull()) {
    stmts.emplace_back(dynamic_cast<AbstractStatement *>(c));
  }
  return stmts;
}

Block *Block::clone(bool keepOriginalUniqueNodeId) const {
  std::vector<AbstractStatement *> clonedStatements;
  for (auto &statement : this->getStatements()) {
    clonedStatements.push_back(statement->clone(keepOriginalUniqueNodeId)->castTo<AbstractStatement>());
  }
  auto clonedNode = clonedStatements.empty() ? new Block() : new Block(clonedStatements);
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

AbstractNode *Block::cloneFlat() {
  auto block = new Block();
  block->setUniqueNodeId(this->getUniqueNodeId());
  return block;
}

int Block::getMaxNumberChildren() {
  return -1;
}

bool Block::supportsCircuitMode() {
  return true;
}

std::string Block::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

bool Block::isEqual(AbstractStatement *otherBlockStatement) {
  if (auto otherAsBlock = dynamic_cast<Block *>(otherBlockStatement)) {
    auto thisStatements = getStatements();
    auto otherStatements = otherAsBlock->getStatements();

    auto thisStatIt = thisStatements.begin();
    auto otherStatIt = otherStatements.begin();

    // compare statement-per-statement
    for (; thisStatIt!=thisStatements.end() && otherStatIt!=otherStatements.end(); ++thisStatIt) {
      // break up if any pair of two compared statements do not match
      if (!(*thisStatIt)->isEqual(*otherStatIt)) return false;
      otherStatIt++;
    }

    // make sure that both blocks reached their end (otherwise one could have more than the other one but still be
    // recognized as being equal)
    return (thisStatIt==thisStatements.end()) && (otherStatIt==otherStatements.end());
  }
  return false;
}
