#include "heco/legacy_ast/ast/Block.h"
#include <exception>
#include <iostream>
#include "heco/legacy_ast/ast_parser/Parser.h"
#include "heco/legacy_ast/ast_utilities/IVisitor.h"

/// Convenience typedef for conciseness
typedef std::unique_ptr<AbstractStatement> stmtPtr;

Block::~Block() = default;

Block::Block() = default;

Block::Block(std::unique_ptr<AbstractStatement> statement)
{
    statements = std::vector<stmtPtr>(1);
    statements[0] = std::move(statement);
}

Block::Block(std::vector<std::unique_ptr<AbstractStatement>> &&vectorOfStatements)
    : statements(std::move(vectorOfStatements)){};

Block::Block(const Block &other)
{
    // deep-copy the statements, including nullptrs
    statements.reserve(other.statements.size());
    for (auto &s : other.statements)
    {
        statements.emplace_back(s ? s->clone(this) : nullptr);
    }
}

Block::Block(Block &&other) noexcept : statements(std::move(other.statements))
{}

Block &Block::operator=(const Block &other)
{
    statements.clear();
    // deep-copy the statements, including nullptrs
    statements.reserve(other.statements.size());
    for (auto &s : other.statements)
    {
        statements.emplace_back(s ? s->clone(this) : nullptr);
    }
    return *this;
}
Block &Block::operator=(Block &&other) noexcept
{
    statements = std::move(other.statements);
    return *this;
}
std::unique_ptr<Block> Block::clone(AbstractNode *parent_) const
{
    return std::unique_ptr<Block>(clone_impl(parent_));
}

bool Block::isEmpty()
{
    return countChildren() == 0;
}

bool Block::hasNullStatements()
{
    // Because std::unique_ptr doesn't have copy, we can't use std::count_if
    size_t count = 0;
    for (auto &s : statements)
    {
        if (s == nullptr)
        {
            count++;
        }
    }
    return count != 0;
}

std::vector<std::unique_ptr<AbstractStatement>> &Block::getStatementPointers()
{
    return statements;
}

std::vector<std::reference_wrapper<AbstractStatement>> Block::getStatements()
{
    std::vector<std::reference_wrapper<AbstractStatement>> r;
    for (auto &s : statements)
    {
        if (s != nullptr)
        {
            r.emplace_back(*s);
        }
    }
    return r;
}

std::vector<std::reference_wrapper<const AbstractStatement>> Block::getStatements() const
{
    std::vector<std::reference_wrapper<const AbstractStatement>> r;
    for (auto &s : statements)
    {
        if (s != nullptr)
        {
            r.emplace_back(*s);
        }
    }
    return r;
}

void Block::appendStatement(std::unique_ptr<AbstractStatement> statement)
{
    statements.emplace_back(std::move(statement));
}

void Block::prependStatement(std::unique_ptr<AbstractStatement> statement)
{
    statements.insert(statements.begin(), std::move(statement));
}

void Block::removeNullStatements()
{
    std::vector<stmtPtr> new_statements;
    for (auto &s : statements)
    {
        if (s != nullptr)
        {
            new_statements.emplace_back(std::move(s));
        }
    }
    statements = std::move(new_statements);
}
///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
Block *Block::clone_impl(AbstractNode *parent_) const
{
    auto p = new Block(*this);
    if (parent_)
    {
        p->setParent(*parent_);
    }
    return p;
}

void Block::accept(IVisitor &v)
{
    v.visit(*this);
}

AbstractNode::iterator Block::begin()
{
    return AbstractNode::iterator(
        std::make_unique<BlockIteratorImpl<AbstractNode>>(*this, statements.begin(), statements.end()));
}

AbstractNode::const_iterator Block::begin() const
{
    return AbstractNode::const_iterator(
        std::make_unique<BlockIteratorImpl<const AbstractNode>>(*this, statements.begin(), statements.end()));
}

AbstractNode::iterator Block::end()
{
    return AbstractNode::iterator(
        std::make_unique<BlockIteratorImpl<AbstractNode>>(*this, statements.end(), statements.end()));
}

AbstractNode::const_iterator Block::end() const
{
    return AbstractNode::const_iterator(
        std::make_unique<BlockIteratorImpl<const AbstractNode>>(*this, statements.end(), statements.end()));
}

size_t Block::countChildren() const
{
    // Only non-null entries in the vector are counted as children
    // Because std::unique_ptr doesn't have copy, we can't use std::count_if
    size_t count = 0;
    for (auto &s : statements)
    {
        if (s != nullptr)
        {
            count++;
        }
    }
    return count;
}

nlohmann::json Block::toJson() const
{
    std::vector<std::reference_wrapper<const AbstractStatement>> stmts = getStatements();
    std::vector<nlohmann::json> stmtsJson;
    for (const AbstractStatement &s : stmts)
    {
        stmtsJson.push_back(s.toJson());
    }
    nlohmann::json j = { { "type", getNodeType() }, { "statements", stmtsJson } };
    return j;
}

std::unique_ptr<Block> Block::fromJson(nlohmann::json j)
{
    std::vector<std::unique_ptr<AbstractStatement>> statements;
    for (auto statement : j["statements"])
    {
        statements.emplace_back(Parser::parseJsonStatement(statement));
    }
    return std::make_unique<Block>(std::move(statements));
}

std::string Block::toString(bool printChildren) const
{
    return AbstractNode::toStringHelper(printChildren, {});
}

std::string Block::getNodeType() const
{
    return "Block";
}
