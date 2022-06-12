#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_UNARYEXPRESSION_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_UNARYEXPRESSION_H_

#include <string>
#include "heco/legacy_ast/ast/AbstractExpression.h"
#include "heco/legacy_ast/ast_utilities/Operator.h"

/// A UnaryExpression has a single operand and an Operator
class UnaryExpression : public AbstractExpression
{
private:
    /// Operand
    std::unique_ptr<AbstractExpression> operand;

    /// Operator (not part of node class hierarchy, i.e. not a child)
    Operator op;

    /// Creates a deep copy of the current node
    /// Should be used only by Nodes' clone()
    /// \return a copy of the current node
    UnaryExpression *clone_impl(AbstractNode *parent) const override;

public:
    /// Destructor
    ~UnaryExpression() override;

    /// Create a UnaryExpression with left hand side, operator and right hand side
    /// \param operand The operand (UnaryExpression will take ownership)
    /// \param operator Operator for this expression
    UnaryExpression(std::unique_ptr<AbstractExpression> operand, Operator op);

    /// Copy constructor
    /// \param other UnaryExpression to copy
    UnaryExpression(const UnaryExpression &other);

    /// Move constructor
    /// \param other UnaryExpression to copy
    UnaryExpression(UnaryExpression &&other) noexcept;

    /// Copy assignment
    /// \param other UnaryExpression to copy
    /// \return This object
    UnaryExpression &operator=(const UnaryExpression &other);

    /// Move assignment
    /// \param other UnaryExpression to move
    /// \return This object
    UnaryExpression &operator=(UnaryExpression &&other) noexcept;

    /// Deep copy of the current node
    /// \return A deep copy of the current node
    std::unique_ptr<UnaryExpression> clone(AbstractNode *parent = nullptr) const;

    /// Does this UnaryExpression have its left hand side set?
    /// \return true iff the assignment has the left hand side set
    bool hasOperand() const;

    /// Does this UnaryExpression have its operator set?
    /// Note: Currently always true
    bool hasOperator() const;

    /// Get (a reference to) the operand (if it exists)
    /// \return A reference to the operand
    /// \throws std::runtime_error if no operand exists
    AbstractExpression &getOperand();

    /// Get (a const reference to) the operand (if it exists)
    /// \return A const reference to the operand
    /// \throws std::runtime_error if no operand exists
    const AbstractExpression &getOperand() const;

    /// Get (a reference to) the Operator
    /// \return A reference to the operator variable
    Operator &getOperator();

    /// Get (a const reference to) the Operator
    /// \return A const reference to the operator variable
    const Operator &getOperator() const;

    /// Set the operand to newOperand, taking ownership of newOperand
    /// This will delete the previous operand!
    /// \param newOperand new operand to set
    void setOperand(std::unique_ptr<AbstractExpression> newOperand);

    /// Set the operator to newOperator
    /// \param newOperator new operator to set
    void setOperator(Operator newOperator);

    ///////////////////////////////////////////////
    ////////// AbstractNode Interface /////////////
    ///////////////////////////////////////////////
    void accept(IVisitor &v) override;
    iterator begin() override;
    const_iterator begin() const override;
    iterator end() override;
    const_iterator end() const override;
    size_t countChildren() const override;
    nlohmann::json toJson() const override;
    std::string toString(bool printChildren) const override;

protected:
    std::string getNodeType() const override;
};

// Designed to be instantiated only with T = (const) AbstractNode
template <typename T>
class UnaryExpressionIteratorImpl : public PositionIteratorImpl<T, UnaryExpression>
{
public:
    // Inherit the constructor from the base class since it does everything we need
    using PositionIteratorImpl<T, UnaryExpression>::PositionIteratorImpl;

    T &operator*() override
    {
        switch (this->position)
        {
        case 0:
            if (this->node.hasOperand())
                return this->node.getOperand();
            else
                throw std::runtime_error("Cannot dereference iterator since node has no children.");
        default:
            // calling dereference on higher elements is an error
            throw std::runtime_error("Trying to dereference iterator past end.");
        }
    }

    std::unique_ptr<BaseIteratorImpl<T>> clone() override
    {
        return std::make_unique<UnaryExpressionIteratorImpl>(this->node, this->position);
    }
};

#endif // AST_OPTIMIZER_INCLUDE_AST_OPT_AST_UNARYEXPRESSION_H_
