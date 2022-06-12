#include "heco/legacy_ast/ast/If.h"
#include "gtest/gtest.h"
#include "heco/legacy_ast/ast/Literal.h"
#include "heco/legacy_ast/ast/UnaryExpression.h"
#include "heco/legacy_ast/ast/Variable.h"
#include "heco/legacy_ast/ast/VariableDeclaration.h"

/// Helper function to handle dynamic casting/etc
/// \param b A Block that should contain a VariableDeclaration
/// \return The identifier of the variable being declared
std::string getNameFromVariable(const Block &b)
{
    const AbstractStatement &s = b.getStatements()[0];
    auto vd = dynamic_cast<const VariableDeclaration &>(s);
    return vd.getTarget().getIdentifier();
}

/// Helper function to handle dynamic casting/etc
/// \param e An AbstractExpression that should be a LiteralBool
/// \return The value of the LiteralBool
inline bool getValue(const AbstractExpression &e)
{
    return dynamic_cast<const LiteralBool &>(e).getValue();
}

TEST(IfTest, values_ValuesGivenInCtorAreRetrievable)
{
    // Ifs are created with a condition and branches
    // This test simply confirms that they are retrievable later

    VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
    VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
    If iff(
        std::make_unique<LiteralBool>(true), std::make_unique<Block>(variableDeclaration1.clone(nullptr)),
        std::make_unique<Block>(variableDeclaration2.clone(nullptr)));

    ASSERT_TRUE(iff.hasCondition());
    EXPECT_EQ(getValue(iff.getCondition()), true);
    ASSERT_TRUE(iff.hasThenBranch());
    EXPECT_EQ(getNameFromVariable(iff.getThenBranch()), "foo");
    ASSERT_TRUE(iff.hasElseBranch());
    EXPECT_EQ(getNameFromVariable(iff.getElseBranch()), "boo");
}

TEST(IfTest, SetAndGet)
{
    // This test simply checks that target and value can be set and get correctly.

    If iff(std::make_unique<LiteralBool>(true), std::make_unique<Block>(), std::make_unique<Block>());

    iff.setCondition(std::make_unique<LiteralBool>(false));
    EXPECT_EQ(getValue(iff.getCondition()), false);

    auto variableDeclaration2 = std::make_unique<VariableDeclaration>(
        Datatype(Type::BOOL), std::make_unique<Variable>("bar"), std::make_unique<LiteralBool>(true));
    iff.setThenBranch(std::make_unique<Block>(std::move(variableDeclaration2)));
    ASSERT_TRUE(iff.hasThenBranch());
    EXPECT_EQ(getNameFromVariable(iff.getThenBranch()), "bar");

    auto variableDeclaration3 = std::make_unique<VariableDeclaration>(
        Datatype(Type::BOOL), std::make_unique<Variable>("mau"), std::make_unique<LiteralBool>(false));
    iff.setElseBranch(std::make_unique<Block>(std::move(variableDeclaration3)));
    ASSERT_TRUE(iff.hasElseBranch());
    EXPECT_EQ(getNameFromVariable(iff.getElseBranch()), "mau");
}

TEST(IfTest, CopyCtorCopiesValue)
{
    // When copying a If, the new object should contain a (deep) copy of the condition and branches

    auto variableDeclaration1 =
        std::make_unique<VariableDeclaration>(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
    auto variableDeclaration2 =
        std::make_unique<VariableDeclaration>(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
    If iff(
        std::make_unique<LiteralBool>(true), std::make_unique<Block>(std::move(variableDeclaration1)),
        std::make_unique<Block>(std::move(variableDeclaration1)));

    // create a copy by using the copy constructor
    If copiedIff(iff);

    // check if copy is a deep copy
    ASSERT_NE(iff.getCondition(), copiedIff.getCondition());
    ASSERT_NE(iff.getThenBranch(), copiedIff.getThenBranch());
    ASSERT_NE(iff.getElseBranch(), copiedIff.getElseBranch());
}

TEST(IfTest, CopyAssignmentCopiesValue)
{
    // When copying a If, the new object should contain a copy of the condition and branches

    auto variableDeclaration1 =
        std::make_unique<VariableDeclaration>(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
    auto variableDeclaration2 =
        std::make_unique<VariableDeclaration>(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
    If iff(
        std::make_unique<LiteralBool>(true), std::make_unique<Block>(std::move(variableDeclaration1)),
        std::make_unique<Block>(std::move(variableDeclaration2)));

    If copiedIff(std::make_unique<LiteralBool>(false), std::make_unique<Block>(), std::make_unique<Block>());
    // create a copy by using the copy constructor
    copiedIff = iff;

    // check if copy is a deep copy
    ASSERT_NE(iff.getCondition(), copiedIff.getCondition());
    ASSERT_NE(iff.getThenBranch(), copiedIff.getThenBranch());
    ASSERT_NE(iff.getElseBranch(), copiedIff.getElseBranch());
}

TEST(IfTest, MoveCtorPreservesValue)
{
    // When moving a If, the new object should contain the same condition and branches

    VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
    VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
    If iff(
        std::make_unique<LiteralBool>(true), std::make_unique<Block>(variableDeclaration1.clone()),
        std::make_unique<Block>(variableDeclaration2.clone()));

    If newIff(std::move(iff));

    ASSERT_FALSE(iff.hasCondition());
    EXPECT_THROW(iff.getCondition(), std::runtime_error);
    ASSERT_FALSE(iff.hasThenBranch());
    EXPECT_THROW(iff.getThenBranch(), std::runtime_error);
    ASSERT_FALSE(iff.hasElseBranch());
    EXPECT_THROW(iff.getElseBranch(), std::runtime_error);

    ASSERT_TRUE(newIff.hasCondition());
    EXPECT_EQ(getValue(newIff.getCondition()), true);
    ASSERT_TRUE(newIff.hasThenBranch());
    EXPECT_EQ(getNameFromVariable(newIff.getThenBranch()), "foo");
    ASSERT_TRUE(newIff.hasElseBranch());
    EXPECT_EQ(getNameFromVariable(newIff.getElseBranch()), "boo");
}

TEST(IfTest, MoveAssignmentPreservesValue)
{
    // When moving a If, the new object should contain the same condition and branches

    VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
    VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
    If iff(
        std::make_unique<LiteralBool>(true), std::make_unique<Block>(variableDeclaration1.clone()),
        std::make_unique<Block>(variableDeclaration2.clone()));

    If newIff = std::move(iff);

    ASSERT_FALSE(iff.hasCondition());
    EXPECT_THROW(iff.getCondition(), std::runtime_error);
    ASSERT_FALSE(iff.hasThenBranch());
    EXPECT_THROW(iff.getThenBranch(), std::runtime_error);
    ASSERT_FALSE(iff.hasElseBranch());
    EXPECT_THROW(iff.getElseBranch(), std::runtime_error);

    ASSERT_TRUE(newIff.hasCondition());
    EXPECT_EQ(getValue(newIff.getCondition()), true);
    ASSERT_TRUE(newIff.hasThenBranch());
    EXPECT_EQ(getNameFromVariable(newIff.getThenBranch()), "foo");
    ASSERT_TRUE(newIff.hasElseBranch());
    EXPECT_EQ(getNameFromVariable(newIff.getElseBranch()), "boo");
}

TEST(IfTest, countChildrenReportsCorrectNumber)
{
    // This tests checks that countChildren delivers the correct number

    VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
    VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
    If iff(
        std::make_unique<LiteralBool>(true), std::make_unique<Block>(variableDeclaration1.clone()),
        std::make_unique<Block>(variableDeclaration2.clone()));
    auto reported_count = iff.countChildren();

    // Iterate through all the children using the iterators
    // (indirectly via range-based for loop for conciseness)
    size_t actual_count = 0;
    for (auto &c : iff)
    {
        c = c; // Use c to suppress warning
        ++actual_count;
    }

    EXPECT_EQ(reported_count, actual_count);
}

TEST(IfTest, node_iterate_children)
{
    // This test checks that we can iterate correctly through the children
    // Even if some of the elements are null (in which case they should not appear)

    auto variableDeclaration1 =
        std::make_unique<VariableDeclaration>(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
    auto variableDeclaration2 =
        std::make_unique<VariableDeclaration>(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
    If iff(
        {}, std::make_unique<Block>(std::move(variableDeclaration1)),
        std::make_unique<Block>(std::move(variableDeclaration2)));

    auto it = iff.begin();
    AbstractNode &child1 = *it;
    ++it;
    AbstractNode &child2 = *it;
    ++it;

    EXPECT_EQ(it, iff.end());

    auto getNameFromDeclaration = [](const AbstractStatement &s) {
        auto vd = dynamic_cast<const VariableDeclaration &>(s);
        return vd.getTarget().getIdentifier();
    };

    // we need to extract the child at index 0 from the if/else block
    EXPECT_EQ(getNameFromDeclaration(dynamic_cast<Block &>(child1).getStatements()[0]), "foo");
    EXPECT_EQ(getNameFromDeclaration(dynamic_cast<Block &>(child2).getStatements()[0]), "boo");
}
TEST(IfTest, JsonOutputTest)
{ /* NOLINT */
    VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
    VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
    If iff(
        std::make_unique<LiteralBool>(true), std::make_unique<Block>(variableDeclaration1.clone()),
        std::make_unique<Block>(variableDeclaration2.clone()));

    nlohmann::json j = { { "type", "If" },
                         { "condition", { { "type", "LiteralBool" }, { "value", true } } },
                         { "thenBranch",
                           { { "type", "Block" },
                             { "statements",
                               { {
                                   { "type", "VariableDeclaration" },
                                   {
                                       "datatype",
                                       "bool",
                                   },
                                   { "target", { { "type", "Variable" }, { "identifier", "foo" } } },
                               } } } } },
                         { "elseBranch",
                           { { "type", "Block" },
                             { "statements",
                               { {
                                   { "type", "VariableDeclaration" },
                                   {
                                       "datatype",
                                       "bool",
                                   },
                                   { "target", { { "type", "Variable" }, { "identifier", "boo" } } },
                               } } } } } };

    EXPECT_EQ(iff.toJson(), j);
}
