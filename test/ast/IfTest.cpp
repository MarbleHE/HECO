#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "gtest/gtest.h"

/// Helper function to handle dynamic casting/etc
/// \param b A Block that should contain a VariableDeclaration
/// \return The identifier of the variable being declared
std::string getNameFromBlock(const Block &b) {
  const AbstractStatement &s = b.getStatements()[0];
  auto vd = dynamic_cast<const VariableDeclaration &>(s);
  return vd.getTarget().getIdentifier();
}

/// Helper function to handle dynamic casting/etc
/// \param e An AbstractExpression that should be a LiteralBool
/// \return The value of the LiteralBool
bool getValue(const AbstractExpression &e) {
  return dynamic_cast<const LiteralBool &>(e).getValue();
}

TEST(IfTest, values_ValuesGivenInCtorAreRetrievable) {
  // Ifs are created with a condition and branches
  // This test simply confirms that they are retrievable later

  VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
  VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
  If iff(std::make_unique<LiteralBool>(true),
         std::make_unique<Block>(variableDeclaration1.clone()),
         std::make_unique<Block>(variableDeclaration2.clone()));

  ASSERT_TRUE(iff.hasCondition());
  EXPECT_EQ(getValue(iff.getCondition()), true);
  ASSERT_TRUE(iff.hasThenBranch());
  EXPECT_EQ(getNameFromBlock(iff.getThenBranch()), "foo");
  ASSERT_TRUE(iff.hasElseBranch());
  EXPECT_EQ(getNameFromBlock(iff.getElseBranch()), "boo");
}

TEST(IfTest, SetAndGet) {
  // TODO: This test simply checks that target and value can be set and get correctly.


}

TEST(IfTest, CopyCtorCopiesValue) {
  // TODO: When copying a If, the new object should contain a (deep) copy of the condition and branches

}

TEST(IfTest, CopyAssignmentCopiesValue) {
  // TODO: When copying a If, the new object should contain a copy of the condition and branches

}

TEST(IfTest, MoveCtorPreservesValue) {
  // TODO: When moving a If, the new object should contain the same condition and branches
}

TEST(IfTest, MoveAssignmentPreservesValue) {
  // TODO: When moving a If, the new object should contain the same condition and branches
}

TEST(IfTest, countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
  VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
  If iff(std::make_unique<LiteralBool>(true),
         std::make_unique<Block>(variableDeclaration1.clone()),
         std::make_unique<Block>(variableDeclaration2.clone()));
  auto reported_count = iff.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for (auto &c: iff) {
    c = c; // Use c to suppress warning
    ++actual_count;
  }

  EXPECT_EQ(reported_count, actual_count);
}

TEST(IfTest, node_iterate_children) {
  // TODO: This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

}

TEST(IfTest, JsonOutputTest) { /* NOLINT */
  VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
  VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
  If iff(std::make_unique<LiteralBool>(true),
         std::make_unique<Block>(variableDeclaration1.clone()),
         std::make_unique<Block>(variableDeclaration2.clone()));

  nlohmann::json j = {{"type", "If"},
                      {"condition", {
                          {"type", "LiteralBool"},
                          {"value", true}}},
                      {"thenBranch", {
                          {"type", "Block"},
                          {"statements", {
                              {{"type", "VariableDeclaration"},
                               {"datatype", "bool",},
                               {"target", {
                                   {"type", "Variable"},
                                   {"identifier", "foo"}}},
                              }}
                          }
                      }},
                      {"elseBranch", {
                          {"type", "Block"},
                          {"statements", {
                              {{"type", "VariableDeclaration"},
                               {"datatype", "bool",},
                               {"target", {
                                   {"type", "Variable"},
                                   {"identifier", "boo"}}},
                              }}
                          }
                      }
                      }};

  EXPECT_EQ(iff.toJson(), j);
}