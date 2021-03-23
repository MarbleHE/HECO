#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/TernaryOperator.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "gtest/gtest.h"

/// Helper function to handle dynamic casting/etc
/// \param e A Block that should contain a VariableDeclaration
/// \return The identifier of the variable being declared
std::string getNameFromVariable(const AbstractExpression &e) {
  auto vd = dynamic_cast<const Variable &>(e);
  return vd.getIdentifier();
}

/// Helper function to handle dynamic casting/etc
/// \param e An AbstractExpression that should be a LiteralBool
/// \return The value of the LiteralBool
inline bool getValue(const AbstractExpression &e) {
  return dynamic_cast<const LiteralBool &>(e).getValue();
}

TEST(TernaryOperatorTest, values_ValuesGivenInCtorAreRetrievable) {
  // TernaryOperators are created with a condition and branches
  // This test simply confirms that they are retrievable later

  TernaryOperator ternaryOperator(std::make_unique<LiteralBool>(true),
                                  std::make_unique<Variable>("foo"),
                                  std::make_unique<Variable>("boo"));

  ASSERT_TRUE(ternaryOperator.hasCondition());
  EXPECT_EQ(getValue(ternaryOperator.getCondition()), true);
  ASSERT_TRUE(ternaryOperator.hasThenExpr());
  EXPECT_EQ(getNameFromVariable(ternaryOperator.getThenExpr()), "foo");
  ASSERT_TRUE(ternaryOperator.hasElseExpr());
  EXPECT_EQ(getNameFromVariable(ternaryOperator.getElseExpr()), "boo");
}

//TODO: Finish adapting tests
TEST(TernaryOperatorTest, SetAndGet) {
  // This test simply checks that target and value can be set and get correctly.

//  TernaryOperator ternaryOperator(std::make_unique<LiteralBool>(true),
//                                  nullptr,
//                                  nullptr);
//
//  iff.setCondition(std::make_unique<LiteralBool>(false));
//  EXPECT_EQ(getValue(iff.getCondition()), false);
//
//  auto variableDeclaration2 = std::make_unique<VariableDeclaration>(Datatype(Type::BOOL),
//                                                                    std::make_unique<Variable>("bar"),
//                                                                    std::make_unique<LiteralBool>(true));
//  iff.setThenExpr(std::make_unique<Block>(std::move(variableDeclaration2)));
//  ASSERT_TRUE(iff.hasThenExpr());
//  EXPECT_EQ(getNameFromVariable(iff.getThenExpr()), "bar");
//
//  auto variableDeclaration3 = std::make_unique<VariableDeclaration>(Datatype(Type::BOOL),
//                                                                    std::make_unique<Variable>("mau"),
//                                                                    std::make_unique<LiteralBool>(false));
//  iff.setElseExpr(std::make_unique<Block>(std::move(variableDeclaration3)));
//  ASSERT_TRUE(iff.hasElseExpr());
//  EXPECT_EQ(getNameFromVariable(iff.getElseExpr()), "mau");
}

TEST(TernaryOperatorTest, CopyCtorCopiesValue) {
  // When copying a TernaryOperator, the new object should contain a (deep) copy of the condition and branches

//  auto variableDeclaration1 =
//      std::make_unique<VariableDeclaration>(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
//  auto variableDeclaration2 =
//      std::make_unique<VariableDeclaration>(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
//  TernaryOperator iff(std::make_unique<LiteralBool>(true),
//         std::make_unique<Block>(std::move(variableDeclaration1)),
//         std::make_unique<Block>(std::move(variableDeclaration1)));
//
//  // create a copy by using the copy constructor
//  TernaryOperator copiedTernaryOperatorf(iff);
//
//  // check if copy is a deep copy
//  ASSERT_NE(iff.getCondition(), copiedTernaryOperatorf.getCondition());
//  ASSERT_NE(iff.getThenExpr(), copiedTernaryOperatorf.getThenExpr());
//  ASSERT_NE(iff.getElseExpr(), copiedTernaryOperatorf.getElseExpr());
}

TEST(TernaryOperatorTest, CopyAssignmentCopiesValue) {
  // When copying a TernaryOperator, the new object should contain a copy of the condition and branches

//  auto variableDeclaration1 = std::make_unique<VariableDeclaration>(
//      Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
//  auto variableDeclaration2 = std::make_unique<VariableDeclaration>(
//      Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
//  TernaryOperator iff(std::make_unique<LiteralBool>(true),
//         std::make_unique<Block>(std::move(variableDeclaration1)),
//         std::make_unique<Block>(std::move(variableDeclaration2)));
//
//  TernaryOperator copiedTernaryOperatorf(std::make_unique<LiteralBool>(false), std::make_unique<Block>(), std::make_unique<Block>());
//  // create a copy by using the copy constructor
//  copiedTernaryOperatorf = iff;
//
//  // check if copy is a deep copy
//  ASSERT_NE(iff.getCondition(), copiedTernaryOperatorf.getCondition());
//  ASSERT_NE(iff.getThenExpr(), copiedTernaryOperatorf.getThenExpr());
//  ASSERT_NE(iff.getElseExpr(), copiedTernaryOperatorf.getElseExpr());
}

TEST(TernaryOperatorTest, MoveCtorPreservesValue) {
  // When moving a TernaryOperator, the new object should contain the same condition and branches

//  VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
//  VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
//  TernaryOperator iff(std::make_unique<LiteralBool>(true),
//         std::make_unique<Block>(variableDeclaration1.clone()),
//         std::make_unique<Block>(variableDeclaration2.clone()));
//
//  TernaryOperator newTernaryOperatorf(std::move(iff));
//
//  ASSERT_FALSE(iff.hasCondition());
//  EXPECT_THROW(iff.getCondition(), std::runtime_error);
//  ASSERT_FALSE(iff.hasThenExpr());
//  EXPECT_THROW(iff.getThenExpr(), std::runtime_error);
//  ASSERT_FALSE(iff.hasElseExpr());
//  EXPECT_THROW(iff.getElseExpr(), std::runtime_error);
//
//  ASSERT_TRUE(newTernaryOperatorf.hasCondition());
//  EXPECT_EQ(getValue(newTernaryOperatorf.getCondition()), true);
//  ASSERT_TRUE(newTernaryOperatorf.hasThenExpr());
//  EXPECT_EQ(getNameFromVariable(newTernaryOperatorf.getThenExpr()), "foo");
//  ASSERT_TRUE(newTernaryOperatorf.hasElseExpr());
//  EXPECT_EQ(getNameFromVariable(newTernaryOperatorf.getElseExpr()), "boo");
}

TEST(TernaryOperatorTest, MoveAssignmentPreservesValue) {
  // When moving a TernaryOperator, the new object should contain the same condition and branches

//  VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
//  VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
//  TernaryOperator iff(std::make_unique<LiteralBool>(true),
//         std::make_unique<Block>(variableDeclaration1.clone()),
//         std::make_unique<Block>(variableDeclaration2.clone()));
//
//  TernaryOperator newTernaryOperatorf = std::move(iff);
//
//  ASSERT_FALSE(iff.hasCondition());
//  EXPECT_THROW(iff.getCondition(), std::runtime_error);
//  ASSERT_FALSE(iff.hasThenExpr());
//  EXPECT_THROW(iff.getThenExpr(), std::runtime_error);
//  ASSERT_FALSE(iff.hasElseExpr());
//  EXPECT_THROW(iff.getElseExpr(), std::runtime_error);
//
//  ASSERT_TRUE(newTernaryOperatorf.hasCondition());
//  EXPECT_EQ(getValue(newTernaryOperatorf.getCondition()), true);
//  ASSERT_TRUE(newTernaryOperatorf.hasThenExpr());
//  EXPECT_EQ(getNameFromVariable(newTernaryOperatorf.getThenExpr()), "foo");
//  ASSERT_TRUE(newTernaryOperatorf.hasElseExpr());
//  EXPECT_EQ(getNameFromVariable(newTernaryOperatorf.getElseExpr()), "boo");
}

TEST(TernaryOperatorTest, countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

//  VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
//  VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
//  TernaryOperator iff(std::make_unique<LiteralBool>(true),
//         std::make_unique<Block>(variableDeclaration1.clone()),
//         std::make_unique<Block>(variableDeclaration2.clone()));
//  auto reported_count = iff.countChildren();
//
//  // Iterate through all the children using the iterators
//  // (indirectly via range-based for loop for conciseness)
//  size_t actual_count = 0;
//  for (auto &c: iff) {
//    c = c; // Use c to suppress warning
//    ++actual_count;
//  }
//
//  EXPECT_EQ(reported_count, actual_count);
}

TEST(TernaryOperatorTest, node_iterate_children) {
  // This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

//  auto variableDeclaration1 = std::make_unique<VariableDeclaration>(
//      Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
//  auto variableDeclaration2 = std::make_unique<VariableDeclaration>(
//      Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
//  TernaryOperator iff({},
//         std::make_unique<Block>(std::move(variableDeclaration1)),
//         std::make_unique<Block>(std::move(variableDeclaration2)));
//
//  auto it = iff.begin();
//  AbstractNode &child1 = *it;
//  ++it;
//  AbstractNode &child2 = *it;
//  ++it;
//
//  EXPECT_EQ(it, iff.end());
//
//  auto getNameFromDeclaration = [](const AbstractStatement &s) {
//    auto vd = dynamic_cast<const VariableDeclaration &>(s);
//    return vd.getTarget().getIdentifier();
//  };
//
//  // we need to extract the child at index 0 from the if/else block
//  EXPECT_EQ(getNameFromDeclaration(dynamic_cast<Block &>(child1).getStatements()[0]), "foo");
//  EXPECT_EQ(getNameFromDeclaration(dynamic_cast<Block &>(child2).getStatements()[0]), "boo");
}
TEST(TernaryOperatorTest, JsonOutputTest) { /* NOLINT */
//  VariableDeclaration variableDeclaration1(Datatype(Type::BOOL), std::make_unique<Variable>("foo"));
//  VariableDeclaration variableDeclaration2(Datatype(Type::BOOL), std::make_unique<Variable>("boo"));
//  TernaryOperator iff(std::make_unique<LiteralBool>(true),
//         std::make_unique<Block>(variableDeclaration1.clone()),
//         std::make_unique<Block>(variableDeclaration2.clone()));
//
//  nlohmann::json j = {{"type", "TernaryOperator"},
//                      {"condition", {
//                          {"type", "LiteralBool"},
//                          {"value", true}}},
//                      {"thenExpr", {
//                          {"type", "Block"},
//                          {"statements", {
//                              {{"type", "VariableDeclaration"},
//                               {"datatype", "bool",},
//                               {"target", {
//                                   {"type", "Variable"},
//                                   {"identifier", "foo"}}},
//                              }}
//                          }
//                      }},
//                      {"elseExpr", {
//                          {"type", "Block"},
//                          {"statements", {
//                              {{"type", "VariableDeclaration"},
//                               {"datatype", "bool",},
//                               {"target", {
//                                   {"type", "Variable"},
//                                   {"identifier", "boo"}}},
//                              }}
//                          }
//                      }
//                      }};
//
//  EXPECT_EQ(iff.toJson(), j);
}
