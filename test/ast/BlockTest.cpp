#include <ast_opt/ast_parser/Parser.h>
#include "ASTComparison.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "gtest/gtest.h"

/// Helper function to handle dynamic casting/etc
/// \param s An Abstract statement that should be a VariableDeclaration
/// \return The identifier of the variable being declared
std::string getNameFromDeclaration(const AbstractStatement &s) {
  auto vd = dynamic_cast<const VariableDeclaration &>(s);
  return vd.getTarget().getIdentifier();
}

TEST(BlockTest, values_ValuesGivenInCtorAreRetrievable) {
  // This test simply confirms that statements supplied via Ctor are retrievable later

  // Single-argument Ctor
  Block blockFromSingleStmtCtor(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));

  // Vector Ctor
  std::vector<std::unique_ptr<AbstractStatement>> statements;
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(true)));
  Block blockFromVector(std::move(statements));

  ASSERT_EQ(blockFromSingleStmtCtor.countChildren(), 1);
  EXPECT_EQ(getNameFromDeclaration(blockFromSingleStmtCtor.getStatements()[0]), "foo");
  ASSERT_EQ(blockFromVector.countChildren(), 2);
  EXPECT_EQ(getNameFromDeclaration(blockFromVector.getStatements()[0]), "foo");
  EXPECT_EQ(getNameFromDeclaration(blockFromVector.getStatements()[1]), "boo");
}

TEST(BlockTest, CopyCtorCopiesValue) {
  // When copying a Block, the new object should contain a (deep) copy of all the statements

  Block block(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));

  Block copy_block(block);

  ASSERT_EQ(copy_block.countChildren(), 1);
  EXPECT_EQ(getNameFromDeclaration(copy_block.getStatements()[0]), "foo");
}

TEST(BlockTest, CopyAssignmentCopiesValue) {
  // When copying a Block, the new object should contain a (deep) copy of all the statements

  Block block(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));

  Block copy_block(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(true)));
  copy_block = block;

  ASSERT_EQ(copy_block.countChildren(), 1);
  EXPECT_EQ(getNameFromDeclaration(copy_block.getStatements()[0]), "foo");
}

TEST(BlockTest, MoveCtorPreservesValue) {
  // When moving a Block, the new object should contain the same statements

  Block block(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));

  Block new_block(std::move(block));

  ASSERT_EQ(new_block.countChildren(), 1);
  EXPECT_EQ(getNameFromDeclaration(new_block.getStatements()[0]), "foo");
}

TEST(BlockTest, MovedAssignmentPreservesValue) {
  // When moving a Block, the new object should contain the same statements

  Block block(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));

  Block new_block(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(true)));
  new_block = std::move(block);

  ASSERT_EQ(new_block.countChildren(), 1);
  EXPECT_EQ(getNameFromDeclaration(new_block.getStatements()[0]), "foo");
}

TEST(BlockTest, NullStatementRemoval) {
  // Removing null statements should not affect the other children

  std::vector<std::unique_ptr<AbstractStatement>> statements;
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));
  statements.emplace_back(nullptr);
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(true)));
  Block block(std::move(statements));

  block.removeNullStatements();

  EXPECT_FALSE(block.hasNullStatements());
  ASSERT_EQ(block.countChildren(), 2);
  EXPECT_EQ(getNameFromDeclaration(block.getStatements()[0]), "foo");
  EXPECT_EQ(getNameFromDeclaration(block.getStatements()[1]), "boo");
}

TEST(BlockTest, CountChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  Block block(std::make_unique<VariableDeclaration>(Datatype(Type::BOOL),
                                                    std::make_unique<Variable>("foo"),
                                                    std::make_unique<LiteralBool>(true)));
  auto reported_count = block.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for (auto &c: block) {
    c = c; // Use c to suppress warning
    ++actual_count;
  }

  EXPECT_EQ(reported_count, actual_count);
}

TEST(BlockTest, node_iterate_children) {
  // This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear

  std::vector<std::unique_ptr<AbstractStatement>> statements;
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));
  statements.emplace_back(nullptr);
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(true)));
  Block block(std::move(statements));

  auto it = block.begin();
  AbstractNode &child1 = *it;
  ++it;
  AbstractNode &child2 = *it;
  ++it;
  // Iterator should now be at end

  EXPECT_EQ(it, block.end());
  EXPECT_EQ(getNameFromDeclaration(dynamic_cast<VariableDeclaration &>(child1)), "foo");
  EXPECT_EQ(getNameFromDeclaration(dynamic_cast<VariableDeclaration &>(child2)), "boo");
}

TEST(BlockTest, appendStatement) {
  // This test checks that we can append a Statement to the Block

  Block block(std::make_unique<VariableDeclaration>(Datatype(Type::BOOL),
                                                    std::make_unique<Variable>("foo"),
                                                    std::make_unique<LiteralBool>(true)));

  block.appendStatement(std::make_unique<VariableDeclaration>(Datatype(Type::BOOL),
                                                              std::make_unique<Variable>("boo"),
                                                              std::make_unique<LiteralBool>(true)));

  ASSERT_EQ(block.countChildren(), 2);
  EXPECT_EQ(getNameFromDeclaration(block.getStatements()[0]), "foo");
  EXPECT_EQ(getNameFromDeclaration(block.getStatements()[1]), "boo");
}

TEST(BlockTest, prependStatement) {
  // This test checks that we can prepend a Statement to the Block

  // This test checks that we can append a Statement to the Block

  Block block(std::make_unique<VariableDeclaration>(Datatype(Type::BOOL),
                                                    std::make_unique<Variable>("foo"),
                                                    std::make_unique<LiteralBool>(true)));

  block.prependStatement(std::make_unique<VariableDeclaration>(Datatype(Type::BOOL),
                                                              std::make_unique<Variable>("boo"),
                                                              std::make_unique<LiteralBool>(true)));

  ASSERT_EQ(block.countChildren(), 2);
  EXPECT_EQ(getNameFromDeclaration(block.getStatements()[0]), "boo");
  EXPECT_EQ(getNameFromDeclaration(block.getStatements()[1]), "foo");
}

TEST(BlockTest, JsonOutputTest) { /* NOLINT */

  std::vector<std::unique_ptr<AbstractStatement>> statements;
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));
  statements.emplace_back(nullptr);
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(false)));
  Block block(std::move(statements));

  nlohmann::json j = {{"type", "Block"},
                      {"statements", {
                          {
                              {"type", "VariableDeclaration"},
                              {"datatype", "bool",},
                              {"target", {
                                  {"type", "Variable"},
                                  {"identifier", "foo"}}},
                              {"value", {
                                  {"type", "LiteralBool"},
                                  {"value", true}}
                              }
                          },
                          {
                              {"type", "VariableDeclaration"},
                              {"datatype", "bool",},
                              {"target", {
                                  {"type", "Variable"},
                                  {"identifier", "boo"}}},
                              {"value", {
                                  {"type", "LiteralBool"},
                                  {"value", false}}
                              }
                          }
                      }
                      }
  };

  EXPECT_EQ(block.toJson(), j);
}


TEST(BlockTest, JsonInputTest) { /* NOLINT */

  std::vector<std::unique_ptr<AbstractStatement>> statements;
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true)));
  statements.emplace_back(nullptr);
  statements.emplace_back(std::make_unique<VariableDeclaration>(
      Datatype(Type::BOOL), std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(false)));
  Block expected(std::move(statements));

  std::string json = R""""({
    "statements": [
      {
        "datatype": "bool",
        "target": {
          "identifier": "foo",
          "type": "Variable"
        },
        "type": "VariableDeclaration",
        "value": {
          "type": "LiteralBool",
          "value": true
        }
      },
      {
        "datatype": "bool",
        "target": {
          "identifier": "boo",
          "type": "Variable"
        },
        "type": "VariableDeclaration",
        "value": {
          "type": "LiteralBool",
          "value": false
        }
      }
    ],
    "type": "Block"
  })"""";

  auto parsed = Parser::parseJson(json);

  ASSERT_TRUE(compareAST(*parsed, expected));
}