#include <heco/ast_parser/Parser.h>
#include "ASTComparison.h"
#include "heco/ast_utilities/Operator.h"
#include "heco/ast_utilities/Datatype.h"
#include "heco/ast/VariableDeclaration.h"
#include "heco/ast/Assignment.h"
#include "heco/ast/BinaryExpression.h"
#include "heco/ast/For.h"
#include "heco/ast/Literal.h"
#include "gtest/gtest.h"

TEST(ForTest, values_ValuesGivenInCtorAreRetrievable)
{ /* NOLINT */
  // For statements are created with an initializer, condition, update and body
  // TODO: This test simply confirms that they are retrievable later
}

TEST(ForTest, SetAndGet)
{ /* NOLINT */
  // TODO: This test simply checks that initializer, condition, update and body can be set and get correctly.
}

TEST(ForTest, CopyCtorCopiesValue)
{ /* NOLINT */
  // TODO: When copying a For, the new object should contain a (deep) copy of the initializer, condition, update and body
}

TEST(ForTest, CopyAssignmentCopiesValue)
{ /* NOLINT */
  // TODO: When copying a For, the new object should contain a copy of the initializer, condition, update and body
}

TEST(ForTest, MoveCtorPreservesValue)
{ /* NOLINT */
  // TODO: When moving a For, the new object should contain the same initializer, condition, update and body
}

TEST(ForTest, MoveAssignmentPreservesValue)
{ /* NOLINT */
  // TODO: When moving a For, the new object should contain the same initializer, condition, update and body
}

TEST(ForTest, countChildrenReportsCorrectNumber)
{ /* NOLINT */
  // TODO: This tests checks that countChildren delivers the correct number
}

TEST(ForTest, node_iterate_children)
{ /* NOLINT */
  // This test checks that we can iterate correctly through the children
  // TODO: Even if some of the elements are null (in which case they should not appear)

  auto initializer = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                           std::make_unique<Variable>("i"),
                                                           std::make_unique<
                                                               LiteralInt>(0));
  auto initializerBlock = std::make_unique<Block>(std::move(initializer));

  auto condition = std::make_unique<BinaryExpression>(std::make_unique<Variable>("i"),
                                                      Operator(LESS),
                                                      std::make_unique<LiteralInt>(100));

  auto updater = std::make_unique<Assignment>(std::make_unique<Variable>("i"),
                                              std::make_unique<BinaryExpression>(std::make_unique<Variable>("i"),
                                                                                 Operator(ADDITION),
                                                                                 std::make_unique<LiteralInt>(1)));
  auto updaterBlock = std::make_unique<Block>(std::move(updater));

  auto bodyStmt = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                        std::make_unique<Variable>("dummy"),
                                                        std::make_unique<
                                                            LiteralInt>(111));
  auto bodyBlock = std::make_unique<Block>(std::move(bodyStmt));

  auto forLoop = std::make_unique<For>(std::move(initializerBlock),
                                       std::move(condition),
                                       std::move(updaterBlock),
                                       std::move(bodyBlock));

  auto it = forLoop->begin();
  EXPECT_EQ(dynamic_cast<Block *>(&(*it)), &forLoop->getInitializer());

  it++;
  EXPECT_EQ(dynamic_cast<BinaryExpression *>(&(*it)), &forLoop->getCondition());

  it++;
  EXPECT_EQ(dynamic_cast<Block *>(&(*it)), &forLoop->getUpdate());

  it++;
  EXPECT_EQ(dynamic_cast<Block *>(&(*it)), &forLoop->getBody());
  it++;
  EXPECT_EQ(it, forLoop->end());
}

TEST(ForTest, JsonOutputTest)
{ /* NOLINT */
  // TODO: Verify JSON output
}

TEST(ForTest, JsonInputTest)
{
  auto initializer = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                           std::make_unique<Variable>("i"),
                                                           std::make_unique<
                                                               LiteralInt>(0));
  auto initializerBlock = std::make_unique<Block>(std::move(initializer));

  auto condition = std::make_unique<BinaryExpression>(std::make_unique<Variable>("i"),
                                                      Operator(LESS),
                                                      std::make_unique<LiteralInt>(100));

  auto updater = std::make_unique<Assignment>(std::make_unique<Variable>("i"),
                                              std::make_unique<BinaryExpression>(std::make_unique<Variable>("i"),
                                                                                 Operator(ADDITION),
                                                                                 std::make_unique<LiteralInt>(1)));
  auto updaterBlock = std::make_unique<Block>(std::move(updater));

  auto bodyStmt = std::make_unique<VariableDeclaration>(Datatype(Type::INT, false),
                                                        std::make_unique<Variable>("dummy"),
                                                        std::make_unique<
                                                            LiteralInt>(111));
  auto bodyBlock = std::make_unique<Block>(std::move(bodyStmt));

  For expected(std::move(initializerBlock), std::move(condition), std::move(updaterBlock), std::move(bodyBlock));

  std::string json = R""""({
    "body": {
      "statements": [
        {
          "datatype": "int",
          "target": {
            "identifier": "dummy",
            "type": "Variable"
          },
          "type": "VariableDeclaration",
          "value": {
            "type": "LiteralInt",
            "value": 111
          }
        }
      ],
      "type": "Block"
    },
    "condition": {
      "left": {
        "identifier": "i",
        "type": "Variable"
      },
      "operator": "<",
      "right": {
        "type": "LiteralInt",
        "value": 100
      },
      "type": "BinaryExpression"
    },
    "initializer": {
      "statements": [
        {
          "datatype": "int",
          "target": {
            "identifier": "i",
            "type": "Variable"
          },
          "type": "VariableDeclaration",
          "value": {
            "type": "LiteralInt",
            "value": 0
          }
        }
      ],
      "type": "Block"
    },
    "type": "For",
    "update": {
      "statements": [
        {
          "target": {
            "identifier": "i",
            "type": "Variable"
          },
          "type": "Assignment",
          "value": {
            "left": {
              "identifier": "i",
              "type": "Variable"
            },
            "operator": "+",
            "right": {
              "type": "LiteralInt",
              "value": 1
            },
            "type": "BinaryExpression"
          }
        }
      ],
      "type": "Block"
    }
  })"""";

  auto parsed = Parser::parseJson(json);

  ASSERT_TRUE(compareAST(*parsed, expected));
}