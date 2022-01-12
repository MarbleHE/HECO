#include <abc/ast_parser/Parser.h>
#include "ASTComparison.h"
#include "abc/ast/Literal.h"
#include "abc/ast/Variable.h"
#include "abc/ast/Return.h"
#include "gtest/gtest.h"

TEST(ReturnTest, values_ValuesGivenInCtorAreRetrievable) {
  // Returns are created with a target Variable and value AbstractExpression and type
  // This test simply confirms that they are retrievable later

  Return r(std::make_unique<LiteralBool>(true));

  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(r.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(ReturnTest, target_and_value_SetAndGet) {
  // This test simply checks that target and value can be set and get correctly.

  Return r(std::make_unique<LiteralBool>(true));

  // Set new target and value
  r.setValue(std::make_unique<LiteralBool>(false));

  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(r.getValue()).getValue(), false);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(ReturnTest, values_CopyCtorCopiesValue) {
  // When copying a Return, the new object should contain a (deep) copy of the target and value

  Return r(std::make_unique<LiteralBool>(true));

  // Copy the r
  Return r_copy(r);

  ASSERT_TRUE(r.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(r.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(ReturnTest, values_CopyAssignmentCopiesValue) {
  // When copying a Return, the new object should contain a copy of the value

  Return r(std::make_unique<LiteralBool>(true));

  // Copy the r
  Return r_copy(std::make_unique<LiteralBool>(false));
  r_copy = r;


  ASSERT_TRUE(r_copy.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(r_copy.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(ReturnTest, values_MoveCtorPreservesValue) {
  // When moving a Return, the new object should contain the same target and value

  Return r(std::make_unique<LiteralBool>(true));

  // Move the r
  Return new_r(std::move(r));

  ASSERT_TRUE(new_r.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(new_r.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(ReturnTest, values_MoveAssignmentPreservesValue) {
  // When moving a Return, the new object should contain the same target and value

  Return r(std::make_unique<LiteralBool>(true));

  // Move the r
  Return new_r(std::make_unique<LiteralBool>(false));
  new_r = std::move(r);

  ASSERT_TRUE(new_r.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(new_r.getValue()).getValue(), true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(ReturnTest, node_countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  Return r(std::make_unique<LiteralBool>(true));
  auto reported_count = r.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for (auto &c: r) {
    c = c; // Use c to suppress warning
    ++actual_count;
  }

  EXPECT_EQ(reported_count, actual_count);
  EXPECT_EQ(reported_count, 1);
}

TEST(ReturnTest, node_iterate_children) {
  // This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

  Return r(std::make_unique<LiteralBool>(true));
  AbstractNode::iterator it = r.begin();
  AbstractNode &value = *it;
  ++it;

  // Iterator should now be at end. Dynamic casts will fail if wrong type.
  EXPECT_EQ(it, r.end());
  EXPECT_EQ(dynamic_cast<LiteralBool &>(value).getValue(), true);
}

TEST(ReturnTest, JsonOutputTest) { /* NOLINT */
  Return r(std::make_unique<LiteralInt>(3));
  nlohmann::json j ={
      {"type", "Return"},
      {"value", {
              {"type", "LiteralInt"},
              {"value", 3}}
      }
  };
  EXPECT_EQ(r.toJson(), j);
}


TEST(ReturnTest, JsonInputTest) { /* NOLINT */
  Return expected(std::make_unique<LiteralInt>(3));

  std::string json = R""""({
    "type": "Return",
    "value": {
      "type": "LiteralInt",
      "value": 3
    }
  })"""";

  auto parsed = Parser::parseJson(json);

  ASSERT_TRUE(compareAST(*parsed, expected));
}