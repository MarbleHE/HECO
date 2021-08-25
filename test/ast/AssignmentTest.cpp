#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/parser/Parser.h"
#include "gtest/gtest.h"
#include "../ASTComparison.h"


TEST(AssignmentTest, values_ValuesGivenInCtorAreRetrievable) {
  // Assignments are created with a target Variable and value AbstractExpression
  // This test simply confirms that they are retrievable later

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  ASSERT_TRUE(assignment.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(assignment.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(assignment.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(assignment.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(AssignmentTest, target_and_value_SetAndGet) {
  // This test simply checks that target and value can be set and get correctly.

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Set new target and value
  assignment.setTarget(std::make_unique<Variable>("boo"));
  assignment.setValue(std::make_unique<LiteralBool>(false));

  ASSERT_TRUE(assignment.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(assignment.getTarget()).getIdentifier(), "boo");
  ASSERT_TRUE(assignment.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(assignment.getValue()).getValue(),false);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(AssignmentTest, values_CopyCtorCopiesValue) {
  // When copying a Assignment, the new object should contain a (deep) copy of the target and value

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Copy the assignment
  Assignment assignment_copy(assignment);

  ASSERT_TRUE(assignment_copy.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(assignment_copy.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(assignment_copy.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(assignment_copy.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(AssignmentTest, values_CopyAssignmentCopiesValue) {
  // When copying a Assignment, the new object should contain a copy of the value

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Copy the assignment
  Assignment assignment_copy(std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(false));
  assignment_copy = assignment;

  ASSERT_TRUE(assignment_copy.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(assignment_copy.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(assignment_copy.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(assignment_copy.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(AssignmentTest, values_MoveCtorPreservesValue) {
  // When moving a Assignment, the new object should contain the same target and value

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Move the assignment
  Assignment new_assignment(std::move(assignment));

  ASSERT_TRUE(new_assignment.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(new_assignment.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(new_assignment.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(new_assignment.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(AssignmentTest, values_MoveAssignmentPreservesValue) {
  // When moving a Assignment, the new object should contain the same target and value

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Move the assignment
  Assignment new_assignment(std::make_unique<Variable>("new"), std::make_unique<LiteralBool>(false));
  new_assignment = std::move(assignment);

  ASSERT_TRUE(new_assignment.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(new_assignment.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(new_assignment.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(new_assignment.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(AssignmentTest, node_countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));
  auto reported_count = assignment.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for(auto &c: assignment) {
    c = c; // Use c to suppress warning
    ++actual_count;
  }

  EXPECT_EQ(reported_count,actual_count);
}

TEST(AssignmentTest, node_iterate_children) {
  // This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

  Assignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));
  AbstractNode::iterator it = assignment.begin();
  AbstractNode& variable = *it;
  ++it;
  AbstractNode& value = *it;
  ++it;

  // Iterator should now be at end. Dynamic casts will fail if wrong type.
  EXPECT_EQ(it, assignment.end());
  EXPECT_EQ(dynamic_cast<Variable&>(variable).getIdentifier(),"foo");
  EXPECT_EQ(dynamic_cast<LiteralBool&>(value).getValue(),true);
}

TEST(AssignmentTest, JsonOutputTest) { /* NOLINT */
  std::string identifier = "myCustomVar";
  int val = 2;
  auto assignment = Assignment(std::make_unique<Variable>(identifier), std::make_unique<LiteralInt>(val));
  nlohmann::json j = {{"type", "Assignment"},
            {"target", {
                {"type", "Variable"},
                {"identifier", identifier}}},
            {"value", {
                {"type", "LiteralInt"},
                {"value", val}
            }
            }};
  EXPECT_EQ(assignment.toJson(), j);
}

TEST(AssignmentTest, JsonInputTest) { /* NOLINT */
  std::string identifier = "myCustomVar";
  int val = 2;
  auto assignment_expected = Assignment(std::make_unique<Variable>(identifier), std::make_unique<LiteralInt>(val));

  std::string assignment_json = R""""({
        "type": "Assignment",
        "value": {
          "type": "LiteralInt",
          "value": 2
        },
        "target": {
          "identifier": "myCustomVar",
          "type": "Variable"
        }
      })"""";

  auto assignment_parsed = Parser::parseJson(assignment_json);

  ASSERT_TRUE(compareAST(*assignment_parsed, assignment_expected));
}