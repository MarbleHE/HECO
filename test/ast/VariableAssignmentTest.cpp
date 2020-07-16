#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableAssignment.h"
#include "gtest/gtest.h"


TEST(VariableAssignmentTest, values_ValuesGivenInCtorAreRetrievable) {
  // VariableAssignments are created with a target Variable and value AbstractExpression
  // This test simply confirms that they are retrievable later

  VariableAssignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  ASSERT_TRUE(assignment.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(assignment.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(assignment.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(assignment.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableAssignmentTest, target_and_value_SetAndGet) {
  // This test simply checks that target and value can be set and get correctly.

  VariableAssignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Set new target and value
  assignment.setTarget(std::make_unique<Variable>("boo"));
  assignment.setValue(std::make_unique<LiteralBool>(false));

  ASSERT_TRUE(assignment.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(assignment.getTarget()).getIdentifier(), "boo");
  ASSERT_TRUE(assignment.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(assignment.getValue()).getValue(),false);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableAssignmentTest, values_CopyCtorCopiesValue) {
  // When copying a VariableAssignment, the new object should contain a (deep) copy of the target and value

  VariableAssignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Copy the assignment
  VariableAssignment assignment_copy(assignment);

  ASSERT_TRUE(assignment_copy.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(assignment_copy.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(assignment_copy.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(assignment_copy.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableAssignmentTest, values_CopyAssignmentCopiesValue) {
  // When copying a VariableAssignment, the new object should contain a copy of the value

  VariableAssignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Copy the assignment
  VariableAssignment assignment_copy(std::make_unique<Variable>("boo"), std::make_unique<LiteralBool>(false));
  assignment_copy = assignment;

  ASSERT_TRUE(assignment_copy.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(assignment_copy.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(assignment_copy.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(assignment_copy.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableAssignmentTest, values_MoveCtorPreservesValue) {
  // When moving a VariableAssignment, the new object should contain the same target and value

  VariableAssignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Move the assignment
  VariableAssignment new_assignment(std::move(assignment));

  ASSERT_TRUE(new_assignment.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(new_assignment.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(new_assignment.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(new_assignment.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableAssignmentTest, values_MoveAssignmentPreservesValue) {
  // When moving a VariableAssignment, the new object should contain the same target and value

  VariableAssignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));

  // Move the assignment
  VariableAssignment new_assignment(std::make_unique<Variable>("new"), std::make_unique<LiteralBool>(false));
  new_assignment = std::move(assignment);

  ASSERT_TRUE(new_assignment.hasTarget());
  EXPECT_EQ(dynamic_cast<Variable&>(new_assignment.getTarget()).getIdentifier(), "foo");
  ASSERT_TRUE(new_assignment.hasValue());
  EXPECT_EQ(dynamic_cast<LiteralBool&>(new_assignment.getValue()).getValue(),true);
  // If the value does not have type LiteralBool, the dynamic cast should fail
}

TEST(VariableAssignmentTest, node_countChildrenReportsCorrectNumber) {
  // This tests checks that countChildren delivers the correct number

  VariableAssignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));
  auto reported_count = assignment.countChildren();

  // Iterate through all the children using the iterators
  // (indirectly via range-based for loop for conciseness)
  size_t actual_count = 0;
  for(auto &c: assignment) {
    ++actual_count;
  }

  EXPECT_EQ(reported_count,actual_count);
}

TEST(VariableAssignmentTest, node_iterate_children) {
  // This test checks that we can iterate correctly through the children
  // Even if some of the elements are null (in which case they should not appear)

  VariableAssignment assignment(std::make_unique<Variable>("foo"), std::make_unique<LiteralBool>(true));
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

TEST(VariableAssignmentTest, JsonOutputTest) { /* NOLINT */
  std::string identifier = "myCustomVar";
  int val = 2;
  auto assignment = VariableAssignment(std::make_unique<Variable>(identifier), std::make_unique<LiteralInt>(val));
  nlohmann::json j = {{"type", "VariableAssignment"},
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