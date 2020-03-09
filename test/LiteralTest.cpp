#include "AbstractLiteral.h"
#include "LiteralBool.h"
#include "LiteralInt.h"
#include "LiteralFloat.h"
#include "LiteralString.h"
#include "gtest/gtest.h"

TEST(LiteralTest, compareLiterals_literalsOfDifferentTypeAreAlwaysUnequal) { /* NOLINT */
  AbstractLiteral *boolFalse = new LiteralBool(false);
  AbstractLiteral *int22 = new LiteralInt(22);
  ASSERT_FALSE(*boolFalse==*int22);
}

TEST(LiteralTest, compareLiterals_bool_bool_differentValue) { /* NOLINT */
  AbstractLiteral *boolTrue = new LiteralBool(true);
  AbstractLiteral *boolFalse = new LiteralBool(false);
  ASSERT_FALSE(*boolTrue==*boolFalse);
}

TEST(LiteralTest, compareLiterals_bool_bool_sameValue) { /* NOLINT */
  AbstractLiteral *boolOne = new LiteralBool(true);
  AbstractLiteral *boolTwo = new LiteralBool(true);
  ASSERT_TRUE(*boolOne==*boolTwo);
}

TEST(LiteralTest, compareLiterals_string_string_differentValue) { /* NOLINT */
  AbstractLiteral *strAlpha = new LiteralString("alpha");
  AbstractLiteral *strBeta = new LiteralString("beta");
  ASSERT_FALSE(*strAlpha==*strBeta);
}

TEST(LiteralTest, compareLiterals_string_string_sameValue) { /* NOLINT */
  AbstractLiteral *strOne = new LiteralString("alpha");
  AbstractLiteral *strTwo = new LiteralString("alpha");
  ASSERT_TRUE(*strOne==*strTwo);
}

TEST(LiteralTest, compareLiterals_float_float_differentValue) { /* NOLINT */
  AbstractLiteral *float434_113f = new LiteralFloat(434.113f);
  AbstractLiteral *float3_3333f = new LiteralFloat(3.3333f);
  ASSERT_FALSE(*float434_113f==*float3_3333f);
}

TEST(LiteralTest, compareLiterals_float_float_sameValue) { /* NOLINT */
  AbstractLiteral *floatOne = new LiteralFloat(434.113f);
  AbstractLiteral *floatTwo = new LiteralFloat(434.113f);
  ASSERT_TRUE(*floatOne==*floatTwo);
}

TEST(LiteralTest, compareLiterals_int_int_differentValue) { /* NOLINT */
  AbstractLiteral *int23324 = new LiteralInt(23324);
  AbstractLiteral *int555 = new LiteralInt(555);
  ASSERT_FALSE(*int23324==*int555);
}

TEST(LiteralTest, compareLiterals_int_int_sameValue) { /* NOLINT */
  AbstractLiteral *intOne = new LiteralInt(87482323);
  AbstractLiteral *intTwo = new LiteralInt(87482323);
  ASSERT_TRUE(*intOne==*intTwo);
}

TEST(LiteralTest, literalBoolMatrix) { /* NOLINT */
  auto pMatrix = new Matrix<bool>({{true, false, false}, {false, false, true}, {true, true, true}});
  auto litBool = new LiteralBool(pMatrix);

  // check that this is not a scalar: getValue() must not work
  EXPECT_THROW(litBool->getValue(), std::logic_error);
  // check that the matrix was stored properly
  EXPECT_EQ(litBool->getMatrix(), pMatrix);

  auto matrix = dynamic_cast<Matrix<bool> *>(litBool->getMatrix());
  auto matrixRef = *matrix;
  // compare element at (0,0)
  EXPECT_EQ(matrix->getElement(0, 0), true);
  EXPECT_EQ(matrixRef(0, 0), true);
  // compare element at (1,0)
  EXPECT_EQ(matrix->getElement(1, 0), false);
  EXPECT_EQ(matrixRef(1, 0), false);
  // compare element at (2,1)
  EXPECT_EQ(matrix->getElement(2, 1), true);
  EXPECT_EQ(matrixRef(2, 1), true);

  matrixRef(0, 0) = false;
  EXPECT_EQ(matrixRef(0, 0), false);
}
