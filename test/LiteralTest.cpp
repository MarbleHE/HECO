#include "Literal.h"
#include "LiteralBool.h"
#include "LiteralInt.h"
#include "LiteralFloat.h"
#include "LiteralString.h"
#include "gtest/gtest.h"

TEST(LiteralTest, compareLiterals_literalsOfDifferentTypeAreAlwaysUnequal) { /* NOLINT */
  Literal* boolFalse = new LiteralBool(false);
  Literal* int22 = new LiteralInt(22);
  ASSERT_FALSE(*boolFalse == *int22);
}

TEST(LiteralTest, compareLiterals_bool_bool_differentValue) { /* NOLINT */
  Literal* boolTrue = new LiteralBool(true);
  Literal* boolFalse = new LiteralBool(false);
  ASSERT_FALSE(*boolTrue == *boolFalse);
}

TEST(LiteralTest, compareLiterals_bool_bool_sameValue) { /* NOLINT */
  Literal* boolOne = new LiteralBool(true);
  Literal* boolTwo = new LiteralBool(true);
  ASSERT_TRUE(*boolOne == *boolTwo);
}

TEST(LiteralTest, compareLiterals_string_string_differentValue) { /* NOLINT */
  Literal* strAlpha = new LiteralString("alpha");
  Literal* strBeta = new LiteralString("beta");
  ASSERT_FALSE(*strAlpha == *strBeta);
}

TEST(LiteralTest, compareLiterals_string_string_sameValue) { /* NOLINT */
  Literal* strOne = new LiteralString("alpha");
  Literal* strTwo = new LiteralString("alpha");
  ASSERT_TRUE(*strOne == *strTwo);
}

TEST(LiteralTest, compareLiterals_float_float_differentValue) { /* NOLINT */
  Literal* float434_113f = new LiteralFloat(434.113f);
  Literal* float3_3333f = new LiteralFloat(3.3333f);
  ASSERT_FALSE(*float434_113f == *float3_3333f);
}

TEST(LiteralTest, compareLiterals_float_float_sameValue) { /* NOLINT */
  Literal* floatOne = new LiteralFloat(434.113f);
  Literal* floatTwo = new LiteralFloat(434.113f);
  ASSERT_TRUE(*floatOne == *floatTwo);
}

TEST(LiteralTest, compareLiterals_int_int_differentValue) { /* NOLINT */
  Literal* int23324 = new LiteralInt(23324);
  Literal* int555 = new LiteralInt(555);
  ASSERT_FALSE(*int23324 == *int555);
}

TEST(LiteralTest, compareLiterals_int_int_sameValue) { /* NOLINT */
  Literal* intOne = new LiteralInt(87482323);
  Literal* intTwo = new LiteralInt(87482323);
  ASSERT_TRUE(*intOne == *intTwo);
}
