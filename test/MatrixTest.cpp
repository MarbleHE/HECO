#include "Matrix.h"
#include "gtest/gtest.h"

TEST(MatrixTest, createMatrix_expectedInvalidMatrixDimensionsException) {  /* NOLINT */
  EXPECT_THROW(Matrix<int>({{3, 3, 2}, {2, 3}}), std::invalid_argument);
}

TEST(MatrixTest, createMatrix_expectedValidMatrix) {  /* NOLINT */
  Matrix<int> m({{1, 2, 3}, {6, 3, 2}, {52, 11, 95}});

  EXPECT_FALSE(m.isScalar());
//  EXPECT_EQ(m.getDimensions(), std::pair(3, 3));
  EXPECT_EQ(m.getDimensions(), Dimension(3, 3));

  // check values using direct access
  EXPECT_EQ(m.values[0][0], 1);
  EXPECT_EQ(m.values[0][1], 2);
  EXPECT_EQ(m.values[0][2], 3);
  EXPECT_EQ(m.values[1][0], 6);
  EXPECT_EQ(m.values[1][1], 3);
  EXPECT_EQ(m.values[1][2], 2);
  EXPECT_EQ(m.values[2][0], 52);
  EXPECT_EQ(m.values[2][1], 11);
  EXPECT_EQ(m.values[2][2], 95);

  // check values using getter
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(0, 2), 3);
  EXPECT_EQ(m(1, 0), 6);
  EXPECT_EQ(m(1, 1), 3);
  EXPECT_EQ(m(1, 2), 2);
  EXPECT_EQ(m(2, 0), 52);
  EXPECT_EQ(m(2, 1), 11);
  EXPECT_EQ(m(2, 2), 95);
}

TEST(MatrixTest, createScalar_expectedScalarValue) {  /* NOLINT */
  Matrix<int> m(2);

  EXPECT_TRUE(m.isScalar());
//  EXPECT_EQ(m.getDimensions(), std::pair(1, 1));
  EXPECT_EQ(m.getDimensions(), Dimension(1, 1));
  EXPECT_EQ(m.getScalarValue(), 2);
}

TEST(MatrixTest, compareMatrices_expectedEquality) {  /* NOLINT */
  Matrix<int> m1({{1, 2, 1}, {3, 1, 1}});
  Matrix<int> m2({{1, 2, 1}, {3, 1, 1}});
  EXPECT_TRUE(m1==m2);
}

TEST(MatrixTest, compareMatrices_expectedInequalityDueToDifferentValues) {  /* NOLINT */
  Matrix<float> m1({{1.2f, 2.76f, 1.0f}, {3.5f, 1.0f, 1.0f}});
  Matrix<float> m2({{10.0f, 9.0f, 1.0f}, {3.6f, 1.0f, 0.0f}});
  EXPECT_TRUE(m1!=m2);
}

TEST(MatrixTest, compareMatrices_expectedInequalityDueToDifferentDims) {  /* NOLINT */
  Matrix<int> m1({{1, 2, 1}, {3, 1, 1}});
  Matrix<int> m2({{1, 2, 1}, {3, 1, 1}, {4, 3, 1}});
  EXPECT_TRUE(m1!=m2);
}

TEST(MatrixTest, jsonRepresentation_scalar) {  /* NOLINT */
  Matrix<int> m(453492);
  EXPECT_EQ(m.toJson(), json(453492));
}

TEST(MatrixTest, jsonRepresentation_booleanMatrix) {  /* NOLINT */
  Matrix<bool> m({{true, false, false}, {false, true, false}, {false, false, true}});
  auto expected = json::array({{true, false, false}, {false, true, false}, {false, false, true}});
  EXPECT_EQ(m.toJson(), expected);
}

TEST(MatrixTest, accessMatrixElement_invalidIndexExceptionOnMatrix) {  /* NOLINT */
  Matrix<int> m({{1, 2, 1}, {3, 1, 1}});
  // first row
  EXPECT_NO_THROW(m(0, 0));
  EXPECT_NO_THROW(m(0, 1));
  EXPECT_NO_THROW(m(0, 2));
  EXPECT_THROW(m(0, 3), std::invalid_argument);
  // second row
  EXPECT_NO_THROW(m(1, 0));
  EXPECT_NO_THROW(m(1, 1));
  EXPECT_NO_THROW(m(1, 2));
  EXPECT_THROW(m(1, 3), std::invalid_argument);
  // third row
  EXPECT_THROW(m(2, 0), std::invalid_argument);
}

TEST(MatrixTest, accessMatrixElement_invalidIndexExceptionOnScalar) {  /* NOLINT */
  auto value = 2.948811f;
  Matrix<float> scalar(value);
  EXPECT_EQ(scalar(0, 0), value);
  EXPECT_EQ(scalar.getScalarValue(), value);
}
