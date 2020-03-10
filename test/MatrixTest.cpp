#include <Variable.h>
#include <type_traits>
#include "Matrix.h"
#include "gtest/gtest.h"
#include "LiteralFloat.h"
#include "LiteralBool.h"
#include "LiteralInt.h"
#include "LiteralString.h"

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

TEST(MatrixTest, modifyMatrix_modifyValues) {  /* NOLINT */
  Matrix<int> m({{5, 2, 1}, {3, 1, 1}, {1, 1, 0}});

  // modify value at (0,2)
  EXPECT_EQ(m(0, 2), 1);
  m(0, 2) = 9;
  EXPECT_EQ(m(0, 2), 9);

  // modify value at (2,2)
  EXPECT_EQ(m(2, 2), 0);
  m(2, 2) = 4224;
  EXPECT_EQ(m(2, 2), 4224);
}

TEST(MatrixTest, modifyMatrixAsPointer_modifyValuesInt) {  /* NOLINT */
  auto *mx = new Matrix<int>({{5, 2, 1}, {3, 1, 1}, {1, 1, 0}});
  Matrix<int> &m = *mx;

  // modify value at (0,2)
  EXPECT_EQ(m(0, 2), 1);
  m(0, 2) = 9;
  EXPECT_EQ(m(0, 2), 9);

  // modify value at (2,2)
  EXPECT_EQ(m(2, 2), 0);
  m(2, 2) = 4224;
  EXPECT_EQ(m(2, 2), 4224);
}

TEST(MatrixTest, modifyMatrixAsPointer_modifyValuesFloat) {  /* NOLINT */
  auto *mx = new Matrix<float>({{5.23f, 2.0f, 1.11f}, {3.0f, 1.221f, 1.9f}, {1.0f, 1.0f, 0.0f}});
  Matrix<float> &m = *mx;

  // modify value at (0,2)
  EXPECT_EQ(m(0, 2), 1.11f);
  m(0, 2) = 9.23f;
  EXPECT_EQ(m(0, 2), 9.23f);

  // modify value at (2,2)
  EXPECT_EQ(m(2, 2), 0.0f);
  m(2, 2) = 4224.333f;
  EXPECT_EQ(m(2, 2), 4224.333f);
}

TEST(MatrixTest, modifyMatrixAsPointer_modifyValuesBoolean) {  /* NOLINT */
  auto *mx = new Matrix<bool>({{true, true, false}, {false, false, false}, {true, false, true}});
  Matrix<bool> &m = *mx;

  // modify value at (0,2)
  EXPECT_EQ(m(0, 0), true);
  m(0, 0) = false;
  EXPECT_EQ(m(0, 0), false);

  // modify value at (2,2)
  EXPECT_EQ(m(2, 2), true);
  m(2, 2) = false;
  EXPECT_EQ(m(2, 2), false);
}

TEST(MatrixTest, modifyMatrix_modifyWholeMatrixUsingAssignmentOp) {  /* NOLINT */
  Matrix<int> m({{5, 2, 1}, {3, 1, 1}, {1, 1, 0}});

  // check values using getter
  EXPECT_EQ(m(0, 0), 5);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(0, 2), 1);
  EXPECT_EQ(m(1, 0), 3);
  EXPECT_EQ(m(1, 1), 1);
  EXPECT_EQ(m(1, 2), 1);
  EXPECT_EQ(m(2, 0), 1);
  EXPECT_EQ(m(2, 1), 1);
  EXPECT_EQ(m(2, 2), 0);

  // overwrite whole matrix
  m = std::vector<std::vector<int>>({{0, 2, 1}, {1, 1, 1}, {90, 1, 34}});

  // check values using getter
  EXPECT_EQ(m(0, 0), 0);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(0, 2), 1);
  EXPECT_EQ(m(1, 0), 1);
  EXPECT_EQ(m(1, 1), 1);
  EXPECT_EQ(m(1, 2), 1);
  EXPECT_EQ(m(2, 0), 90);
  EXPECT_EQ(m(2, 1), 1);
  EXPECT_EQ(m(2, 2), 34);
}

TEST(MatrixTest, modifyMatrix_modifyWholeMatrixUsingDedicatedFunction) {  /* NOLINT */
  Matrix<int> m({{5, 2, 1}, {3, 1, 1}, {1, 1, 0}});

  // check values using getter
  EXPECT_EQ(m(0, 0), 5);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(0, 2), 1);
  EXPECT_EQ(m(1, 0), 3);
  EXPECT_EQ(m(1, 1), 1);
  EXPECT_EQ(m(1, 2), 1);
  EXPECT_EQ(m(2, 0), 1);
  EXPECT_EQ(m(2, 1), 1);
  EXPECT_EQ(m(2, 2), 0);

  // overwrite whole matrix
  m.setValues({{0, 2, 1}, {1, 1, 1}, {90, 1, 34}});

  // check values using getter
  EXPECT_EQ(m(0, 0), 0);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(0, 2), 1);
  EXPECT_EQ(m(1, 0), 1);
  EXPECT_EQ(m(1, 1), 1);
  EXPECT_EQ(m(1, 2), 1);
  EXPECT_EQ(m(2, 0), 90);
  EXPECT_EQ(m(2, 1), 1);
  EXPECT_EQ(m(2, 2), 34);
}

TEST(MatrixTest, toStringTestMatrix) {  /* NOLINT */
  Matrix<int> m({{5, 2, 1}, {3, 1, 1}, {1, 1, 0}});
  EXPECT_EQ(m.toString(), std::string("[5 2 1; 3 1 1; 1 1 0]"));
}

TEST(MatrixTest, toStringTestScalar) {  /* NOLINT */
  Matrix<int> m(52'147);
  EXPECT_EQ(m.toString(), std::string("52147"));
}

TEST(MatrixTest, cloneMatrix) {  /* NOLINT */
  Matrix<int> m({{1, 0}, {3, 3}});
  Matrix<int> clonedM = m;

  // check that the clone has the same dimension and values
  EXPECT_EQ(m.getDimensions(), clonedM.getDimensions());
  EXPECT_EQ(clonedM(0, 0), 1);
  EXPECT_EQ(clonedM(0, 1), 0);
  EXPECT_EQ(clonedM(1, 0), 3);
  EXPECT_EQ(clonedM(1, 1), 3);

  // check that it's a real clone, i.e., changing the original matrix does not change the clone
  m(0, 0) = 425;
  EXPECT_EQ(m(0, 0), 425);
  EXPECT_EQ(clonedM(0, 0), 1);
  m(1, 1) = 32;
  EXPECT_EQ(m(1, 1), 32);
  EXPECT_EQ(clonedM(1, 1), 3);
}

TEST(MatrixTest, transposeMatrixFromColumnVecToRowVecInPlace) {  /* NOLINT */
  // elements in (0,0), (1,0), (2,0) -> all elements in one column vector
  Matrix<int> m({{1}, {4}, {9}});
  EXPECT_TRUE(m.getDimensions().equals(3, 1));
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(1, 0), 4);
  EXPECT_EQ(m(2, 0), 9);
  m.transpose(true);
  // elements in (0,0), (0,1), (0,2) -> all elements in one row vector
  EXPECT_TRUE(m.getDimensions().equals(1, 3));
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 4);
  EXPECT_EQ(m(0, 2), 9);
}

TEST(MatrixTest, transposeMatrixFromColumnVecToRowVecCopy) {  /* NOLINT */
  // elements in (0,0), (1,0), (2,0) -> all elements in one column vector
  Matrix<int> m({{1}, {4}, {9}});
  auto transposedM = *m.transpose(false);
  // elements in (0,0), (0,1), (0,2) -> all elements in one row vector
  EXPECT_TRUE(transposedM.getDimensions().equals(1, 3));
  EXPECT_EQ(transposedM(0, 0), 1);
  EXPECT_EQ(transposedM(0, 1), 4);
  EXPECT_EQ(transposedM(0, 2), 9);
  // check that the original matrix did not change
  EXPECT_TRUE(m.getDimensions().equals(3, 1));
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(1, 0), 4);
  EXPECT_EQ(m(2, 0), 9);
}

TEST(MatrixTest, transposeMatrixFromRowVecToColumnVecInPlace) {  /* NOLINT */
  // elements in (0,0), (0,1), (0,2) -> all elements in one row vector
  Matrix<int> m({{1, 4, 9}});
  EXPECT_TRUE(m.getDimensions().equals(1, 3));
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 4);
  EXPECT_EQ(m(0, 2), 9);
  m.transpose(true);
  // elements in (0,0), (1,0), (2,0) -> all elements in one column vector
  EXPECT_TRUE(m.getDimensions().equals(3, 1));
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(1, 0), 4);
  EXPECT_EQ(m(2, 0), 9);
}

TEST(MatrixTest, transposeMatrixFromRowVecToColumnVecCopy) {  /* NOLINT */
  // elements in (0,0), (0,1), (0,2) -> all elements in one row vector
  Matrix<int> m({{1, 4, 9}});
  auto transposedM = *m.transpose(false);
  // elements in (0,0), (1,0), (2,0) -> all elements in one column vector
  EXPECT_TRUE(transposedM.getDimensions().equals(3, 1));
  EXPECT_EQ(transposedM(0, 0), 1);
  EXPECT_EQ(transposedM(1, 0), 4);
  EXPECT_EQ(transposedM(2, 0), 9);
  // check that the original matrix did not change
  EXPECT_TRUE(m.getDimensions().equals(1, 3));
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 4);
  EXPECT_EQ(m(0, 2), 9);
}

TEST(MatrixTest, rotateRowVectorInPlace) {  /* NOLINT */
  Matrix<int> m({{1, 2, 3, 4, 5}});
  EXPECT_TRUE(m.getDimensions().equals(1, 5));
  m.rotate(-2, true);
  EXPECT_TRUE(m.getDimensions().equals(1, 5));
  // [1 2 3 4 5] --rotate(-2)--> [3 4 5 1 2]
  EXPECT_EQ(m(0, 0), 3);
  EXPECT_EQ(m(0, 1), 4);
  EXPECT_EQ(m(0, 2), 5);
  EXPECT_EQ(m(0, 3), 1);
  EXPECT_EQ(m(0, 4), 2);
}

TEST(MatrixTest, rotateRowVectorCopy) {  /* NOLINT */
  Matrix<int> m({{1, 2, 3, 4, 5}});
  EXPECT_TRUE(m.getDimensions().equals(1, 5));

  // rotate and check that rotation has expected effect
  auto rotatedM = *m.rotate(-2, false);
  EXPECT_TRUE(rotatedM.getDimensions().equals(1, 5));
  // [1 2 3 4 5] --rotate(-2)--> [3 4 5 1 2]
  EXPECT_EQ(rotatedM(0, 0), 3);
  EXPECT_EQ(rotatedM(0, 1), 4);
  EXPECT_EQ(rotatedM(0, 2), 5);
  EXPECT_EQ(rotatedM(0, 3), 1);
  EXPECT_EQ(rotatedM(0, 4), 2);

  // check that the original matrix did not change as inPlace=false was passed to rotate
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(0, 2), 3);
  EXPECT_EQ(m(0, 3), 4);
  EXPECT_EQ(m(0, 4), 5);
}

TEST(MatrixTest, rotateColumnVectorInPlace) {  /* NOLINT */
  Matrix<int> m({{9}, {4}, {3}, {8}, {7}});
  EXPECT_TRUE(m.getDimensions().equals(5, 1));
  m.rotate(2, true);
  EXPECT_TRUE(m.getDimensions().equals(5, 1));
  // [9; 4; 3; 8; 7] --rotate(2)--> [8; 7; 9; 4; 3]
  EXPECT_EQ(m(0, 0), 8);
  EXPECT_EQ(m(1, 0), 7);
  EXPECT_EQ(m(2, 0), 9);
  EXPECT_EQ(m(3, 0), 4);
  EXPECT_EQ(m(4, 0), 3);
}

TEST(MatrixTest, rotateColumnVectorCopy) {  /* NOLINT */
  Matrix<int> m({{9}, {4}, {3}, {8}, {7}});
  EXPECT_TRUE(m.getDimensions().equals(5, 1));
  auto rotatedM = *m.rotate(2, false);
  // dimension should not change
  EXPECT_TRUE(m.getDimensions().equals(5, 1));
  // [9; 4; 3; 8; 7] --rotate(2)--> [8; 7; 9; 4; 3]
  EXPECT_EQ(rotatedM(0, 0), 8);
  EXPECT_EQ(rotatedM(1, 0), 7);
  EXPECT_EQ(rotatedM(2, 0), 9);
  EXPECT_EQ(rotatedM(3, 0), 4);
  EXPECT_EQ(rotatedM(4, 0), 3);
  // check that the original matrix did not change
  EXPECT_EQ(m(0, 0), 9);
  EXPECT_EQ(m(1, 0), 4);
  EXPECT_EQ(m(2, 0), 3);
  EXPECT_EQ(m(3, 0), 8);
  EXPECT_EQ(m(4, 0), 7);
}

TEST(MatrixTest, rotateMatrix_expectedException) {  /* NOLINT */
  Matrix<int> m({{9, 4, 2}, {3, 2, 1}, {1, 1, 0}});
  EXPECT_THROW(m.rotate(3, true), std::invalid_argument);
}

TEST(MatrixTest, applyOperatorComponentwise) {  /* NOLINT */
  Matrix<int> A({{3, 2, 1}, {4, 4, 1}, {9, 4, 3}});
  Matrix<int> B({{7, 8, 9}, {1, 1, 4}, {6, 11, 3}});

  auto addition = [](int a, int b) { return a + b; };
  auto result = applyComponentwise<int>(&A, &B, addition);
  auto expectedMatrix = Matrix<int>({{10, 10, 10}, {5, 5, 5}, {15, 15, 6}});
  EXPECT_EQ(*result, expectedMatrix);

  auto subtraction = [](int a, int b) { return a - b; };
  result = applyComponentwise<int>(&A, &B, subtraction);
  expectedMatrix = Matrix<int>({{-4, -6, -8}, {3, 3, -3}, {3, -7, 0}});
  EXPECT_EQ(*result, expectedMatrix);
}

TEST(MatrixTest, applyScalarProduct) {  /* NOLINT */
  Matrix<int> A({{3, 2, 1}, {4, 4, 1}, {9, 4, 3}});
  Matrix<int> scalar(1);
  auto addition = [](int a, int b) { return a + b; };

  // scalar product where lhs operand is matrix and rhs operand is scalar
  auto result = applyComponentwise<int>(&A, &scalar, addition);
  auto expectedMatrix = Matrix<int>({{4, 3, 2}, {5, 5, 2}, {10, 5, 4}});
  EXPECT_EQ(*result, expectedMatrix);
  // scalar product where lhs operand is scalar and rhs operand is matrix
  result = applyComponentwise<int>(&scalar, &A, addition);
  EXPECT_EQ(*result, expectedMatrix);
}

TEST(MatrixTest, applyMatrixMultiplication) {  /* NOLINT */
  // 4x2 matrix
  Matrix<int> A({{2, 3}, {4, 5}, {8, 6}, {1, 5}});
  // 2x3 matrix
  Matrix<int> B({{0, 1, 1}, {4, 5, 7}});

  auto result = applyMatrixMultiplication(&A, &B);
  // 4x3 matrix
  auto expectedMatrix = Matrix<int>({{12, 17, 23}, {20, 29, 39}, {24, 38, 50}, {20, 26, 36}});
  EXPECT_EQ(*result, expectedMatrix);
}

class MatrixOperationFixture : public ::testing::Test {
 protected:
  AbstractLiteral *intScalar;
  AbstractLiteral *intMatrix1;
  AbstractLiteral *intMatrix2;
  AbstractLiteral *boolScalar;
  AbstractLiteral *boolMatrix1;
  AbstractLiteral *boolMatrix2;
  AbstractLiteral *floatScalar;
  AbstractLiteral *floatMatrix1;
  AbstractLiteral *floatMatrix2;
  AbstractLiteral *stringScalar;
  AbstractLiteral *stringMatrix1;
  AbstractLiteral *stringMatrix2;

  // TODO(pjattke): Write some more tests.

 public:
  MatrixOperationFixture() {
    intScalar = new LiteralInt(42);
    intMatrix1 = new LiteralInt(new Matrix<int>(
        {{1, 2, 52}, {3, 4, 1}}));
//    intMatrix2 = new LiteralInt(new Matrix<int>({{}}));
//    boolScalar = new LiteralBool(false);
    boolMatrix1 = new LiteralBool(new Matrix<bool>({{true, false, false}, {false, true, false}, {false, false, true}}));
//    boolMatrix2 = new LiteralBool(new Matrix<bool>({{}}));
//    floatScalar = new LiteralFloat(83.11139f);
//    floatMatrix1 = new LiteralFloat(new Matrix<float>({{}}));
//    floatMatrix2 = new LiteralFloat(new Matrix<float>({{}}));
//    stringScalar = new LiteralString("Fully Homomorphic Encryption Acceleration Engine");
//    stringMatrix1 = new LiteralString(new Matrix<std::string>({{}}));
//    stringMatrix2 = new LiteralString(new Matrix<std::string>({{}}));
  }
};

TEST_F(MatrixOperationFixture, matrix_scalar_add) {
  auto result = Operator(ArithmeticOp::multiplication).applyOperator(intScalar, intMatrix1)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result), *new Matrix<int>({{42, 84, 2'184}, {126, 168, 42}}));
}

TEST_F(MatrixOperationFixture, matrix_negate) {
  auto result = Operator(UnaryOp::negation).applyOperator(boolMatrix1)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<bool> *>(result),
            *new Matrix<bool>({{false, true, true}, {true, false, true}, {true, true, false}}));
}
