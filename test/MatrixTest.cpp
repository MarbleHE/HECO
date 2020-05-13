#include "gtest/gtest.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/Matrix.h"
#include "ast_opt/ast/LiteralFloat.h"
#include "ast_opt/ast/LiteralBool.h"
#include "ast_opt/ast/LiteralInt.h"
#include "ast_opt/ast/LiteralString.h"
#include "ast_opt/ast/ArithmeticExpr.h"
#include "ast_opt/ast/Rotate.h"

class MatrixTestFixture : public ::testing::Test {

};

TEST_F(MatrixTestFixture, createMatrix_expectedInvalidMatrixDimensionsException) {  /* NOLINT */
  EXPECT_THROW(Matrix<int>({{3, 3, 2}, {2, 3}}), std::invalid_argument);
}

TEST_F(MatrixTestFixture, createMatrix_expectedValidMatrix) {  /* NOLINT */
  Matrix<int> m({{1, 2, 3}, {6, 3, 2}, {52, 11, 95}});

  EXPECT_FALSE(m.isScalar());
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

TEST_F(MatrixTestFixture, setMatrixValues_expectedCorrectDimensions) {  /* NOLINT */
  Matrix<int> m({{1, 2, 3}, {6, 3, 2}, {52, 11, 95}});
  EXPECT_EQ(m.getDimensions(), Dimension(3, 3));

  m.setValues({{1, 2}, {2, 3}, {2, 1}});
  EXPECT_EQ(m.getDimensions(), Dimension(3, 2));
}

TEST_F(MatrixTestFixture, setInvalidMatrixValues_expectedException) {  /* NOLINT */
  Matrix<int> m({{1, 2, 3}, {6, 3, 2}, {52, 11, 9}});
  EXPECT_THROW(m.setValues({{1, 2}, {2}, {2, 1}, {3, 1}}), std::invalid_argument);
}

TEST_F(MatrixTestFixture, createInvalidMatrix_expectedException) {  /* NOLINT */
  EXPECT_THROW(Matrix<int> m({{1, 2, 3}, {6, 3, 2}, {52, 11}}), std::invalid_argument);
}

TEST_F(MatrixTestFixture, createScalar_expectedScalarValue) {  /* NOLINT */
  Matrix<int> m(2);

  EXPECT_TRUE(m.isScalar());
  EXPECT_EQ(m.getDimensions(), Dimension(1, 1));
  EXPECT_EQ(m.getScalarValue(), 2);
}

TEST_F(MatrixTestFixture, compareMatrices_expectedEquality) {  /* NOLINT */
  Matrix<int> m1({{1, 2, 1}, {3, 1, 1}});
  Matrix<int> m2({{1, 2, 1}, {3, 1, 1}});
  EXPECT_TRUE(m1==m2);
}

TEST_F(MatrixTestFixture, compareMatrices_expectedInequalityDueToDifferentValues) {  /* NOLINT */
  Matrix<float> m1({{1.2f, 2.76f, 1.0f}, {3.5f, 1.0f, 1.0f}});
  Matrix<float> m2({{10.0f, 9.0f, 1.0f}, {3.6f, 1.0f, 0.0f}});
  EXPECT_TRUE(m1!=m2);
}

TEST_F(MatrixTestFixture, compareMatrices_expectedInequalityDueToDifferentDims) {  /* NOLINT */
  Matrix<int> m1({{1, 2, 1}, {3, 1, 1}});
  Matrix<int> m2({{1, 2, 1}, {3, 1, 1}, {4, 3, 1}});
  EXPECT_TRUE(m1!=m2);
}

TEST_F(MatrixTestFixture, jsonRepresentation_scalar) {  /* NOLINT */
  Matrix<int> m(453492);
  EXPECT_EQ(m.toJson(), json(453492));
}

TEST_F(MatrixTestFixture, jsonRepresentation_booleanMatrix) {  /* NOLINT */
  Matrix<bool> m({{true, false, false}, {false, true, false}, {false, false, true}});
  auto expected = json::array({{true, false, false}, {false, true, false}, {false, false, true}});
  EXPECT_EQ(m.toJson(), expected);
}

//TODO: This test is outdated. We now allow writing to any index,
// but because m() is read/write, this also makes arbitrary reads possible
TEST_F(MatrixTestFixture, accessMatrixElement_invalidIndexExceptionOnMatrix) {  /* NOLINT */
  Matrix<int> m({{1, 2, 1}, {3, 1, 1}});
  // first row
  EXPECT_NO_THROW(m(0, 0));
  EXPECT_NO_THROW(m(0, 1));
  EXPECT_NO_THROW(m(0, 2));
  //EXPECT_THROW(m(0, 3), std::invalid_argument);
  // second row
  EXPECT_NO_THROW(m(1, 0));
  EXPECT_NO_THROW(m(1, 1));
  EXPECT_NO_THROW(m(1, 2));
  //EXPECT_THROW(m(1, 3), std::invalid_argument);
  // third row
  //EXPECT_THROW(m(2, 0), std::invalid_argument);
}

TEST_F(MatrixTestFixture, accessMatrixElement_invalidIndexExceptionOnScalar) {  /* NOLINT */
  auto value = 2.948811f;
  Matrix<float> scalar(value);
  EXPECT_EQ(scalar(0, 0), value);
  EXPECT_EQ(scalar.getScalarValue(), value);
}

TEST_F(MatrixTestFixture, modifyMatrix_modifyValues) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, modifyMatrixAsPointer_modifyValuesInt) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, modifyMatrixAsPointer_modifyValuesFloat) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, modifyMatrixAsPointer_modifyValuesBoolean) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, modifyMatrix_modifyWholeMatrixUsingAssignmentOp) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, modifyMatrix_modifyWholeMatrixUsingDedicatedFunction) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, toStringTestMatrix) {  /* NOLINT */
  Matrix<int> m({{5, 2, 1}, {3, 1, 1}, {1, 1, 0}});
  EXPECT_EQ(m.toString(), std::string("[5 2 1; 3 1 1; 1 1 0]"));
}

TEST_F(MatrixTestFixture, toStringTestScalar) {  /* NOLINT */
  Matrix<int> m(52'147);
  EXPECT_EQ(m.toString(), std::string("52147"));
}

TEST_F(MatrixTestFixture, cloneMatrix) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, transposeMatrixFromColumnVecToRowVecInPlace) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, transposeMatrixFromColumnVecToRowVecCopy) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, transposeMatrixFromRowVecToColumnVecInPlace) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, transposeMatrixFromRowVecToColumnVecCopy) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, rotateRowVectorInPlace) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, rotateRowVectorCopy) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, rotateColumnVectorInPlace) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, rotateColumnVectorCopy) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, rotateMatrix_expectedException) {  /* NOLINT */
  Matrix<int> m({{9, 4, 2}, {3, 2, 1}, {1, 1, 0}});
  EXPECT_THROW(m.rotate(3, true), std::invalid_argument);
}

TEST_F(MatrixTestFixture, rotateMatrixIncludingArithmeticExpr_expectedNoDimensionCheck) {  /* NOLINT */
  EXPECT_NO_THROW(new Rotate(
      new ArithmeticExpr(new Variable("a"), ArithmeticOp::ADDITION, new Variable("b")),
      new LiteralInt(5)));
}

TEST_F(MatrixTestFixture, rotateMatrixIncludingAbstractExpr_expectedNoDimensionCheck) {  /* NOLINT */
  EXPECT_NO_THROW(new Rotate(
      new LiteralInt(
          new Matrix<AbstractExpr *>(
              {{new ArithmeticExpr(new Variable("a"), ArithmeticOp::ADDITION, new Variable("b"))}})),
      new LiteralInt(5)));
}

TEST_F(MatrixTestFixture, applyOperatorComponentwise) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, applyScalarProduct) {  /* NOLINT */
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

TEST_F(MatrixTestFixture, applyMatrixMultiplication) {  /* NOLINT */
  // 4x2 matrix
  Matrix<int> A({{2, 3}, {4, 5}, {8, 6}, {1, 5}});
  // 2x3 matrix
  Matrix<int> B({{0, 1, 1}, {4, 5, 7}});

  auto result = applyMatrixMultiplication(&A, &B);
  // 4x3 matrix
  auto expectedMatrix = Matrix<int>({{12, 17, 23}, {20, 29, 39}, {24, 38, 50}, {20, 26, 36}});
  EXPECT_EQ(*result, expectedMatrix);
}

TEST_F(MatrixTestFixture, detectMalformedNestedMatrix) {  /* NOLINT */
  // Test to check whether detection of matrix elements with dimension != (1,1) works.
  // The following would create a matrix like:
  //  [true false [x y]]
  auto nestedExpr = new LiteralBool(new Matrix<AbstractExpr *>({{new Variable("x"), new Variable("y")}}));
  EXPECT_THROW(Matrix<AbstractExpr *> abstractExprM({{new LiteralBool(true), new LiteralBool(false), nestedExpr}}),
               std::logic_error);
}

TEST_F(MatrixTestFixture, detectMalformedDimension_AbstractExprMatrix) {  /* NOLINT */
  EXPECT_THROW(Matrix<AbstractExpr *> abstractExprM(
                   {
                     { new LiteralBool(true), new LiteralBool(false), new LiteralBool(true) },
                     { new LiteralBool(false), new LiteralBool(true) }
                   }),
               std::logic_error);
}

TEST_F(MatrixTestFixture, detectMalformedDimension_intMatrix) {  /* NOLINT */
  EXPECT_THROW(Matrix<int> intM(
                   {
                     { 34, 33, 1 },
                     { 34, 322, 455, 44 }
                   }),
               std::logic_error);
}

TEST_F(MatrixTestFixture, testRowRetrieval) {  /* NOLINT */
  Matrix<int> A({{3, 2, 1}, {4, 0, 1}, {9, 7, 3}});
  EXPECT_THROW(A.getNthRowVector(-3), std::invalid_argument);
  EXPECT_THROW(A.getNthRowVector(-1), std::invalid_argument);
  EXPECT_EQ(A.getNthRowVector(0), std::vector<int>({3, 2, 1}));
  EXPECT_EQ(A.getNthRowVector(1), std::vector<int>({4, 0, 1}));
  EXPECT_EQ(A.getNthRowVector(2), std::vector<int>({9, 7, 3}));
  EXPECT_THROW(A.getNthRowVector(3), std::invalid_argument);
  EXPECT_THROW(A.getNthRowVector(34), std::invalid_argument);
}

TEST_F(MatrixTestFixture, testRowRetrievalEmptyMatrix) {  /* NOLINT */
  Matrix<int> A;
  EXPECT_THROW(A.getNthRowVector(-1), std::invalid_argument);
  EXPECT_THROW(A.getNthRowVector(0), std::invalid_argument);
}

TEST_F(MatrixTestFixture, testRowRetrievalSingleElement) {  /* NOLINT */
  Matrix<int> A({{3}});
  EXPECT_EQ(A.getNthRowVector(0), std::vector<int>({3}));
}

TEST_F(MatrixTestFixture, testColumnRetrieval) {  /* NOLINT */
  Matrix<int> A({{3, 2, 1}, {4, 0, 1}, {9, 7, 3}});
  EXPECT_THROW(A.getNthColumnVector(-1), std::invalid_argument);
  EXPECT_EQ(A.getNthColumnVector(0), std::vector<int>({3, 4, 9}));
  EXPECT_EQ(A.getNthColumnVector(1), std::vector<int>({2, 0, 7}));
  EXPECT_EQ(A.getNthColumnVector(2), std::vector<int>({1, 1, 3}));
  EXPECT_THROW(A.getNthColumnVector(3), std::invalid_argument);
  EXPECT_THROW(A.getNthColumnVector(11), std::invalid_argument);
}

TEST_F(MatrixTestFixture, testColumnRetrievalEmptyMatrix) {  /* NOLINT */
  Matrix<int> A;
  EXPECT_THROW(A.getNthColumnVector(-1), std::invalid_argument);
  EXPECT_THROW(A.getNthColumnVector(0), std::invalid_argument);
}

TEST_F(MatrixTestFixture, testColumnRetrievalSingleElement) {  /* NOLINT */
  Matrix<int> A({{3}});
  EXPECT_EQ(A.getNthColumnVector(0), std::vector<int>({3}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_rowVec_appendToEmptyMatrix) {  /* NOLINT */
  Matrix<int> A;
  A.appendVectorAt(0, new Matrix<int>({{1, 3, 3}}));
  EXPECT_EQ(A, Matrix<int>({{1, 3, 3}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_rowVec_appendByReplacingExisting) {  /* NOLINT */
  Matrix<int> A({{7, 5, 3, 2}, {4, 2, 1, 5}});
  A.appendVectorAt(1, new Matrix<int>({{5, 3, 2, 9}}));
  EXPECT_EQ(A, Matrix<int>({{7, 5, 3, 2}, {5, 3, 2, 9}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_rowVec_appendByExtendingExisting) {  /* NOLINT */
  Matrix<int> A({{7, 5, 3, 2}, {4, 2, 1, 5}});
  A.appendVectorAt(2, new Matrix<int>({{5, 3, 2, 9}}));
  EXPECT_EQ(A, Matrix<int>({{7, 5, 3, 2}, {4, 2, 1, 5}, {5, 3, 2, 9}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_rowVec_appendIncludingResize) {  /* NOLINT */
  Matrix<int> A({{7, 5, 3, 2}, {4, 2, 1, 5}});
  A.appendVectorAt(4, new Matrix<int>({{5, 3, 2, 9}}));
  EXPECT_EQ(A, Matrix<int>({{7, 5, 3, 2}, {4, 2, 1, 5}, {0, 0, 0, 0}, {0, 0, 0, 0}, {5, 3, 2, 9}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_emptyVec) {  /* NOLINT */
  Matrix<int> A;
  A.appendVectorAt(2, new Matrix<int>({{}}));
  EXPECT_EQ(A, Matrix<int>({{}, {}, {}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_columnVec_appendToEmptyMatrix) {  /* NOLINT */
  Matrix<int> A;
  A.appendVectorAt(0, new Matrix<int>({{1}, {3}, {4}}));
  EXPECT_EQ(A, Matrix<int>({{1}, {3}, {4}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_columnVec_appendByReplacingExisting) {  /* NOLINT */
  Matrix<int> A({{7, 4}, {5, 2}, {3, 1}, {2, 5}});
  A.appendVectorAt(1, new Matrix<int>({{5}, {3}, {2}, {9}}));
  EXPECT_EQ(A, Matrix<int>({{7, 5}, {5, 3}, {3, 2}, {2, 9}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_columnVec_appendByExtendingExisting) {  /* NOLINT */
  Matrix<int> A({{7, 4}, {5, 2}, {3, 1}, {2, 5}});
  A.appendVectorAt(2, new Matrix<int>({{9}, {0}, {3}, {7}}));
  EXPECT_EQ(A, Matrix<int>({{7, 4, 9}, {5, 2, 0}, {3, 1, 3}, {2, 5, 7}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_columnVec_appendIncludingResize) {  /* NOLINT */
  Matrix<int> A({{7, 4}, {5, 2}, {3, 1}, {2, 5}});
  A.appendVectorAt(4, new Matrix<int>({{7}, {4}, {3}, {2}}));
  EXPECT_EQ(A, Matrix<int>({{7, 4, 0, 0, 7}, {5, 2, 0, 0, 4}, {3, 1, 0, 0, 3}, {2, 5, 0, 0, 2}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_scalar) {  /* NOLINT */
  Matrix<int> A({{7}, {5}, {3}, {2}});
  A.appendVectorAt(2, new Matrix<int>({{5}}));
  EXPECT_EQ(A, Matrix<int>({{7}, {5}, {5}, {2}}));
}

TEST_F(MatrixTestFixture, testAppendVectorAt_notRowOrColumnVec_expectedException) {  /* NOLINT */
  Matrix<int> A;
  EXPECT_THROW(A.appendVectorAt(3, new Matrix<int>({{14, 31, 3}, {3, 1, 3}})), std::runtime_error);
}

TEST_F(MatrixTestFixture, testAppendVectorAt_dimensionMismatch_expectedException) {  /* NOLINT */
  Matrix<int> A({{7, 5, 3, 2}, {4, 2, 1, 5}});
  EXPECT_THROW(A.appendVectorAt(3, new Matrix<int>({{14, 31, 2, 9, 1}})), std::runtime_error);
}

class MatrixOperationFixture : public ::testing::Test {
//  ╔═══════════════════════════════════════════════════════════════╗
//  ║ Tested Combinations (non-exhaustive testing)                  ║
//  ╚═══════════════════════════════════════════════════════════════╝
//                             ┌────────────────────────────────────┐
//   Operator \ Combination    │ Matrix-Matrix       Matrix-Scalar  │
//  ┌──────────────────────────┼────────────────────────────────────┤
//  │ ADDITION                 │ int                 string         │
//  │ SUBTRACTION              │ int                 -              │
//  │ MULTIPLICATION           │ float               int            │
//  │ DIVISION                 │ -                   int            │
//  │ MODULO                   │ -                   int            │
//  ├──────────────────────────┼────────────────────────────────────┤
//  │ LOGICAL_AND              │ bool                bool           │
//  │ LOGICAL_OR               │ bool                bool           │
//  │ LOGICAL_XOR              │ bool                bool           │
//  ├──────────────────────────┼────────────────────────────────────┤
//  │ SMALLER                  │ -                   float          │
//  │ SMALLER_EQUAL            │ -                   float          │
//  │ GREATER                  │ -                   float          │
//  │ GREATER_EQUAL            │ -                   float          │
//  │ EQUAL                    │ int                 -              │
//  │ UNEQUAL                  │ int                 -              │
//  └──────────────────────────┴────────────────────────────────────┘
//  ┌───────────────────────────────────────────────────────────────┐
//  │ Sign "-" indicates an yet untested combination.               │
//  └───────────────────────────────────────────────────────────────┘

 protected:
  AbstractLiteral *intScalar1, *intScalar2, *intMatrix1, *intMatrix2, *intMatrix3, *boolScalar, *boolMatrix1,
      *boolMatrix2, *floatScalar1, *floatScalar2, *floatMatrix1, *floatMatrix2, *stringMatrix, *stringScalar;

 public:
  MatrixOperationFixture() { /* NOLINT */
    intScalar1 = new LiteralInt(42);
    intScalar2 = new LiteralInt(2);
    intMatrix1 = new LiteralInt(new Matrix<int>({{1, 2, 52}, {3, 4, 1}}));
    intMatrix2 = new LiteralInt(new Matrix<int>({{3, 2, 1}, {3, 4, 3}}));
    intMatrix3 = new LiteralInt(new Matrix<int>({{3, 9, 1}, {3, 4, 1}}));
    boolScalar = new LiteralBool(false);
    boolMatrix1 = new LiteralBool(new Matrix<bool>(
        {{true, false, false}, {false, true, false}, {false, false, true}}));
    boolMatrix2 = new LiteralBool(new Matrix<bool>(
        {{false, false, true}, {true, true, false}, {true, true, true}}));
    floatScalar1 = new LiteralFloat(2.22f);
    floatScalar2 = new LiteralFloat(2.5f);
    floatMatrix1 = new LiteralFloat(new Matrix<float>({{1.32f, 3.11f, 2.22f}, {2.55f, 2.1f, 6.2f}}));
    floatMatrix2 = new LiteralFloat(new Matrix<float>({{0.0f, 1.0f}, {3.5f, 2.5f}, {7.43f, 2.1f}}));
    stringScalar = new LiteralString("zh");
    stringMatrix = new LiteralString(new Matrix<std::string>({{"a", "b", "c"}, {"d", "e", "f"}, {"g", "h", "i"}}));
  }
};

TEST_F(MatrixOperationFixture, matrix_scalar_mult) {  /* NOLINT */
  auto result = Operator(ArithmeticOp::MULTIPLICATION).applyOperator(intScalar1, intMatrix1)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result), *new Matrix<int>({{42, 84, 2'184}, {126, 168, 42}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_div) {  /* NOLINT */
  auto result = Operator(ArithmeticOp::DIVISION).applyOperator(intMatrix1, intScalar2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result), *new Matrix<int>({{0, 1, 26}, {1, 2, 0}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_mod) {  /* NOLINT */
  auto result = Operator(ArithmeticOp::MODULO).applyOperator(intMatrix1, intScalar2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result), *new Matrix<int>({{1, 0, 0}, {1, 0, 1}}));
}

TEST_F(MatrixOperationFixture, matrix_matrix_sub) {  /* NOLINT */
  auto result = Operator(ArithmeticOp::SUBTRACTION).applyOperator(intMatrix1, intMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result), *new Matrix<int>({{-2, 0, 51}, {0, 0, -2}}));
}

TEST_F(MatrixOperationFixture, matrix_matrix_int) {  /* NOLINT */
  auto result = Operator(ArithmeticOp::ADDITION).applyOperator(intMatrix1, intMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result), *new Matrix<int>({{4, 4, 53}, {6, 8, 4}}));
}

TEST_F(MatrixOperationFixture, matrix_matrix_mult) {  /* NOLINT */
  auto result = Operator(ArithmeticOp::MULTIPLICATION).applyOperator(floatMatrix1, floatMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<float> *>(result),
            *new Matrix<float>({{27.3796005, 13.756999}, {53.4159966, 20.8199997}}));
}

TEST_F(MatrixOperationFixture, matrix_bool_negate) {  /* NOLINT */
  auto result = Operator(UnaryOp::NEGATION).applyOperator(boolMatrix1)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<bool> *>(result),
            *new Matrix<bool>({{false, true, true}, {true, false, true}, {true, true, false}}));
}

TEST_F(MatrixOperationFixture, matrix_int_negate) {  /* NOLINT */
  auto result = Operator(UnaryOp::NEGATION).applyOperator(intMatrix1)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result), *new Matrix<int>({{-1, -2, -52}, {-3, -4, -1}}));
}

TEST_F(MatrixOperationFixture, matrix_matrix_logicalAnd) {  /* NOLINT */
  auto result = Operator(LogCompOp::LOGICAL_AND).applyOperator(boolMatrix1, boolMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<bool> *>(result),
            *new Matrix<bool>({{false, false, false}, {false, true, false}, {false, false, true}}));
}

TEST_F(MatrixOperationFixture, matrix_matrix_logicalOr) {  /* NOLINT */
  auto result = Operator(LogCompOp::LOGICAL_OR).applyOperator(boolMatrix1, boolMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<bool> *>(result),
            *new Matrix<bool>({{true, false, true}, {true, true, false}, {true, true, true}}));
}

TEST_F(MatrixOperationFixture, matrix_matrix_logicalXor) {  /* NOLINT */
  auto result = Operator(LogCompOp::LOGICAL_XOR).applyOperator(boolMatrix1, boolMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<bool> *>(result),
            *new Matrix<bool>({{true, false, true}, {true, false, false}, {true, true, false}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_logicalAnd) {  /* NOLINT */
  auto result = Operator(LogCompOp::LOGICAL_AND).applyOperator(boolScalar, boolMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<bool> *>(result),
            *new Matrix<bool>({{false, false, false}, {false, false, false}, {false, false, false}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_logicalOr) {  /* NOLINT */
  auto result = Operator(LogCompOp::LOGICAL_OR).applyOperator(boolScalar, boolMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<bool> *>(result),
            *new Matrix<bool>({{false, false, true}, {true, true, false}, {true, true, true}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_logicalXor) {  /* NOLINT */
  auto result = Operator(LogCompOp::LOGICAL_XOR).applyOperator(boolScalar, boolMatrix2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<bool> *>(result),
            *new Matrix<bool>({{false, false, true}, {true, true, false}, {true, true, true}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_smaller) {  /* NOLINT */
  auto result = Operator(LogCompOp::SMALLER).applyOperator(floatMatrix2, floatScalar2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<float> *>(result),
            *new Matrix<float>({{1.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_smallerEqual) {  /* NOLINT */
  auto result = Operator(LogCompOp::SMALLER_EQUAL).applyOperator(floatMatrix2, floatScalar2)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<float> *>(result),
            *new Matrix<float>({{1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_greater) {  /* NOLINT */
  auto result = Operator(LogCompOp::GREATER).applyOperator(floatMatrix1, floatScalar1)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<float> *>(result),
            *new Matrix<float>({{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_greaterEqual) {  /* NOLINT */
  auto result = Operator(LogCompOp::GREATER_EQUAL).applyOperator(floatMatrix1, floatScalar1)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<float> *>(result),
            *new Matrix<float>({{0.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}));
}

TEST_F(MatrixOperationFixture, matrix_matrix_equal) {  /* NOLINT */
  auto result = Operator(LogCompOp::EQUAL).applyOperator(intMatrix2, intMatrix3)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result),
            *new Matrix<int>({{1, 0, 1}, {1, 1, 0}}));
}

TEST_F(MatrixOperationFixture, matrix_matrix_unequal) {  /* NOLINT */
  auto result = Operator(LogCompOp::UNEQUAL).applyOperator(intMatrix2, intMatrix3)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<int> *>(result),
            *new Matrix<int>({{0, 1, 0}, {0, 0, 1}}));
}

TEST_F(MatrixOperationFixture, matrix_scalar_add) {  /* NOLINT */
  auto result = Operator(ArithmeticOp::ADDITION).applyOperator(stringMatrix, stringScalar)->getMatrix();
  EXPECT_EQ(*dynamic_cast<Matrix<std::string> *>(result),
            *new Matrix<std::string>({{"azh", "bzh", "czh"}, {"dzh", "ezh", "fzh"}, {"gzh", "hzh", "izh"}}));
}

