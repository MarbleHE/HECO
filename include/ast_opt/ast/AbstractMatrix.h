#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTMATRIX_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTMATRIX_H_

#include <string>
#include <nlohmann/json.hpp>
#include "AbstractNode.h"

class Dimension;
class AbstractLiteral;
class Operator;
class AbstractExpr;

template<typename T>
class Matrix;

using json = nlohmann::json;

/// A helper class that allows to access and use all Matrix<T> instances using a unified interface, without
/// needing exact knowledge about the specific template type T.
class AbstractMatrix : public AbstractNode {
 public:
  /// Rotates a matrix by a given rotationFactor.
  /// \param rotationFactor Determines the number of rotations and its direction. If rotationFactor is positive (>0),
  ///        then the elements are rotated to the right, otherwise (rotationFactor<0) the elements are rotated to the
  ///        left.
  /// \param inPlace If True, then modifies the current matrix and returns the modified matrix. In case that
  ///        inPlace is False, keeps the current matrix untouched and returns instead a transposed copy.
  /// \return The rotated matrix that is either the rotated current matrix (inPlace=True) or a rotated copy of
  ///         the current matrix (inPlace=False).
  virtual AbstractMatrix *rotate(int rotationFactor, bool inPlace) = 0;

  /// Transposes a matrix, i.e., moves every element a_ij to a_ji.
  /// \param inPlace If True, then modifies the current matrix and returns the modified matrix. In case that
  ///        inPlace is False, keeps the current matrix untouched and returns instead a transposed copy.
  /// \return The transposed matrix that is either the transposed current matrix (inPlace=True) or a transposed copy of
  ///         the current matrix (inPlace=False).
  virtual AbstractMatrix *transpose(bool inPlace) = 0;

  /// Returns the dimension object that indicates the matrix dimensions.
  /// \return A reference to the dimension object associated to this matrix.
  virtual Dimension &getDimensions() = 0;

  /// Creates a string representation of this matrix using the matrix text representation by MATLAB.
  /// For example, a 3x2 matrix would look like [2 3; 2 332; 43 3] where the semicolon (;) serves as row delimiter.
  /// \return A string representation of this matrix.
  virtual std::string toString() = 0;

  /// Checks whether this matrix is a scalar, i.e., has dimension (1,1).
  /// \return True if this matrix is a scalar, otherwise False.
  [[nodiscard]] virtual bool isScalar() const = 0;

  /// Checks whether this matrix is empty, i.e., has dimension (0,0).
  /// \return True if this matrix is empty, otherwise False.
  [[nodiscard]] virtual bool isEmpty() const = 0;

  /// Creates a JSON representation of this matrix.
  /// \return The JSON representation of this matrix.
  [[nodiscard]] json toJson() const override = 0;

  /// A helper method that is defined in AbstractMatrix and overridden here. It casts the first parameter
  /// (AbstractMatrix) to Matrix<T> where T is the template type (e.g., int, float). This is needed to call the suitable
  /// applyBinaryOperatorComponentwise method that handles the binary operator application.
  /// \param rhsOperand The operand on the right hand-side. The current matrix is the left hand-side operand.
  /// \param op The operator to be applied on the two given matrices.
  /// \return The AbstractMatrix resulting from applying the operator op on the two matrices.
  virtual AbstractMatrix *applyBinaryOperator(AbstractMatrix *rhsOperand, Operator *os) = 0;

  /// Applies the unary operator specified by the given Operator os to every element of the matrix.
  /// \param os The operator to be applied to every element of the matrix.
  /// \return A new matrix where the Operator as was applied to every element.
  virtual AbstractMatrix *applyUnaryOperatorComponentwise(Operator *os) = 0;

  /// Returns an AbstractExpr containing the value of the element at index specified by the given (row, column) pair.
  /// The value is an AbstractExpr pointer because matrix elements can be defined using expressions.
  /// \param row The row number of the element to be returned as a pointer.
  /// \param column The column number of the element to be returned as a pointer.
  /// \return An AbstractExpr pointer pointing to the value of the element at position (row, column).
  virtual AbstractExpr *getElementAt(int row, int column) = 0;

  /// Sets a new value to the matrix element at the position indicated by (row, column) parameters.
  /// \param row The row index of the element where the given value should be written to.
  /// \param column The column index of the element where the given value should be written to.
  /// \param value The value to write to the given matrix position.
  virtual void setElementAt(int row, int column, AbstractExpr *value) = 0;

  /// Returns True if this AbstractMatrix is of template type AbstractExpr*, i.e., a Matrix<AbstractExpr*>.
  /// \return True if this is a Matrix<AbstractExpr*>, otherwise False.
  virtual bool containsAbstractExprs() = 0;

  /// Appends a row or column to the current matrix, depending on the structure of the given matrix mx. Note that our
  /// model does not have a dedicated class for vectors, thus we require that either a row vector (i.e., single row)
  /// or a column vector (i.e., single column) is given. The row/column vector must have the same dimension as this
  /// matrix, otherwise an exception will be thrown.
  /// The row/column is appended to the position indicated by the given idx parameter, this is realized as follow:
  /// - If the matrix has exactly idx-1 rows/columns, the given row/column is simply appended.
  /// - If the matrix has less than idx-1 rows/columns, the matrix is resized by filling up empty elements by
  ///   the underlying type's default value (e.g., 0 for integer) and then the given row/column is appended.
  /// - If the matrix has more than idx-1 rows/columns, the row/column indicated at index idx is overwritten.
  /// \param idx Indicates the position (index) in the matrix at which the given row/column vector should be
  /// appended to. For example 3 denotes the third row or column.
  /// \param mx A vector, i.e., (1,x)- or (x,1)-matrix containing the elements to append. If this a row vector, the
  /// given idx will be interpreted as row index where this new row should be added. If this a column vector, the
  /// given idx will be interpreted as column index where this new column should be added.
  /// \throws std::runtime_exception Throws a std::runtime_exception if mx is either a row nor a column vector, or if
  /// the number of elements in mx and this matrix do not match.
  virtual void appendVectorAt(int idx, AbstractMatrix *mx) = 0;

  bool operator==(const AbstractMatrix &rhs) const;

  bool operator!=(const AbstractMatrix &rhs) const;

  AbstractMatrix *clone(bool keepOriginalUniqueNodeId) override = 0;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_AST_ABSTRACTMATRIX_H_
