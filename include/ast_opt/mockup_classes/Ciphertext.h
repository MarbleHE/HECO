#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_MOCKUP_CLASSES_CIPHERTEXT_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_MOCKUP_CLASSES_CIPHERTEXT_H_

#include <vector>
#include <ast_opt/ast/AbstractExpr.h>

class Ciphertext {
 private:
  /// the encrypted data in this Ciphertext (supports plaintext data for testing)
  std::vector<double> data;

  /// the offset to the first element after applying any operations on the ciphertext, e.g.,
  /// initial data [a b c d ... z] -> rot(7) -> [h i j k l ... z a b c .. g] then offset would be 7.
  int offsetOfFirstElement;

  /// the number of elements in the ciphertext starting at position offsetOfFirstElement
  int numCiphertextElements;

  Ciphertext sumAndRotate(int initialRotationFactor);

 public:
  explicit Ciphertext(std::vector<double> data, int numCiphertextSlots = 8'192);

  Ciphertext(const Ciphertext &ctxt); // copy constructor

  Ciphertext operator+(const Ciphertext &ctxt) const;

  Ciphertext operator+(double plaintextScalar) const;

  Ciphertext operator*(const Ciphertext &ctxt) const;

  Ciphertext operator*(double plaintextScalar) const;

  bool operator==(const Ciphertext &rhs) const;

  bool operator!=(const Ciphertext &rhs) const;

  void verifyNumElementsAndAlignment(const Ciphertext &ctxt) const;

  Ciphertext applyBinaryOp(const std::function<double(double, double)> &binaryOp, const Ciphertext &lhs,
                           const Ciphertext &rhs) const;

  static int cyclicIncrement(int i, const std::vector<double> &vec);

  [[nodiscard]] int getNumCiphertextElements() const;

  [[nodiscard]] int getOffsetOfFirstElement() const;

  [[nodiscard]] int getNumCiphertextSlots() const;

  [[nodiscard]] int computeCyclicEndIndex(int startIndex, int numElements) const;

  [[nodiscard]] Ciphertext generateCiphertext(double plaintextScalar, int fillNSlots, int totalNumCtxtSlots) const;

  Ciphertext rotate(int n);

  double &getElementAt(int n);

  Ciphertext sumaAndRotateAll();

  Ciphertext sumAndRotatePartially(int numElementsToSum);

};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_MOCKUP_CLASSES_CIPHERTEXT_H_
