#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_MOCKUP_CLASSES_CIPHERTEXT_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_MOCKUP_CLASSES_CIPHERTEXT_H_

#include <vector>
#include <ast_opt/ast/AbstractExpr.h>

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#endif

class Ciphertext {
 private:

#ifdef HAVE_SEAL_BFV
  /// the encrypted data in this Ciphertext
  seal::Ciphertext ciphertext;

  /// the seal context, i.e. object that holds params/etc
  static std::shared_ptr<seal::SEALContext> context;

  /// secret key, also used for (more efficient) encryption
  static seal::SecretKey secretKey;

  /// public key
  static seal::PublicKey publicKey;

  /// keys required to rotate
  static seal::GaloisKeys galoisKeys;
#endif

  /// plaintext data for testing
  std::vector<double> data;

  /// the offset to the first element after applying any operations on the ciphertext, e.g.,
  /// initial data [a b c d ... z] -> rot(7) -> [h i j k l ... z a b c .. g] then offset would be 7.
  int offsetOfFirstElement{0};

  /// the number of elements in the ciphertext starting at position offsetOfFirstElement
  int numCiphertextElements{0};

  Ciphertext sumAndRotate(int initialRotationFactor);

 protected:
  const static int DEFAULT_NUM_SLOTS = 8'192;

 public:
  Ciphertext() = default;

  explicit Ciphertext(std::vector<double> data, int numCiphertextSlots = DEFAULT_NUM_SLOTS);

  explicit Ciphertext(double scalar, int numCiphertextSlots = DEFAULT_NUM_SLOTS);

  Ciphertext(const Ciphertext &ctxt); // copy constructor

  Ciphertext operator+(const Ciphertext &ctxt) const;

  Ciphertext operator+(double plaintextScalar) const;

  Ciphertext operator*(const Ciphertext &ctxt) const;

  Ciphertext operator*(double plaintextScalar) const;

  Ciphertext operator-(const Ciphertext &ctxt) const;

  Ciphertext operator-(double plaintextScalar) const;

  Ciphertext operator/(const Ciphertext &ctxt) const;

  Ciphertext operator/(double plaintextScalar) const;

  bool operator==(const Ciphertext &rhs) const;

  bool operator!=(const Ciphertext &rhs) const;

  void verifyNumElementsAndAlignment(const Ciphertext &ctxt) const;

  Ciphertext applyBinaryOp(const std::function<double(double, double)> &binaryOp, const Ciphertext &lhs,
                           const Ciphertext &rhs) const;

  //TODO: What does this do?
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

  void printCiphertextData();
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_MOCKUP_CLASSES_CIPHERTEXT_H_
