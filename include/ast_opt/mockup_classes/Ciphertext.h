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

  /// secret key, also used for (more efficient) encryption (ptr for consistency)
  static std::unique_ptr<seal::SecretKey> secretKey;

  /// public key (ptr because PublicKey() segfaults)
  static std::unique_ptr<seal::PublicKey> publicKey;

  /// keys required to rotate (ptr because GaloisKeys() segfaults)
  static std::unique_ptr<seal::GaloisKeys> galoisKeys;
#endif

  /// plaintext data for testing
  std::vector<double> data;

  /// the offset to the first element after applying any operations on the ciphertext, e.g.,
  /// initial data [a b c d ... z] -> rot(7) -> [h i j k l ... z a b c .. g] then offset would be 7.
  int offsetOfFirstElement{0};

  /// the number of elements in the ciphertext starting at position offsetOfFirstElement
  int numCiphertextElements{0};

  Ciphertext sumAndRotate(int initialRotationFactor);

 public:
  Ciphertext() = default;

  explicit Ciphertext(std::vector<double> data, int numCiphertextSlots = DEFAULT_NUM_SLOTS);

  explicit Ciphertext(double scalar, int numCiphertextSlots = DEFAULT_NUM_SLOTS);

  Ciphertext(const Ciphertext &ctxt); // copy constructor

  static bool isInteger(double k);

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

  std::vector<int64_t> decryptAndDecode();

  const static int DEFAULT_NUM_SLOTS = 16'384;
};

#ifdef HAVE_SEAL_BFV
/// Hack until we have real parameter setup
/// Sets up a context and keys, iff the context is not yet setup
/// Takes everything by reference, so we don't have to make it a friend and expose its existence in the header
/// \param context
/// \param secretKey
/// \param publicKey
/// \param galoisKeys
void setup_context(std::shared_ptr<seal::SEALContext> &context,
                   std::unique_ptr<seal::SecretKey> &secretKey,
                   std::unique_ptr<seal::PublicKey> &publicKey,
                   std::unique_ptr<seal::GaloisKeys> &galoisKeys);
#endif

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_MOCKUP_CLASSES_CIPHERTEXT_H_
