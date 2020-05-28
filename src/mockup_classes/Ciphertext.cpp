#include "ast_opt/mockup_classes/Ciphertext.h"
#include <iostream>
#include <cmath>

#ifdef HAVE_SEAL_BFV
// Initialize static members
std::shared_ptr<seal::SEALContext> Ciphertext::context = nullptr;

// SecretKey() actually works, ptr for consistency
std::unique_ptr<seal::SecretKey> Ciphertext::secretKey = nullptr;

// The default constructor used by SEAL in PublicKey() segfaults. Therefore, it's a ptr
std::unique_ptr<seal::PublicKey> Ciphertext::publicKey = nullptr;

// The default constructor used by SEAL in GaloisKey() segfaults. Therefore, it's a ptr
std::unique_ptr<seal::GaloisKeys> Ciphertext::galoisKeys = nullptr;

void setup_context(std::shared_ptr<seal::SEALContext> &context,
                   std::unique_ptr<seal::SecretKey> &secretKey,
                   std::unique_ptr<seal::PublicKey> &publicKey,
                   std::unique_ptr<seal::GaloisKeys> &galoisKeys) {
  if (!context || !context->parameters_set()) {
    /// Wrapper for parameters
    seal::EncryptionParameters params(seal::scheme_type::BFV);

    // in BFV, this degree is also the number of slots.
    params.set_poly_modulus_degree(Ciphertext::DEFAULT_NUM_SLOTS);

    // Let SEAL select a "suitable" coefficient modulus (not necessarily maximal)
    params.set_coeff_modulus(seal::CoeffModulus::BFVDefault(params.poly_modulus_degree()));

    // Let SEAL select a plaintext modulus that actually supports batching
    params.set_plain_modulus(seal::PlainModulus::Batching(params.poly_modulus_degree(), 20));

    // Instantiate context
    context = seal::SEALContext::Create(params);

    /// Helper object to create keys
    seal::KeyGenerator keyGenerator(context);

    secretKey = std::make_unique<seal::SecretKey>(keyGenerator.secret_key());
    publicKey = std::make_unique<seal::PublicKey>(keyGenerator.public_key());
    galoisKeys = std::make_unique<seal::GaloisKeys>(keyGenerator.galois_keys_local());
  }
}
#endif

Ciphertext::Ciphertext(std::vector<double> inputData, int numCiphertextSlots)
    : numCiphertextElements(inputData.size()), offsetOfFirstElement(0) {
  //TODO (Alex): numCiphertextSlots should probably be static after first init and input should  match exactly
  if (inputData.size() > numCiphertextSlots) {
    throw std::runtime_error("Cannot add more elements than ciphertext slots are available!");
  }
  data.insert(data.begin(), inputData.begin(), inputData.end());
  data.resize(numCiphertextSlots);
#ifdef HAVE_SEAL_BFV
  // Ensure context is setup
  setup_context(context, secretKey, publicKey, galoisKeys);

  /// Helper object for encoding
  seal::BatchEncoder batchEncoder(context);
  seal::Plaintext plaintext;

  // Throw exception if non-integer data and mode is BFV
  if (std::any_of(inputData.begin(), inputData.end(), [this](double d) { return !isInteger(d); })) {
    throw std::invalid_argument("Unsupported: Cannot create ciphertext of doubles when using SEAL with BFV.");
  }
  /// Encode, force conversion from double to int64
  batchEncoder.encode(std::vector<std::int64_t>(data.begin(), data.end()), plaintext);

  /// Helper object for encryption
  seal::Encryptor encryptor(context, *publicKey);

  encryptor.encrypt(plaintext, ciphertext);
#endif
}

Ciphertext::Ciphertext(double scalar, int numCiphertextSlots)
    : numCiphertextElements(numCiphertextSlots), offsetOfFirstElement(0) {
  data.resize(numCiphertextSlots);
  std::fill(data.begin(), data.end(), scalar);
#ifdef HAVE_SEAL_BFV
  // Ensure context is setup
  setup_context(context, secretKey, publicKey, galoisKeys);

  /// Helper object for encoding
  seal::BatchEncoder batchEncoder(context);
  seal::Plaintext plaintext;

  // Throw exception if non-integer data and mode is BFV
  if (!isInteger(scalar)) {
    throw std::invalid_argument("Unsupported: Cannot create ciphertext of double when using SEAL with BFV.");
  }
  /// Encode, force conversion from double to int64
  batchEncoder.encode(std::vector<std::int64_t>(data.begin(), data.end()), plaintext);

  /// Helper object for encryption
  seal::Encryptor encryptor(context, *publicKey);

  encryptor.encrypt(plaintext, ciphertext);
#endif
}

Ciphertext Ciphertext::operator+(const Ciphertext &ctxt) const {
  auto result = applyBinaryOp(std::plus<double>{}, *this, ctxt);
#ifdef HAVE_SEAL_BFV
  seal::Evaluator evaluator(context);
  evaluator.add(ciphertext, ctxt.ciphertext, result.ciphertext);
#endif
  return result;
}

Ciphertext Ciphertext::operator+(const double plaintextScalar) const {
  Ciphertext ctxt = generateCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  auto result = applyBinaryOp(std::plus<double>{}, *this, ctxt);
#ifdef HAVE_SEAL_BFV
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batchEncoder(context);
  seal::Plaintext plaintext;
  batchEncoder.encode(std::vector<int64_t>(getNumCiphertextSlots(), plaintextScalar), plaintext);
  evaluator.add_plain(ciphertext, plaintext, result.ciphertext);
#endif
  return result;
}

Ciphertext Ciphertext::generateCiphertext(const double plaintextScalar, int fillNSlots, int totalNumCtxtSlots) const {
  Ciphertext ctxt({}, totalNumCtxtSlots);
  ctxt.offsetOfFirstElement = getOffsetOfFirstElement();
  auto targetIdx = computeCyclicEndIndex(ctxt.getOffsetOfFirstElement(), getNumCiphertextElements());
  for (auto i = offsetOfFirstElement; i!=cyclicIncrement(targetIdx, ctxt.data); i = cyclicIncrement(i, ctxt.data)) {
    ctxt.data[i] = plaintextScalar;
  }
  ctxt.numCiphertextElements = getNumCiphertextElements();
  return ctxt;
}

Ciphertext Ciphertext::operator*(const Ciphertext &ctxt) const {
  auto result = applyBinaryOp(std::multiplies<double>{}, *this, ctxt);
#ifdef HAVE_SEAL_BFV
  seal::Evaluator evaluator(context);
  evaluator.multiply(ciphertext, ctxt.ciphertext, result.ciphertext);
#endif
  return result;
}

Ciphertext Ciphertext::operator-(const Ciphertext &ctxt) const {
  auto result = applyBinaryOp(std::minus<double>{}, *this, ctxt);
#ifdef HAVE_SEAL_BFV
  seal::Evaluator evaluator(context);
  evaluator.sub(ciphertext, ctxt.ciphertext, result.ciphertext);
#endif
  return result;
}

Ciphertext Ciphertext::operator-(double plaintextScalar) const {
  Ciphertext ctxt = generateCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  auto result = applyBinaryOp(std::minus<double>{}, *this, ctxt);
#ifdef HAVE_SEAL_BFV
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batchEncoder(context);
  seal::Plaintext plaintext;
  batchEncoder.encode(std::vector<int64_t>(getNumCiphertextSlots(), plaintextScalar), plaintext);
  evaluator.sub_plain(ciphertext, plaintext, result.ciphertext);
#endif
  return result;
}

Ciphertext Ciphertext::operator/(const Ciphertext &ctxt) const {
#ifdef HAVE_SEAL_BFV
  throw std::runtime_error("Cannot perform division on encrypted data when using BFV.");
#endif
  return applyBinaryOp(std::divides<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::operator/(double plaintextScalar) const {
#ifdef HAVE_SEAL_BFV
  throw std::runtime_error("Cannot perform division on encrypted data when using BFV.");
#endif
  Ciphertext ctxt = generateCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  return applyBinaryOp(std::divides<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::operator*(const double plaintextScalar) const {
  Ciphertext ctxt = generateCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  auto result = applyBinaryOp(std::multiplies<double>{}, *this, ctxt);
#ifdef HAVE_SEAL_BFV
  seal::Evaluator evaluator(context);
  seal::BatchEncoder batchEncoder(context);
  seal::Plaintext plaintext;
  batchEncoder.encode(std::vector<int64_t>(getNumCiphertextSlots(), plaintextScalar), plaintext);
  evaluator.multiply_plain(ciphertext, plaintext, result.ciphertext);
#endif
  return result;
}

int Ciphertext::computeCyclicEndIndex(int startIndex, int numElements) const {
  // computes the target index
  // for example, startIndex = 32, numElement = 112, numCtxtSlots = 128 results in
  // (32 + 112) % 128 = 144 % 128 = 16, i.e., first element is in slot 32, second in slot 33, etc. and last in slot
  // 16-1=15
  return ((startIndex + numElements)%getNumCiphertextSlots()) - 1;
}

int Ciphertext::getNumCiphertextElements() const {
  return numCiphertextElements;
}

void Ciphertext::verifyNumElementsAndAlignment(const Ciphertext &ctxt) const {
  std::stringstream ss;
  if (getNumCiphertextElements()!=ctxt.getNumCiphertextElements()) {
    ss << "Cannot perform action on ciphertexts as number of elements differ: " << getNumCiphertextElements();
    ss << " vs. " << ctxt.getNumCiphertextElements() << std::endl;
    throw std::runtime_error(ss.str());
  } else if (getOffsetOfFirstElement()!=ctxt.getOffsetOfFirstElement()) {
    ss << "Cannot peform action on ciphertexts as offset differs: " << getOffsetOfFirstElement();
    ss << " vs. " << ctxt.getOffsetOfFirstElement() << std::endl;
    throw std::runtime_error(ss.str());
  }
}

int Ciphertext::getOffsetOfFirstElement() const {
  return offsetOfFirstElement;
}

int Ciphertext::getNumCiphertextSlots() const {
  return data.capacity();
}

bool Ciphertext::operator==(const Ciphertext &rhs) const {
#ifdef HAVE_SEAL_BFV
  throw std::runtime_error("Cannot perform boolean tests on encrypted data when using BFV.");
#endif
  return data==rhs.data &&
      offsetOfFirstElement==rhs.offsetOfFirstElement &&
      numCiphertextElements==rhs.numCiphertextElements;
}

bool Ciphertext::operator!=(const Ciphertext &rhs) const {
#ifdef HAVE_SEAL_BFV
  throw std::runtime_error("Cannot perform boolean tests on encrypted data when using BFV.");
#endif
  return !(rhs==*this);
}

Ciphertext Ciphertext::applyBinaryOp(const std::function<double(double, double)> &binaryOp,
                                     const Ciphertext &lhs,
                                     const Ciphertext &rhs) const {
  //lhs.verifyNumElementsAndAlignment(rhs);
  if (lhs.getNumCiphertextSlots()!=rhs.getNumCiphertextSlots())
    throw std::runtime_error("");

  Ciphertext result({}, lhs.getNumCiphertextSlots());
  result.offsetOfFirstElement = lhs.getOffsetOfFirstElement() + rhs.getOffsetOfFirstElement();

  for (auto i = 0; i < lhs.getNumCiphertextSlots(); ++i) {
    result.data[i] = binaryOp(lhs.data[i], rhs.data[i]);
  }
  result.numCiphertextElements = getNumCiphertextElements();

  return result;
}

Ciphertext::Ciphertext(const Ciphertext &ctxt)
    : data(ctxt.data),
      offsetOfFirstElement(ctxt.offsetOfFirstElement),
      numCiphertextElements(ctxt.numCiphertextElements)
#ifdef HAVE_SEAL_BFV
    , ciphertext(ctxt.ciphertext)
#endif
{

}

int Ciphertext::cyclicIncrement(int i, const std::vector<double> &vec) {
  return (i < vec.size()) ? i + 1 : 0;
}

Ciphertext Ciphertext::rotate(int n) {
  if (n==0 || getNumCiphertextElements()==0) return Ciphertext(*this);

  Ciphertext result = *this;
  auto rotTarget = (n > 0) ? (result.data.begin() + result.getNumCiphertextSlots() - n) : (result.data.begin() - n);
  std::rotate(result.data.begin(), rotTarget, result.data.end());
  result.offsetOfFirstElement = (getOffsetOfFirstElement() + n)%result.getNumCiphertextSlots();
#ifdef HAVE_SEAL_BFV
  seal::Evaluator evaluator(context);
  evaluator.rotate_rows_inplace(result.ciphertext, n, *galoisKeys);
#endif
  return result;
}

double &Ciphertext::getElementAt(int n) {
  return data.at(n);
}

std::vector<std::int64_t> Ciphertext::decryptAndDecode() {
#ifdef HAVE_SEAL_BFV
  seal::Plaintext plaintext;

  // Helper object for decryption
  seal::Decryptor decryptor(context, *secretKey);
  decryptor.decrypt(ciphertext, plaintext);

  // Helper object for decoding
  seal::BatchEncoder batchEncoder(context);
  std::vector<std::int64_t> decodedValues;
  batchEncoder.decode(plaintext, decodedValues);

  return decodedValues;
#else
  std::vector<std::int64_t> r;
  r.reserve(data.size());
  for(auto &d: data) {
    r.push_back(d); //implicit conversion
  }
  return r;
#endif
}

Ciphertext Ciphertext::sumaAndRotateAll() {
  return sumAndRotate(getNumCiphertextSlots()/2);
}

Ciphertext Ciphertext::sumAndRotatePartially(int numElementsToSum) {
  return sumAndRotate(numElementsToSum/2); //TODO: Why divide by two?
}

Ciphertext Ciphertext::sumAndRotate(int initialRotationFactor) {
  int rotationFactor = initialRotationFactor;
  // create a copy of this ctxt as otherwise we would need to treat the first iteration differently
  auto ctxt = *this;
#ifdef HAVE_SEAL_BFV
  seal::Evaluator evaluator(context);
#endif
  // perform rotate-and-sum in total requiring log_2(#ciphertextSlots) rotations
  while (rotationFactor >= 1) {
    auto rotatedCtxt = ctxt.rotate(rotationFactor);
#ifdef HAVE_SEAL_BFV
    evaluator.rotate_rows_inplace(rotatedCtxt.ciphertext, rotationFactor, *galoisKeys);
#endif
    ctxt = ctxt + rotatedCtxt;
    rotationFactor = rotationFactor/2;
  }
  return ctxt;
}

void Ciphertext::printCiphertextData() {
  for (double val : data) {
    std::cout << val << std::endl;
  }
}

bool Ciphertext::isInteger(double k) {
  return std::floor(k)==k;
}
Ciphertext::Ciphertext() {
  data = std::vector<double>(numCiphertextElements);
#ifdef HAVE_SEAL_BFV
  ciphertext = seal::Ciphertext(context);
#endif
}
Ciphertext::Ciphertext(Ciphertext &&ctxt) : data(std::move(ctxt.data)),
                                            offsetOfFirstElement(ctxt.offsetOfFirstElement),
                                            numCiphertextElements(ctxt.numCiphertextElements)
#ifdef HAVE_SEAL_BFV
    , ciphertext(std::move(ctxt.ciphertext))
#endif
{}

Ciphertext &Ciphertext::operator=(const Ciphertext &ctxt) {
  ciphertext = seal::Ciphertext(ctxt.ciphertext);
  offsetOfFirstElement = ctxt.offsetOfFirstElement;
  numCiphertextElements = ctxt.numCiphertextElements;
  data = ctxt.data;
  return *this;
}
Ciphertext &Ciphertext::operator=(Ciphertext &&ctxt) {
  ciphertext = std::move(seal::Ciphertext(ctxt.ciphertext));
  offsetOfFirstElement =  std::move(ctxt.offsetOfFirstElement);
  numCiphertextElements = std::move(ctxt.numCiphertextElements);
  data = std::move(ctxt.data);
  return *this;
}
