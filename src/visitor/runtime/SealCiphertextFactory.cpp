#include "ast_opt/visitor/runtime/SealCiphertext.h"
#include "ast_opt/visitor/runtime/SealCiphertextFactory.h"
#include "ast_opt/visitor/runtime/Cleartext.h"

#ifdef HAVE_SEAL_BFV
#include <memory>
#include <seal/seal.h>

std::unique_ptr<AbstractCiphertext> SealCiphertextFactory::createCiphertext(const std::vector<int64_t> &data) const {
  auto ptxt = createPlaintext(data);
  std::unique_ptr<SealCiphertext> ctxt = std::make_unique<SealCiphertext>(*this);
  encryptor->encrypt(*ptxt, ctxt->getCiphertext());
  return ctxt;
}

std::unique_ptr<AbstractCiphertext> SealCiphertextFactory::createCiphertext(const std::vector<int> &data) const {
  std::vector<int64_t> ciphertextData(data.begin(), data.end());
  return createCiphertext(ciphertextData);
}

std::unique_ptr<AbstractCiphertext> SealCiphertextFactory::createCiphertext(int64_t data) const {
  std::vector<int64_t> values = {data};
  return createCiphertext(values);
}

SealCiphertextFactory::SealCiphertextFactory(const SealCiphertextFactory &other) :
    ciphertextSlotSize(other.ciphertextSlotSize),
    context(other.context), // TODO: This should be a real copy, not just shared ownership (copying the shared_ptr)
    secretKey(std::make_unique<seal::SecretKey>(*other.secretKey)),
    publicKey(std::make_unique<seal::PublicKey>(*other.publicKey)),
    galoisKeys(std::make_unique<seal::GaloisKeys>(*other.galoisKeys)),
    relinKeys(std::make_unique<seal::RelinKeys>(*other.relinKeys)),
    encoder(std::make_unique<seal::BatchEncoder>(*context)),
    evaluator(std::make_unique<seal::Evaluator>(*other.context)),
    encryptor(std::make_unique<seal::Encryptor>(*context, *publicKey)),
    decryptor(std::make_unique<seal::Decryptor>(*other.context, *secretKey)) {  // copy constructor
}

SealCiphertextFactory::SealCiphertextFactory(SealCiphertextFactory &&other) noexcept // move constructor
    : ciphertextSlotSize(other.ciphertextSlotSize), context(std::move(other.context)),
      secretKey(std::move(other.secretKey)), publicKey(std::move(other.publicKey)),
      galoisKeys(std::move(other.galoisKeys)), relinKeys(std::move(other.relinKeys)),
      encoder(std::move(other.encoder)), evaluator(std::move(other.evaluator)),
      encryptor(std::move(other.encryptor)), decryptor(std::move(other.decryptor)) {
}

SealCiphertextFactory &SealCiphertextFactory::operator=(const SealCiphertextFactory &other) {  // copy assignment
  return *this = SealCiphertextFactory(other);
}

SealCiphertextFactory &SealCiphertextFactory::operator=(SealCiphertextFactory &&other) noexcept {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;

  if (ciphertextSlotSize!=other.ciphertextSlotSize) {
    std::cerr << "Move assignment failed as const ciphertextSlotSize differs and cannot be changed!" << std::endl;
    exit(1);
  }

  context = std::move(other.context);
  secretKey = std::move(other.secretKey);
  publicKey = std::move(other.publicKey);
  galoisKeys = std::move(other.galoisKeys);
  relinKeys = std::move(other.relinKeys);
  encoder = std::move(other.encoder);
  evaluator = std::move(other.evaluator);
  encryptor = std::move(other.encryptor);
  decryptor = std::move(other.decryptor);
  return *this;
}

void SealCiphertextFactory::setupSealContext() {
  // Wrapper for parameters
  seal::EncryptionParameters params(seal::scheme_type::bfv);

  // in BFV, this degree is also the number of slots.
  params.set_poly_modulus_degree(ciphertextSlotSize);

  // Let SEAL select a "suitable" coefficient modulus (not necessarily optimal)
  params.set_coeff_modulus(seal::CoeffModulus::BFVDefault(params.poly_modulus_degree()));

  // Let SEAL select a plaintext modulus and 20 bit primes that actually support batching
  params.set_plain_modulus(seal::PlainModulus::Batching(params.poly_modulus_degree(), 20));

  // Instantiate context
  context = std::make_shared<seal::SEALContext>(params);

  // Create keys
  seal::KeyGenerator keyGenerator(*context);
  secretKey = std::make_unique<seal::SecretKey>(keyGenerator.secret_key());
  keyGenerator.create_public_key(*publicKey);
  keyGenerator.create_galois_keys(*galoisKeys);
  keyGenerator.create_relin_keys(*relinKeys);

  // Create helpers for en-/decoding, en-/decryption, and ciphertext evaluation
  encoder = std::make_unique<seal::BatchEncoder>(*context);
  encryptor = std::make_unique<seal::Encryptor>(*context, *publicKey);
  decryptor = std::make_unique<seal::Decryptor>(*context, *secretKey);
  evaluator = std::make_unique<seal::Evaluator>(*context);
}

template<typename T>
std::vector<T> SealCiphertextFactory::expandVector(const std::vector<T> &values) const {
  // passing the vector by value to implicitly get a copy somehow didn't work here
  std::vector<T> expandedVector(values.begin(), values.end());
  if (expandedVector.size() > encoder->slot_count()) {
    throw std::runtime_error("Cannot encode " + std::to_string(expandedVector.size())
                                 + " elements in a ciphertext of size "
                                 + std::to_string(encoder->slot_count()) + ". ");
  }
  // fill vector up to size of ciphertext with last element in given expandedVector
  auto lastValue = expandedVector.back();
  expandedVector.insert(expandedVector.end(), encoder->slot_count() - expandedVector.size(), lastValue);
  return expandedVector;
}

std::unique_ptr<seal::Plaintext> SealCiphertextFactory::createPlaintext(int64_t value) const {
  std::vector<int64_t> valueAsVec = {value};
  return createPlaintext(valueAsVec);
}

std::unique_ptr<seal::Plaintext> SealCiphertextFactory::createPlaintext(const std::vector<int> &value) const {
  std::vector<int64_t> vecInt64(value.begin(), value.end());
  return createPlaintext(vecInt64);
}

std::unique_ptr<seal::Plaintext> SealCiphertextFactory::createPlaintext(const std::vector<int64_t> &value) const {
  auto expandedVector = expandVector(value);
  auto ptxt = std::make_unique<seal::Plaintext>();
  encoder->encode(expandedVector, *ptxt);
  return ptxt;
}

const seal::SEALContext &SealCiphertextFactory::getContext() const {
  return *context;
}

const seal::RelinKeys &SealCiphertextFactory::getRelinKeys() const {
  return *relinKeys;
}

const seal::SecretKey &SealCiphertextFactory::getSecretKey() const {
  return *secretKey;
}

void SealCiphertextFactory::decryptCiphertext(AbstractCiphertext &abstractCiphertext,
                                              std::vector<int64_t> &ciphertextData) const {
  auto &ctxt = dynamic_cast<SealCiphertext &>(abstractCiphertext);
  seal::Plaintext ptxt;
  decryptor->decrypt(ctxt.getCiphertext(), ptxt);
  encoder->decode(ptxt, ciphertextData);
}

seal::Evaluator &SealCiphertextFactory::getEvaluator() const {
  return *evaluator;
}

const seal::GaloisKeys &SealCiphertextFactory::getGaloisKeys() const {
  return *galoisKeys;
}
SealCiphertextFactory::SealCiphertextFactory() {
  setupSealContext();
}

SealCiphertextFactory::SealCiphertextFactory(unsigned int numElementsPerCiphertextSlot)
    : ciphertextSlotSize(numElementsPerCiphertextSlot) {
  setupSealContext();
}

unsigned int SealCiphertextFactory::getCiphertextSlotSize() const {
  return ciphertextSlotSize;
}

std::string SealCiphertextFactory::getString(AbstractCiphertext &abstractCiphertext) const {
  // decrypt the ciphertext to get its values
  std::vector<int64_t> plainValues;
  decryptCiphertext(abstractCiphertext, plainValues);

  // print ciphertext as: [ value 1, value2
  std::stringstream ss;
  ss << "[";
  for (const auto value : plainValues) {
    ss << " " << value << ", ";
  }
  ss.seekp(-1, ss.cur);  // set ptr to overwrite last comma
  ss << " ]";

  return ss.str();
}

std::unique_ptr<AbstractCiphertext> SealCiphertextFactory::createCiphertext(std::unique_ptr<AbstractValue> &&abstractValue) const {
  if (auto castedCleartext = dynamic_cast<Cleartext<int> *>(abstractValue.get())) {
    // extract data and from std::vector<int> to std::vector<int64_t>
    auto castedCleartextData = castedCleartext->getData();
    std::vector<int64_t> data(castedCleartextData.begin(), castedCleartextData.end());
    // avoid duplicate code -> delegate creation to other constructor
    return createCiphertext(data);
  } else {
    throw std::runtime_error("Cannot create ciphertext from any other than a Cleartext<int> as used ciphertext factory "
                             "(SealCiphertextFactory) uses BFV that only supports integers.");
  }
}

#endif // ifdef HAVE_SEAL_BFV
