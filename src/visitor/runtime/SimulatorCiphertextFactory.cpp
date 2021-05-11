#include "ast_opt/visitor/runtime/SimulatorCiphertext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/runtime/Cleartext.h"

#ifdef HAVE_SEAL_BFV
#include <memory>
#include <seal/seal.h>

std::unique_ptr<AbstractCiphertext> SimulatorCiphertextFactory::createCiphertext(const std::vector<int64_t> &data) {
  auto ptxt = createPlaintext(data);
  std::unique_ptr<SimulatorCiphertext> ctxt = std::make_unique<SimulatorCiphertext>(*this); // Constructs a simulator ciphertext given all the data
  ctxt->createFresh(ptxt); // calcs initial noise and sets the variables needed, also stores the plaintext as _plaintext
  return ctxt;
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertextFactory::createCiphertext(const std::vector<int> &data) {
  std::vector<int64_t> ciphertextData(data.begin(), data.end());
  return createCiphertext(ciphertextData);
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertextFactory::createCiphertext(int64_t data) {
  std::vector<int64_t> values = {data};
  return createCiphertext(values);
}

SimulatorCiphertextFactory::SimulatorCiphertextFactory(const SimulatorCiphertextFactory &other) :
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

SimulatorCiphertextFactory::SimulatorCiphertextFactory(SimulatorCiphertextFactory &&other) noexcept // move constructor
    : ciphertextSlotSize(other.ciphertextSlotSize), context(std::move(other.context)),
      secretKey(std::move(other.secretKey)), publicKey(std::move(other.publicKey)),
      galoisKeys(std::move(other.galoisKeys)), relinKeys(std::move(other.relinKeys)),
      encoder(std::move(other.encoder)), evaluator(std::move(other.evaluator)),
      encryptor(std::move(other.encryptor)), decryptor(std::move(other.decryptor)) {
}

SimulatorCiphertextFactory &SimulatorCiphertextFactory::operator=(const SimulatorCiphertextFactory &other) {  // copy assignment
  return *this = SimulatorCiphertextFactory(other);
}

SimulatorCiphertextFactory &SimulatorCiphertextFactory::operator=(SimulatorCiphertextFactory &&other) noexcept {  // move assignment
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

void SimulatorCiphertextFactory::setupSealContext() {
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
  keyGenerator = std::make_unique<seal::KeyGenerator>(*context);
  //NOTE: The following is inefficient, since we're doing a byte-by-byte copy of the key here!
  secretKey = std::make_unique<seal::SecretKey>(keyGenerator->secret_key());
  keyGenerator->create_public_key(*publicKey);
  keyGenerator->create_galois_keys(*galoisKeys);
  keyGenerator->create_relin_keys(*relinKeys);

  // Create helpers for en-/decoding, en-/decryption, and ciphertext evaluation
  encoder = std::make_unique<seal::BatchEncoder>(*context);
  encryptor = std::make_unique<seal::Encryptor>(*context, *publicKey);
  decryptor = std::make_unique<seal::Decryptor>(*context, *secretKey);
  evaluator = std::make_unique<seal::Evaluator>(*context);
}

template<typename T>
std::vector<T> SimulatorCiphertextFactory::expandVector(const std::vector<T> &values) {
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

std::unique_ptr<seal::Plaintext> SimulatorCiphertextFactory::createPlaintext(int64_t value) {
  std::vector<int64_t> valueAsVec = {value};
  return createPlaintext(valueAsVec);
}

std::unique_ptr<seal::Plaintext> SimulatorCiphertextFactory::createPlaintext(const std::vector<int> &value) {
  std::vector<int64_t> vecInt64(value.begin(), value.end());
  return createPlaintext(vecInt64);
}

std::unique_ptr<seal::Plaintext> SimulatorCiphertextFactory::createPlaintext(const std::vector<int64_t> &value) {
  if (!context || !context->parameters_set()) setupSealContext();
  auto expandedVector = expandVector(value);
  auto ptxt = std::make_unique<seal::Plaintext>();
  encoder->encode(expandedVector, *ptxt);
  return ptxt;
}

const seal::SEALContext &SimulatorCiphertextFactory::getContext() const {
  return *context;
}

const seal::RelinKeys &SimulatorCiphertextFactory::getRelinKeys() const {
  return *relinKeys;
}

void SimulatorCiphertextFactory::decryptCiphertext(AbstractCiphertext &abstractCiphertext,
                                              std::vector<int64_t> &ciphertextData) {
  auto &ctxt = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext);
  seal::Plaintext ptxt = ctxt.getPlaintext(); // this simply returns the stored _plaintext
  encoder->decode(ptxt, ciphertextData);
}

double SimulatorCiphertextFactory::getNoise(AbstractCiphertext &abstractCiphertext) {
  auto &ctxt = dynamic_cast<SimulatorCiphertext &>(abstractCiphertext);
  return ctxt.getNoise();
}

seal::Evaluator &SimulatorCiphertextFactory::getEvaluator() const {
  return *evaluator;
}

const seal::GaloisKeys &SimulatorCiphertextFactory::getGaloisKeys() const {
  return *galoisKeys;
}

SimulatorCiphertextFactory::SimulatorCiphertextFactory(unsigned int numElementsPerCiphertextSlot)
    : ciphertextSlotSize(numElementsPerCiphertextSlot) {}

unsigned int SimulatorCiphertextFactory::getCiphertextSlotSize() const {
  return ciphertextSlotSize;
}

std::string SimulatorCiphertextFactory::getString(AbstractCiphertext &abstractCiphertext) {
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

std::unique_ptr<AbstractCiphertext> SimulatorCiphertextFactory::createCiphertext(std::unique_ptr<AbstractValue> &&abstractValue) {
  if (auto castedCleartext = dynamic_cast<Cleartext<int> *>(abstractValue.get())) {
    // extract data and from std::vector<int> to std::vector<int64_t>
    auto castedCleartextData = castedCleartext->getData();
    std::vector<int64_t> data(castedCleartextData.begin(), castedCleartextData.end());
    // avoid duplicate code -> delegate creation to other constructor
    return createCiphertext(data);
  } else {
    throw std::runtime_error("Cannot create ciphertext from any other than a Cleartext<int> as used ciphertext factory "
                             "(SimulatorCiphertextFactory) uses BFV that only supports integers.");
  }
}

#endif // ifdef HAVE_SEAL_BFV