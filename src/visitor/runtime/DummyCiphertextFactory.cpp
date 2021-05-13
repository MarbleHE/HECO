#include "ast_opt/visitor/runtime/DummyCiphertext.h"
#include "ast_opt/visitor/runtime/DummyCiphertext.h"
#include "ast_opt/visitor/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/runtime/Cleartext.h"

#ifdef HAVE_SEAL_BFV
#include <memory>
#include <seal/seal.h>

std::unique_ptr<AbstractCiphertext> DummyCiphertextFactory::createCiphertext(const std::vector<int64_t> &data) const {
  std::unique_ptr<DummyCiphertext>
      ctxt = std::make_unique<DummyCiphertext>(*this); // Constructs a dummy ciphertext given all the data
  ctxt->createFresh(data); // calcs initial noise and sets the variables needed, also stores the plaintext as _plaintext
  return ctxt;
}

std::unique_ptr<AbstractCiphertext> DummyCiphertextFactory::createCiphertext(const std::vector<int> &data) const {
  std::vector<int64_t> ciphertextData(data.begin(), data.end());
  return createCiphertext(ciphertextData);
}

std::unique_ptr<AbstractCiphertext> DummyCiphertextFactory::createCiphertext(int64_t data) const {
  std::vector<int64_t> values = {data};
  return createCiphertext(values);
}

DummyCiphertextFactory::DummyCiphertextFactory(const DummyCiphertextFactory &other) :
    ciphertextSlotSize(other.ciphertextSlotSize) {  // copy constructor
}

DummyCiphertextFactory::DummyCiphertextFactory(DummyCiphertextFactory &&other) noexcept // move constructor
    : ciphertextSlotSize(other.ciphertextSlotSize) {
}

DummyCiphertextFactory &DummyCiphertextFactory::operator=(const DummyCiphertextFactory &other) {  // copy assignment
  return *this = DummyCiphertextFactory(other);
}

DummyCiphertextFactory &DummyCiphertextFactory::operator=(DummyCiphertextFactory &&other) noexcept {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;

  if (ciphertextSlotSize!=other.ciphertextSlotSize) {
    std::cerr << "Move assignment failed as const ciphertextSlotSize differs and cannot be changed!" << std::endl;
    exit(1);
  }
  return *this;
}


template<typename T>
std::vector<T> DummyCiphertextFactory::expandVector(const std::vector<T> &values) const {
  // passing the vector by value to implicitly get a copy somehow didn't work here
  std::vector<T> expandedVector(values.begin(), values.end());
  //TODO: implement
  return expandedVector;
}

std::unique_ptr<seal::Plaintext> DummyCiphertextFactory::createPlaintext(int64_t value) const {
  std::vector<int64_t> valueAsVec = {value};
  return createPlaintext(valueAsVec);
}

std::unique_ptr<seal::Plaintext> DummyCiphertextFactory::createPlaintext(const std::vector<int> &value) const {
  std::vector<int64_t> vecInt64(value.begin(), value.end());
  return createPlaintext(vecInt64);
}

std::unique_ptr<seal::Plaintext> DummyCiphertextFactory::createPlaintext(const std::vector<int64_t> &value) const {
  return nullptr;
}


void DummyCiphertextFactory::decryptCiphertext(AbstractCiphertext &abstractCiphertext,
                                                   std::vector<int64_t> &ciphertextData) const {
  // do nothing
}

DummyCiphertextFactory::DummyCiphertextFactory(unsigned int numElementsPerCiphertextSlot)
    : ciphertextSlotSize(numElementsPerCiphertextSlot) {

}

unsigned int DummyCiphertextFactory::getCiphertextSlotSize() const {
  return ciphertextSlotSize;
}

std::string DummyCiphertextFactory::getString(AbstractCiphertext &abstractCiphertext) const {
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

std::unique_ptr<AbstractCiphertext> DummyCiphertextFactory::createCiphertext(std::unique_ptr<AbstractValue> &&abstractValue) const {
  if (auto castedCleartext = dynamic_cast<Cleartext<int> *>(abstractValue.get())) {
    // extract data and from std::vector<int> to std::vector<int64_t>
    auto castedCleartextData = castedCleartext->getData();
    std::vector<int64_t> data(castedCleartextData.begin(), castedCleartextData.end());
    // avoid duplicate code -> delegate creation to other constructor
    return createCiphertext(data);
  } else {
    throw std::runtime_error("Cannot create ciphertext from any other than a Cleartext<int> as used ciphertext factory "
                             "(DummyCiphertextFactory) uses BFV that only supports integers.");
  }
}

#endif // ifdef HAVE_SEAL_BFV
