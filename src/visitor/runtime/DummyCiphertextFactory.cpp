#include "ast_opt/visitor/runtime/DummyCiphertext.h"
#include "ast_opt/visitor/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/runtime/Cleartext.h"

#include <memory>

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

DummyCiphertextFactory::DummyCiphertextFactory(const DummyCiphertextFactory &other) {}  // copy constructor

DummyCiphertextFactory::DummyCiphertextFactory(DummyCiphertextFactory &&other) noexcept {} // move constructor

DummyCiphertextFactory &DummyCiphertextFactory::operator=(const DummyCiphertextFactory &other) {  // copy assignment
  return *this = DummyCiphertextFactory(other);
}

DummyCiphertextFactory &DummyCiphertextFactory::operator=(DummyCiphertextFactory &&other) noexcept {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;
  return *this;
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
  // cast to DummyCtxt and getData()
  auto dummyCtxt = dynamic_cast<DummyCiphertext *>(&abstractCiphertext);
  ciphertextData = dummyCtxt->getData();
}


DummyCiphertextFactory::DummyCiphertextFactory(unsigned int numElementsPerCiphertextSlot) {}

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