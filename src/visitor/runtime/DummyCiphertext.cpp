#include "ast_opt/utilities/Operator.h"
#include "ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/visitor/runtime/DummyCiphertext.h"
#include "ast_opt/visitor/runtime/DummyCiphertextFactory.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"

DummyCiphertext::DummyCiphertext(const std::reference_wrapper<const AbstractCiphertextFactory> dummyFactory)
    : AbstractCiphertext(
    dummyFactory) {}

DummyCiphertext::DummyCiphertext(const DummyCiphertext &other)  // copy constructor
    : AbstractCiphertext(other.factory) {
  _data = other._data;
}

DummyCiphertext &DummyCiphertext::operator=(DummyCiphertext &&other) {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;
  // check if factory is the same, otherwise this is invalid
  if (&factory.get()!=&(other.factory.get())) {
    throw std::runtime_error("Cannot move Ciphertext from factory A into Ciphertext created by Factory B.");
  }
  _data = std::move(other._data);
  return *this;
}

DummyCiphertext &cast_dummy(AbstractCiphertext &abstractCiphertext) {
  if (auto dummyCtxt = dynamic_cast<DummyCiphertext *>(&abstractCiphertext)) {
    return *dummyCtxt;
  } else {
    throw std::runtime_error("Cast of AbstractCiphertext to DummyCiphertext failed!");
  }
}

const DummyCiphertext &cast_dummy(const AbstractCiphertext &abstractCiphertext) {
  if (auto dummyCtxt = dynamic_cast<const DummyCiphertext *>(&abstractCiphertext)) {
    return *dummyCtxt;
  } else {
    throw std::runtime_error("Cast of AbstractCiphertext to DummyCiphertext failed!");
  }
}

// return datavector
std::vector<int64_t> DummyCiphertext::getData() {
  return this->_data;
}

// initialise the dummy ciphertext (i.e set _data variable)
void DummyCiphertext::createFresh(const std::vector<int64_t> &data) {
  this->_data = data;
}

std::unique_ptr<AbstractCiphertext> DummyCiphertext::multiply(const AbstractCiphertext &operand) const {
  DummyCiphertext operand_ctxt = cast_dummy(operand);
  // sizes of data vectors must match
  std::vector<int64_t> result;
  if (operand_ctxt.getData().size() !=this->_data.size()) {
    throw std::runtime_error("Sizes of data vectors do not match");
  }
  else {
    for (int i = 0; i < _data.size(); i++) {
      result.push_back(this->_data[i] * operand_ctxt.getData()[i]);
    }
  }
  auto r = std::make_unique<DummyCiphertext>(*this);
  r->_data = result;
  return r;
}

void DummyCiphertext::multiplyInplace(const AbstractCiphertext &operand) {
  DummyCiphertext operand_ctxt = cast_dummy(operand);
  // sizes of data vectors must match
  std::vector<int64_t> result;
  if (operand_ctxt.getData().size() !=this->_data.size()) {
    throw std::runtime_error("Sizes of data vectors do not match");
  }
  else {
    for (int i = 0; i < _data.size(); i++) {
      result.push_back(this->_data[i] * operand_ctxt.getData()[i]);
    }
  }
  this->_data = result;
}


std::unique_ptr<AbstractCiphertext> DummyCiphertext::multiplyPlain(const ICleartext &operand) const {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    // sizes of data vectors must match
    std::vector<int64_t> result;
    if (cleartextInt->getData().size() !=this->_data.size()) {
      throw std::runtime_error("Sizes of data vectors do not match");
    }
    else {
      for (int i = 0; i < _data.size(); i++) {
        result.push_back(this->_data[i] * cleartextInt->getData()[i]);
      }
    }
    auto r = std::make_unique<DummyCiphertext>(*this);
    r->_data = result;
    return r;
  }
  else {
    throw std::runtime_error("Multiply(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

void DummyCiphertext::multiplyPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    // sizes of data vectors must match
    std::vector<int64_t> result;
    if (cleartextInt->getData().size() !=this->_data.size()) {
      throw std::runtime_error("Sizes of data vectors do not match");
    }
    else {
      for (int i = 0; i < _data.size(); i++) {
        result.push_back(this->_data[i] * cleartextInt->getData()[i]);
      }
    }
    this->_data = result;
  }
  else {
    throw std::runtime_error("Multiply(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> DummyCiphertext::add(const AbstractCiphertext &operand) const {
  DummyCiphertext operand_ctxt = cast_dummy(operand);
  // sizes of data vectors must match
  std::vector<int64_t> result;
  if (operand_ctxt.getData().size() !=this->_data.size()) {
    throw std::runtime_error("Sizes of data vectors do not match");
  }
  else {
    for (int i = 0; i < _data.size(); i++) {
      result.push_back(this->_data[i] + operand_ctxt.getData()[i]);
    }
  }
  auto r = std::make_unique<DummyCiphertext>(*this);
  r->_data = result;
  return r;
}

void DummyCiphertext::addInplace(const AbstractCiphertext &operand) {
  DummyCiphertext operand_ctxt = cast_dummy(operand);
  // sizes of data vectors must match
  std::vector<int64_t> result;
  if (operand_ctxt.getData().size() !=this->_data.size()) {
    throw std::runtime_error("Sizes of data vectors do not match");
  }
  else {
    for (int i = 0; i < _data.size(); i++) {
      result.push_back(this->_data[i] + operand_ctxt.getData()[i]);
    }
  }
  this->_data = result;
}


std::unique_ptr<AbstractCiphertext> DummyCiphertext::addPlain(const ICleartext &operand) const {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    // sizes of data vectors must match
    std::vector<int64_t> result;
    if (cleartextInt->getData().size() !=this->_data.size()) {
      throw std::runtime_error("Sizes of data vectors do not match");
    }
    else {
      for (int i = 0; i < _data.size(); i++) {
        result.push_back(this->_data[i] + cleartextInt->getData()[i]);
      }
    }
    auto r = std::make_unique<DummyCiphertext>(*this);
    r->_data = result;
    return r;
  }
  else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

void DummyCiphertext::addPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    // sizes of data vectors must match
    std::vector<int64_t> result;
    if (cleartextInt->getData().size() !=this->_data.size()) {
      throw std::runtime_error("Sizes of data vectors do not match");
    }
    else {
      for (int i = 0; i < _data.size(); i++) {
        result.push_back(this->_data[i] + cleartextInt->getData()[i]);
      }
    }
    this->_data = result;
  }
  else {
    throw std::runtime_error("Multiply(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> DummyCiphertext::subtract(const AbstractCiphertext &operand) const {
  DummyCiphertext operand_ctxt = cast_dummy(operand);
  // sizes of data vectors must match
  std::vector<int64_t> result;
  if (operand_ctxt.getData().size() !=this->_data.size()) {
    throw std::runtime_error("Sizes of data vectors do not match");
  }
  else {
    for (int i = 0; i < _data.size(); i++) {
      result.push_back(this->_data[i] - operand_ctxt.getData()[i]);
    }
  }
  auto r = std::make_unique<DummyCiphertext>(*this);
  r->_data = result;
  return r;
}
void DummyCiphertext::subtractInplace(const AbstractCiphertext &operand) {
  DummyCiphertext operand_ctxt = cast_dummy(operand);
  // sizes of data vectors must match
  std::vector<int64_t> result;
  if (operand_ctxt.getData().size() !=this->_data.size()) {
    throw std::runtime_error("Sizes of data vectors do not match");
  }
  else {
    for (int i = 0; i < _data.size(); i++) {
      result.push_back(this->_data[i] - operand_ctxt.getData()[i]);
    }
  }
  this->_data = result;
}
std::unique_ptr<AbstractCiphertext> DummyCiphertext::subtractPlain(const ICleartext &operand) const {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    // sizes of data vectors must match
    std::vector<int64_t> result;
    if (cleartextInt->getData().size() !=this->_data.size()) {
      throw std::runtime_error("Sizes of data vectors do not match");
    }
    else {
      for (int i = 0; i < _data.size(); i++) {
        result.push_back(this->_data[i] - cleartextInt->getData()[i]);
      }
    }
    auto r = std::make_unique<DummyCiphertext>(*this);
    r->_data = result;
    return r;
  }
  else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}
void DummyCiphertext::subtractPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    // sizes of data vectors must match
    std::vector<int64_t> result;
    if (cleartextInt->getData().size() !=this->_data.size()) {
      throw std::runtime_error("Sizes of data vectors do not match");
    }
    else {
      for (int i = 0; i < _data.size(); i++) {
        result.push_back(this->_data[i] - cleartextInt->getData()[i]);
      }
    }
    this->_data = result;
  }
  else {
    throw std::runtime_error("Multiply(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> DummyCiphertext::rotateRows(int steps) const {
  throw std::runtime_error("Not yet implemented.");
}
void DummyCiphertext::rotateRowsInplace(int steps) {
  throw std::runtime_error("Not yet implemented.");
}

std::unique_ptr<AbstractCiphertext> DummyCiphertext::clone() const {
  return clone_impl();
}

std::unique_ptr<DummyCiphertext> DummyCiphertext::clone_impl() const {
  return std::make_unique<DummyCiphertext>(*this);
}

void DummyCiphertext::add_inplace(const AbstractValue &other) {
  if (auto
      otherAsDummyCiphertext = dynamic_cast<const DummyCiphertext *>(&other)) {  // ctxt-ctxt operation
    addInplace(*otherAsDummyCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    addPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation ADD only supported for (AbstractCiphertext,AbstractCiphertext) "
                             "and (DummyCiphertext, ICleartext).");
  }
}

void DummyCiphertext::subtract_inplace(const AbstractValue &other) {
  if (auto otherAsDummyCiphertext = dynamic_cast<const DummyCiphertext *>(&other)) {  // ctxt-ctxt operation
    subtractInplace(*otherAsDummyCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    subtractPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation SUBTRACT only supported for (DummyCiphertext,DummyCiphertext) "
                             "and (DummyCiphertext, ICleartext).");
  }
}

void DummyCiphertext::multiply_inplace(const AbstractValue &other) {
  if (auto otherAsDummyCiphertext = dynamic_cast<const DummyCiphertext *>(&other)) {  // ctxt-ctxt operation
    multiplyInplace(*otherAsDummyCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    multiplyPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation MULTIPLY only supported for (DummyCiphertext,DummyCiphertext) "
                             "and (DummyCiphertext, ICleartext).");
  }
}

void DummyCiphertext::divide_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation divide_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::modulo_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation modulo_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::logicalAnd_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalAnd_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::logicalOr_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalOr_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::logicalLess_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalLess_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::logicalLessEqual_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalLessEqual_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::logicalGreater_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalGreater_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::logicalGreaterEqual_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalGreaterEqual_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::logicalEqual_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalEqual_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::logicalNotEqual_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalNotEqual_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::bitwiseAnd_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseAnd_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::bitwiseXor_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseXor_inplace not supported for (DummyCiphertext, ANY).");
}

void DummyCiphertext::bitwiseOr_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseOr_inplace not supported for (DummyCiphertext, ANY).");
}

const DummyCiphertextFactory &DummyCiphertext::getFactory() const {
  if (auto sealFactory = dynamic_cast<const DummyCiphertextFactory *>(&factory.get())) {
    return *sealFactory;
  } else {
    throw std::runtime_error("Cast of AbstractFactory to DummyCiphertextFactory failed. DummyCiphertext is probably invalid.");
  }
}

void DummyCiphertext::logicalNot_inplace() {
  throw std::runtime_error("Operation logicalNot_inplace not supported for (DummyCiphertext, ANY). "
                           "For an arithmetic negation, multiply_inplace by (-1) instead.");
}

void DummyCiphertext::bitwiseNot_inplace() {
  throw std::runtime_error("Operation bitwiseNot_inplace not supported for (DummyCiphertext, ANY). "
                           "For an arithmetic negation, multiply_inplace by (-1) instead.");
}

