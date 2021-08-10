#include "ast_opt/utilities/Operator.h"
#include "ast_opt/runtime/Cleartext.h"
#include "ast_opt/runtime/SealCiphertext.h"
#include "ast_opt/runtime/SealCiphertextFactory.h"
#include "ast_opt/runtime/AbstractCiphertext.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>

SealCiphertext::SealCiphertext(const std::reference_wrapper<const SealCiphertextFactory> sealFactory) :
AbstractCiphertext( (const std::reference_wrapper<const AbstractCiphertextFactory>) sealFactory){}

SealCiphertext::SealCiphertext(const SealCiphertext &other)  // copy constructor
    : AbstractCiphertext(other.factory) {
  ciphertext = seal::Ciphertext(other.ciphertext);
}

SealCiphertext &SealCiphertext::operator=(const SealCiphertext &other) {  // copy assignment
  return *this = SealCiphertext(other);
}

SealCiphertext::SealCiphertext(SealCiphertext &&other) noexcept  // move constructor
    : AbstractCiphertext(other.factory), ciphertext(std::move(other.ciphertext)) {}

SealCiphertext &SealCiphertext::operator=(SealCiphertext &&other) {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;
  // check if factory is the same, otherwise this is invalid
  if (&factory.get()!=&(other.factory.get())) {
    throw std::runtime_error("Cannot move Ciphertext from factory A into Ciphertext created by Factory B.");
  }
  ciphertext = std::move(other.ciphertext);
  return *this;
}

SealCiphertext &cast(AbstractCiphertext &abstractCiphertext) {
  if (auto sealCtxt = dynamic_cast<SealCiphertext *>(&abstractCiphertext)) {
    return *sealCtxt;
  } else {
    throw std::runtime_error("Cast of AbstractCiphertext to SealCiphertext failed!");
  }
}

const SealCiphertext &cast(const AbstractCiphertext &abstractCiphertext) {
  if (auto sealCtxt = dynamic_cast<const SealCiphertext *>(&abstractCiphertext)) {
    return *sealCtxt;
  } else {
    throw std::runtime_error("Cast of AbstractCiphertext to SealCiphertext failed!");
  }
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::rotateRows(int steps) const {
  auto resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
  getFactory().getEvaluator()
      .rotate_rows(ciphertext, steps, getFactory().getGaloisKeys(), resultCiphertext->ciphertext);
  return resultCiphertext;
}

void SealCiphertext::rotateRowsInplace(int steps) {
  getFactory().getEvaluator().rotate_rows_inplace(ciphertext, steps, getFactory().getGaloisKeys());
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::modSwitch(int steps) {
  for (int i=0; i < steps; i++) {
    getFactory().getEvaluator().mod_switch_to_next_inplace(ciphertext);
  }
}

const seal::Ciphertext &SealCiphertext::getCiphertext() const {
  return ciphertext;
}

seal::Ciphertext &SealCiphertext::getCiphertext() {
  return ciphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::clone() const {
  return clone_impl();
}

std::unique_ptr<SealCiphertext> SealCiphertext::clone_impl() const {
  //TODO: check
  return std::make_unique<SealCiphertext>(*this);
}

int SealCiphertext::noiseBits() const {
  seal::Decryptor decryptor(this->getFactory().getContext(), this->getFactory().getSecretKey());
  return decryptor.invariant_noise_budget(this->getCiphertext());
}


// =======================================
// == CTXT-CTXT operations with returned result
// =======================================

std::unique_ptr<AbstractCiphertext> SealCiphertext::add(const AbstractCiphertext &operand) const {
  auto resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
  getFactory().getEvaluator().add(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::subtract(const AbstractCiphertext &operand) const {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
  getFactory().getEvaluator().sub(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::multiply(const AbstractCiphertext &operand) const {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
  getFactory().getEvaluator().multiply(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  getFactory().getEvaluator().relinearize_inplace(resultCiphertext->ciphertext, getFactory().getRelinKeys());
  return resultCiphertext;
}

// =======================================
// == CTXT-CTXT in-place operations
// =======================================

void SealCiphertext::addInplace(const AbstractCiphertext &operand) {
  getFactory().getEvaluator().add_inplace(ciphertext, cast(operand).ciphertext);
}

void SealCiphertext::subtractInplace(const AbstractCiphertext &operand) {
  getFactory().getEvaluator().sub_inplace(ciphertext, cast(operand).ciphertext);
}

void SealCiphertext::multiplyInplace(const AbstractCiphertext &operand) {
  getFactory().getEvaluator().multiply_inplace(ciphertext, cast(operand).ciphertext);
  getFactory().getEvaluator().relinearize_inplace(ciphertext, getFactory().getRelinKeys());
}

// =======================================
// == CTXT-PLAIN operations with returned result
// =======================================

std::unique_ptr<AbstractCiphertext> SealCiphertext::addPlain(const ICleartext &operand) const {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
    getFactory().getEvaluator().add_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
    return resultCiphertext;
  } else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::subtractPlain(const ICleartext &operand) const {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
    getFactory().getEvaluator().sub_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
    return resultCiphertext;
  } else {
    throw std::runtime_error("SUB(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::multiplyPlain(const ICleartext &operand) const {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
    if (cleartextInt->allEqual(-1)) {
      getFactory().getEvaluator().negate(ciphertext, resultCiphertext->ciphertext);
    } else {
      getFactory().getEvaluator().multiply_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
      getFactory().getEvaluator().relinearize_inplace(resultCiphertext->ciphertext, getFactory().getRelinKeys());
    }
    return resultCiphertext;
  } else {
    throw std::runtime_error("MULTIPLY(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

// =======================================
// == CTXT-PLAIN in-place operations
// =======================================

void SealCiphertext::addPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    getFactory().getEvaluator().add_plain_inplace(ciphertext, *plaintext);
  } else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

void SealCiphertext::subtractPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    getFactory().getEvaluator().sub_plain_inplace(ciphertext, *plaintext);
  } else {
    throw std::runtime_error("SUBTRACT(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

void SealCiphertext::multiplyPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    if (cleartextInt->allEqual(-1)) {
      getFactory().getEvaluator().negate_inplace(ciphertext);
    } else {
      std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
      getFactory().getEvaluator().multiply_plain_inplace(ciphertext, *plaintext);
      getFactory().getEvaluator().relinearize_inplace(ciphertext, getFactory().getRelinKeys());
    }
  } else {
    throw std::runtime_error("MULTIPLY(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

// =======================================
// == Overriden methods from AbstractCiphertext interface
// =======================================

void SealCiphertext::add_inplace(const AbstractValue &other) {
  if (auto otherAsSealCiphertext = dynamic_cast<const SealCiphertext *>(&other)) {  // ctxt-ctxt operation
    addInplace(*otherAsSealCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    addPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation ADD only supported for (AbstractCiphertext,AbstractCiphertext) "
                             "and (SealCiphertext, ICleartext).");
  }
}

void SealCiphertext::subtract_inplace(const AbstractValue &other) {
  if (auto otherAsSealCiphertext = dynamic_cast<const SealCiphertext *>(&other)) {  // ctxt-ctxt operation
    subtractInplace(*otherAsSealCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    subtractPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation SUBTRACT only supported for (SealCiphertext,SealCiphertext) "
                             "and (SealCiphertext, ICleartext).");
  }
}

void SealCiphertext::multiply_inplace(const AbstractValue &other) {
  if (auto otherAsSealCiphertext = dynamic_cast<const SealCiphertext *>(&other)) {  // ctxt-ctxt operation
    multiplyInplace(*otherAsSealCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    multiplyPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation MULTIPLY only supported for (SealCiphertext,SealCiphertext) "
                             "and (SealCiphertext, ICleartext).");
  }
}

void SealCiphertext::divide_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation divide_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::modulo_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation modulo_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::logicalAnd_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalAnd_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::logicalOr_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalOr_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::logicalLess_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalLess_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::logicalLessEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalLessEqual_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::logicalGreater_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalGreater_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::logicalGreaterEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalGreaterEqual_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::logicalEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalEqual_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::logicalNotEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalNotEqual_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::bitwiseAnd_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation bitwiseAnd_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::bitwiseXor_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation bitwiseXor_inplace not supported for (SealCiphertext, ANY).");
}

void SealCiphertext::bitwiseOr_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation bitwiseOr_inplace not supported for (SealCiphertext, ANY).");
}

const SealCiphertextFactory &SealCiphertext::getFactory() const {
  if (auto sealFactory = dynamic_cast<const SealCiphertextFactory *>(&factory.get())) {
    return *sealFactory;
  } else {
    throw std::runtime_error("Cast of AbstractFactory to SealFactory failed. SealCiphertext is probably invalid.");
  }
}

void SealCiphertext::logicalNot_inplace() {
  throw std::runtime_error("Operation logicalNot_inplace not supported for (SealCiphertext, ANY). "
                           "For an arithmetic negation, multiply_inplace by (-1) instead.");
}

void SealCiphertext::bitwiseNot_inplace() {
  throw std::runtime_error("Operation bitwiseNot_inplace not supported for (SealCiphertext, ANY). "
                           "For an arithmetic negation, multiply_inplace by (-1) instead.");
}

#endif
