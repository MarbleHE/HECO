#include "ast_opt/utilities/Operator.h"
#include "ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/visitor/runtime/SealCiphertext.h"
#include "ast_opt/visitor/runtime/SealCiphertextFactory.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"

#ifdef HAVE_SEAL_BFV

SealCiphertext::SealCiphertext(SealCiphertextFactory &sealFactory) : AbstractCiphertext(sealFactory) {}

SealCiphertext::SealCiphertext(const SealCiphertext &other)  // copy constructor
    : AbstractCiphertext(other.factory) {
  ciphertext = seal::Ciphertext(other.ciphertext);
}

SealCiphertext &SealCiphertext::operator=(const SealCiphertext &other) {  // copy assignment
  return *this = SealCiphertext(other);
}

SealCiphertext::SealCiphertext(SealCiphertext &&other) noexcept  // move constructor
    : AbstractCiphertext(other.factory), ciphertext(std::move(other.ciphertext)) {}

SealCiphertext &SealCiphertext::operator=(SealCiphertext &&other) noexcept {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;
  ciphertext = other.ciphertext;
  factory = std::move(other.factory);
  return *this;
}

SealCiphertext &cast(AbstractCiphertext &abstractCiphertext) {
  if (auto sealCtxt = dynamic_cast<SealCiphertext *>(&abstractCiphertext)) {
    return *sealCtxt;
  } else {
    throw std::runtime_error("Cast of AbstractCiphertext to SealCiphertext failed!");
  }
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::rotateRows(int steps) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
  getFactory().getEvaluator()
      .rotate_rows(ciphertext, steps, getFactory().getGaloisKeys(), resultCiphertext->ciphertext);
  return resultCiphertext;
}

void SealCiphertext::rotateRowsInplace(int steps) {
  getFactory().getEvaluator().rotate_rows_inplace(ciphertext, steps, getFactory().getGaloisKeys());
}

const seal::Ciphertext &SealCiphertext::getCiphertext() const {
  return ciphertext;
}

seal::Ciphertext &SealCiphertext::getCiphertext() {
  return ciphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::clone() {
  // call the copy constructor to create a clone of this ciphertext
  return std::make_unique<SealCiphertext>(*this);
}

// =======================================
// == CTXT-CTXT operations with returned result
// =======================================

std::unique_ptr<AbstractCiphertext> SealCiphertext::add(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
  getFactory().getEvaluator().add(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::subtract(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
  getFactory().getEvaluator().sub(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::multiply(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
  getFactory().getEvaluator().multiply(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  getFactory().getEvaluator().relinearize_inplace(resultCiphertext->ciphertext, getFactory().getRelinKeys());
  return resultCiphertext;
}

// =======================================
// == CTXT-CTXT in-place operations
// =======================================

void SealCiphertext::addInplace(AbstractCiphertext &operand) {
  getFactory().getEvaluator().add_inplace(ciphertext, cast(operand).ciphertext);
}

void SealCiphertext::subtractInplace(AbstractCiphertext &operand) {
  getFactory().getEvaluator().sub_inplace(ciphertext, cast(operand).ciphertext);
}

void SealCiphertext::multiplyInplace(AbstractCiphertext &operand) {
  getFactory().getEvaluator().multiply_inplace(ciphertext, cast(operand).ciphertext);
  getFactory().getEvaluator().relinearize_inplace(ciphertext, getFactory().getRelinKeys());
}

// =======================================
// == CTXT-PLAIN operations with returned result
// =======================================

std::unique_ptr<AbstractCiphertext> SealCiphertext::addPlain(ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
    getFactory().getEvaluator().add_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
    return resultCiphertext;
  } else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::subtractPlain(ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
    getFactory().getEvaluator().sub_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
    return resultCiphertext;
  } else {
    throw std::runtime_error("SUB(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::multiplyPlain(ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(getFactory());
    getFactory().getEvaluator().multiply_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
    getFactory().getEvaluator().relinearize_inplace(resultCiphertext->ciphertext, getFactory().getRelinKeys());
    return resultCiphertext;
  } else {
    throw std::runtime_error("MULTIPLY(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

// =======================================
// == CTXT-PLAIN in-place operations
// =======================================

void SealCiphertext::addPlainInplace(ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    getFactory().getEvaluator().add_plain_inplace(ciphertext, *plaintext);
  } else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

void SealCiphertext::subtractPlainInplace(ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    getFactory().getEvaluator().sub_plain_inplace(ciphertext, *plaintext);
  } else {
    throw std::runtime_error("SUBTRACT(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

void SealCiphertext::multiplyPlainInplace(ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    getFactory().getEvaluator().multiply_plain_inplace(ciphertext, *plaintext);
    getFactory().getEvaluator().relinearize_inplace(ciphertext, getFactory().getRelinKeys());
  } else {
    throw std::runtime_error("MULTIPLY(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

// =======================================
// == Overriden methods from AbstractCiphertext interface
// =======================================

void SealCiphertext::add(AbstractValue &other) {
  if (auto otherAsSealCiphertext = dynamic_cast<SealCiphertext *>(&other)) {  // ctxt-ctxt operation
    addInplace(*otherAsSealCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<ICleartext *>(&other)) {  // ctxt-ptxt operation
    addPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation ADD only supported for (AbstractCiphertext,AbstractCiphertext) "
                             "and (AbstractCiphertext, ICleartext).");
  }
}

void SealCiphertext::subtract(AbstractValue &other) {
  if (auto otherAsSealCiphertext = dynamic_cast<SealCiphertext *>(&other)) {  // ctxt-ctxt operation
    subtractInplace(*otherAsSealCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<ICleartext *>(&other)) {  // ctxt-ptxt operation
    subtractPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation SUBTRACT only supported for (AbstractCiphertext,AbstractCiphertext) "
                             "and (AbstractCiphertext, ICleartext).");
  }
}

void SealCiphertext::multiply(AbstractValue &other) {
  if (auto otherAsSealCiphertext = dynamic_cast<SealCiphertext *>(&other)) {  // ctxt-ctxt operation
    multiplyInplace(*otherAsSealCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<ICleartext *>(&other)) {  // ctxt-ptxt operation
    multiplyPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation MULTIPLY only supported for (AbstractCiphertext,AbstractCiphertext) "
                             "and (AbstractCiphertext, ICleartext).");
  }
}

void SealCiphertext::divide(AbstractValue &other) {
  throw std::runtime_error("Operation divide not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::modulo(AbstractValue &other) {
  throw std::runtime_error("Operation modulo not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalAnd(AbstractValue &other) {
  throw std::runtime_error("Operation logicalAnd not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalOr(AbstractValue &other) {
  throw std::runtime_error("Operation logicalOr not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalLess(AbstractValue &other) {
  throw std::runtime_error("Operation logicalLess not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalLessEqual(AbstractValue &other) {
  throw std::runtime_error("Operation logicalLessEqual not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalGreater(AbstractValue &other) {
  throw std::runtime_error("Operation logicalGreater not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalGreaterEqual(AbstractValue &other) {
  throw std::runtime_error("Operation logicalGreaterEqual not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalEqual(AbstractValue &other) {
  throw std::runtime_error("Operation logicalEqual not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalNotEqual(AbstractValue &other) {
  throw std::runtime_error("Operation logicalNotEqual not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalBitwiseAnd(AbstractValue &other) {
  throw std::runtime_error("Operation logicalBitwiseAnd not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalBitwiseXor(AbstractValue &other) {
  throw std::runtime_error("Operation logicalBitwiseXor not supported for (AbstractCiphertext, *).");
}

void SealCiphertext::logicalBitwiseOr(AbstractValue &other) {
  throw std::runtime_error("Operation logicalBitwiseOr not supported for (AbstractCiphertext, *).");
}

SealCiphertextFactory &SealCiphertext::getFactory() {
  // removes const qualifier from const getFactory (https://stackoverflow.com/a/856839/3017719)
  return const_cast<SealCiphertextFactory &>(const_cast<const SealCiphertext *>(this)->getFactory());
}

const SealCiphertextFactory &SealCiphertext::getFactory() const {
  if (auto sealFactory = dynamic_cast<SealCiphertextFactory *>(&factory)) {
    return *sealFactory;
  } else {
    throw std::runtime_error("Cast of AbstractFactory to SealFactory failed. SealCiphertext is probably invalid.");
  }
}

#endif
