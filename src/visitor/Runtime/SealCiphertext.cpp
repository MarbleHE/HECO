#include "ast_opt/visitor/Runtime/SealCiphertext.h"
#include "ast_opt/visitor/Runtime/SealCiphertextFactory.h"

#ifdef HAVE_SEAL_BFV

SealCiphertext::SealCiphertext(SealCiphertextFactory &sealFactory) : factory(sealFactory) {}

SealCiphertext &cast(AbstractCiphertext &abstractCiphertext) {
  if (auto sealCtxt = dynamic_cast<SealCiphertext *>(&abstractCiphertext)) {
    return *sealCtxt;
  } else {
    throw std::runtime_error("Cast of AbstractCiphertext to SealCiphertext failed!");
  }
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::multiply(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  auto &evaluator = factory.getEvaluator();
  evaluator.multiply(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  evaluator.relinearize_inplace(resultCiphertext->ciphertext, *factory.getRelinKeys());
  return resultCiphertext;
}

void SealCiphertext::multiplyInplace(AbstractCiphertext &operand) {
  auto &evaluator = factory.getEvaluator();
  evaluator.multiply_inplace(ciphertext, cast(operand).ciphertext);
  evaluator.relinearize_inplace(ciphertext, *factory.getRelinKeys());
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::add(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  auto &evaluator = factory.getEvaluator();
  evaluator.add(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

void SealCiphertext::addInplace(AbstractCiphertext &operand) {
  auto &evaluator = factory.getEvaluator();
  evaluator.add_inplace(ciphertext, cast(operand).ciphertext);
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::subtract(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  auto &evaluator = factory.getEvaluator();
  evaluator.sub(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

void SealCiphertext::subtractInplace(AbstractCiphertext &operand) {
  auto &evaluator = factory.getEvaluator();
  evaluator.sub_inplace(ciphertext, cast(operand).ciphertext);
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::rotateRows(int steps) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  auto &evaluator = factory.getEvaluator();
  evaluator.rotate_rows(ciphertext, steps, factory.getGaloisKeys(), resultCiphertext->ciphertext);
  return resultCiphertext;
}

void SealCiphertext::rotateRowsInplace(int steps) {
  auto &evaluator = factory.getEvaluator();
  evaluator.rotate_rows_inplace(ciphertext, steps, factory.getGaloisKeys());
}

const seal::Ciphertext &SealCiphertext::getCiphertext() const {
  return ciphertext;
}

seal::Ciphertext &SealCiphertext::getCiphertext() {
  return ciphertext;
}

#endif
