#include <ast_opt/utilities/Operator.h>
#include "ast_opt/visitor/Runtime/SealCiphertext.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/visitor/Runtime/SealCiphertextFactory.h"

#ifdef HAVE_SEAL_BFV

SealCiphertext::SealCiphertext(SealCiphertextFactory &sealFactory) : factory(sealFactory) {}

SealCiphertext::SealCiphertext(const SealCiphertext &other)  // copy constructor
    : factory(other.factory) {
  ciphertext = seal::Ciphertext(other.ciphertext);
}

SealCiphertext &SealCiphertext::operator=(const SealCiphertext &other) {  // copy assignment
  return *this = SealCiphertext(other);
}

SealCiphertext::SealCiphertext(SealCiphertext &&other) noexcept  // move constructor
    : factory(other.factory), ciphertext(std::move(other.ciphertext)) {}

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
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  factory.getEvaluator().rotate_rows(ciphertext, steps, factory.getGaloisKeys(), resultCiphertext->ciphertext);
  return resultCiphertext;
}

void SealCiphertext::rotateRowsInplace(int steps) {
  factory.getEvaluator().rotate_rows_inplace(ciphertext, steps, factory.getGaloisKeys());
}

const seal::Ciphertext &SealCiphertext::getCiphertext() const {
  return ciphertext;
}

seal::Ciphertext &SealCiphertext::getCiphertext() {
  return ciphertext;
}

std::vector<int64_t> castExpressionList(ExpressionList &expressionList) {
  std::vector<int64_t> result;
  auto exprs = expressionList.getExpressions();
  std::for_each(exprs.begin(), exprs.end(), [&result](const std::reference_wrapper<AbstractExpression> &expression) {
    if (auto expressionAsLiteralInt = dynamic_cast<const LiteralInt *>(&expression.get())) {
      result.push_back(expressionAsLiteralInt->getValue());
    } else {
      throw std::runtime_error(
          "Expected expression of type LiteralInt in ExpressionList, "
          "found instead " + std::string(typeid(expression).name()) + ".");
    }
  });
  return result;
}

// =======================================
// == CTXT-CTXT operations with returned result
// =======================================

std::unique_ptr<AbstractCiphertext> SealCiphertext::add(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  factory.getEvaluator().add(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::subtract(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  factory.getEvaluator().sub(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::multiply(AbstractCiphertext &operand) {
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  factory.getEvaluator().multiply(ciphertext, cast(operand).ciphertext, resultCiphertext->ciphertext);
  factory.getEvaluator().relinearize_inplace(resultCiphertext->ciphertext, factory.getRelinKeys());
  return resultCiphertext;
}

// =======================================
// == CTXT-CTXT in-place operations
// =======================================

void SealCiphertext::addInplace(AbstractCiphertext &operand) {
  factory.getEvaluator().add_inplace(ciphertext, cast(operand).ciphertext);
}

void SealCiphertext::subtractInplace(AbstractCiphertext &operand) {
  factory.getEvaluator().sub_inplace(ciphertext, cast(operand).ciphertext);
}

void SealCiphertext::multiplyInplace(AbstractCiphertext &operand) {
  factory.getEvaluator().multiply_inplace(ciphertext, cast(operand).ciphertext);
  factory.getEvaluator().relinearize_inplace(ciphertext, factory.getRelinKeys());
}

// =======================================
// == CTXT-PLAIN operations with returned result
// =======================================

std::unique_ptr<AbstractCiphertext> SealCiphertext::addPlain(ExpressionList &operand) {
  auto castedOperand = castExpressionList(operand);
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  auto plaintext = factory.createPlaintext(castedOperand);
  factory.getEvaluator().add_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::subtractPlain(ExpressionList &operand) {
  auto castedOperand = castExpressionList(operand);
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  auto plaintext = factory.createPlaintext(castedOperand);
  factory.getEvaluator().sub_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
  return resultCiphertext;
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::multiplyPlain(ExpressionList &operand) {
  auto castedOperand = castExpressionList(operand);
  std::unique_ptr<SealCiphertext> resultCiphertext = std::make_unique<SealCiphertext>(factory);
  auto plaintext = factory.createPlaintext(castedOperand);
  factory.getEvaluator().multiply_plain(ciphertext, *plaintext, resultCiphertext->ciphertext);
  factory.getEvaluator().relinearize_inplace(resultCiphertext->ciphertext, factory.getRelinKeys());
  return resultCiphertext;
}

// =======================================
// == CTXT-PLAIN in-place operations
// =======================================

void SealCiphertext::addPlainInplace(ExpressionList &operand) {
  auto castedOperand = castExpressionList(operand);
  auto plaintext = factory.createPlaintext(castedOperand);
  factory.getEvaluator().add_plain_inplace(ciphertext, *plaintext);
}

void SealCiphertext::subtractPlainInplace(ExpressionList &operand) {
  auto castedOperand = castExpressionList(operand);
  auto plaintext = factory.createPlaintext(castedOperand);
  factory.getEvaluator().sub_plain_inplace(ciphertext, *plaintext);
}

void SealCiphertext::multiplyPlainInplace(ExpressionList &operand) {
  auto castedOperand = castExpressionList(operand);
  auto plaintext = factory.createPlaintext(castedOperand);
  factory.getEvaluator().multiply_plain_inplace(ciphertext, *plaintext);
  factory.getEvaluator().relinearize_inplace(ciphertext, factory.getRelinKeys());
}

std::unique_ptr<AbstractCiphertext> SealCiphertext::clone() {
  // call the copy constructor to create a clone of this ciphertext
  return std::make_unique<SealCiphertext>(*this);
}

#endif
