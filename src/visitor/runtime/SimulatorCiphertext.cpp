
#include "ast_opt/utilities/Operator.h"
#include "ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#include "ast_opt/utilities/seal2.3.0_util/biguint.h"


// Constructor for a simulated ciphertext that is created given seal params, size and noise
SimulatorCiphertext::SimulatorCiphertext(AbstractCiphertextFactory &acf,
                                         const seal::EncryptionParameters &parms,
                                         int ciphertext_size,
                                         int noise_budget) : AbstractNoiseMeasuringCiphertext(acf) : parms_(parms),
  ciphertext_size_(ciphertext_size)
{
  // Compute product coeff modulus
  coeff_modulus_ = 1;
  for (auto mod : parms_.coeff_modulus())
  {
    coeff_modulus_ *= mod.value();
  }
  coeff_modulus_bit_count_ = coeff_modulus_.significant_bit_count();

  // Verify parameters
  if (noise_budget < 0 || noise_budget >= coeff_modulus_bit_count_ - 1)
  {
    throw std::invalid_argument("noise_budget is not in the valid range");
  }
  if (ciphertext_size < 2)
  {
    throw std::invalid_argument("ciphertext_size must be at least 2");
  }

  // Set the noise (scaled by coeff_modulus) to have given noise budget
  // noise_ = 2^(coeff_sig_bit_count - noise_budget - 1) - 1
  int noise_sig_bit_count = coeff_modulus_bit_count_ - noise_budget - 1;
  noise_.resize(coeff_modulus_bit_count_);
  noise_[0] = 1;
  left_shift_uint(noise_.pointer(), noise_sig_bit_count, noise_.uint64_count(), noise_.pointer());
  decrement_uint(noise_.pointer(), noise_.uint64_count(), noise_.pointer());
}

SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertextFactory &simulatorFactory)
    : AbstractNoiseMeasuringCiphertext(
    simulatorFactory) {}

SimulatorCiphertext::SimulatorCiphertext(const SimulatorCiphertext &other)  // copy constructor
    : AbstractNoiseMeasuringCiphertext(other.factory) {
  ciphertext = seal::Ciphertext(other.ciphertext);
}

SimulatorCiphertext &SimulatorCiphertext::operator=(const SimulatorCiphertext &other) {  // copy assignment
  return *this = SimulatorCiphertext(other);
}

SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertext &&other) noexcept  // move constructor
    : AbstractNoiseMeasuringCiphertext(other.factory), ciphertext(std::move(other.ciphertext)) {}

SimulatorCiphertext &SimulatorCiphertext::operator=(SimulatorCiphertext &&other) noexcept {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;
  ciphertext = other.ciphertext;
  factory = std::move(other.factory);
  return *this;
}

SimulatorCiphertext &cast(AbstractCiphertext &abstractCiphertext) {
  if (auto sealCtxt = dynamic_cast<SimulatorCiphertext *>(&abstractCiphertext)) {
    return *sealCtxt;
  } else {
    throw std::runtime_error("Cast of AbstractCiphertext to SealCiphertext failed!");
  }
}

std::unique_ptr<AbstractCiphertext> createFresh(const seal::EncryptionParameters &parms, int plain_max_coeff_count,
                                                uint64_t plain_max_abs_value) {

  // Compute product coeff modulus
  int64_t coeff_modulus = 1;
  for (auto mod : parms.coeff_modulus()) {
    coeff_modulus *= mod.value();
  }
  //seal::util::ge
  // int coeff_bit_count = seal::util::get_significant_bit_count(coeff_modulus);
  // int coeff_uint64_count = seal::util::divide_round_up(coeff_bit_count, seal::util::bits_per_uint64);
  // int poly_modulus_degree = parms.poly_modulus_degree() - 1; // check this if it is actually the same!! (here and everywhere else)

  // Widen plain_modulus_ and noise_
  //TODO continue
  // Noise is ~ r_t(q)*plain_max_abs_value*plain_max_coeff_count + 7 * min(B, 6*sigma)*t*n
  //TODO
  throw std::runtime_error("Not yet implemented.");
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::relinearize() {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // Noise is ~ old + 2 * min(B, 6*sigma) * t * n * (ell+1) * w * relinearize_one_step_calls
  int64_t old_noise = new_ctxt->noise_;
  //TODO: Continue new_ctxt->getFactory().getContext()

  throw std::runtime_error("Not implemented yet.");
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiply(AbstractCiphertext &operand) {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // cast operand
  std::unique_ptr<SimulatorCiphertext> operand_ctxt = std::unique_ptr<SimulatorCiphertext>(&cast(operand));

  uint64_t
      poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree() - 1;

  // determine size of new ciphertext
  int result_ciphertext_size = new_ctxt->ciphertext_size_ + operand_ctxt->ciphertext_size_ - 1;

  uint64_t plain_modulus = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  // compute product coeff modulus
  uint64_t coeff_modulus = 1;

  for (auto mod : new_ctxt->getFactory().getContext().first_context_data()->parms().coeff_modulus()) {
    coeff_modulus *= mod.value();
  }

  // Compute Noise (this is as in Seal v2.3.0):
  // Noise is ~ t * sqrt(3n) * [ (12n)^(j1/2)*noise2 + (12n)^(j2/2)*noise1 + (12n)^((j1+j2)/2) ]
  // First compute sqrt(12n) (rounding up) and the powers needed
  uint64_t sqrt_factor_base = static_cast<uint64_t>(ceil(sqrt(static_cast<double>(12*poly_modulus_degree))));
  uint64_t sqrt_factor_1 = seal::util::exponentiate_uint(sqrt_factor_base,
                                                         new_ctxt->getFactory().getCiphertextSlotSize() - 1);
  uint64_t sqrt_factor_2 =
      seal::util::exponentiate_uint(sqrt_factor_base, operand_ctxt->getFactory().getCiphertextSlotSize() - 1);
  uint64_t sqrt_factor_total = seal::util::exponentiate_uint(sqrt_factor_base,
                                                             new_ctxt->getFactory().getCiphertextSlotSize() - 1
                                                                 + operand_ctxt->getFactory().getCiphertextSlotSize()
                                                                 - 1);
  // Compute also t * sqrt(3n)
  uint64_t leading_sqrt_factor = static_cast<uint64_t>(ceil(sqrt(static_cast<double>(3*poly_modulus_degree))));
  uint64_t leading_factor = plain_modulus*leading_sqrt_factor;

  int64_t result_noise = operand_ctxt->_noise*sqrt_factor_1
      + this->_noise*sqrt_factor_2
      + sqrt_factor_total;
  result_noise *= leading_factor; // this is the resulting invariant noise

  // update the noise of the ciphertext
  this->noise_ = result_noise;
  this->noise_budget = noiseBits();

  return std::make_unique<SimulatorCiphertext>(this->getFactory(), this->parms_, result_ciphertext_size, result_noise);
}

void SimulatorCiphertext::multiplyInplace(AbstractCiphertext &operand) {

}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiplyPlain(ICleartext &operand) {
  return std::unique_ptr<AbstractCiphertext>();
}
void SimulatorCiphertext::multiplyPlainInplace(ICleartext &operand) {

}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::add(AbstractCiphertext &operand) {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  std::unique_ptr<SimulatorCiphertext> operand_ctxt = std::unique_ptr<SimulatorCiphertext>(&cast(operand));
  uint64_t result_noise = new_ctxt->_noise + operand_ctxt->_noise;

  // auto new_noiseBits = this->noiseBits() + operand.noiseBits();
  // std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // new_ctxt->noise_budget = new_noiseBits;
  return std::unique_ptr<AbstractCiphertext>();
}

void SimulatorCiphertext::addInplace(AbstractCiphertext &operand) {
  if (auto cast_operand = dynamic_cast<AbstractNoiseMeasuringCiphertext* >(&operand)) {
    //TODO: Do addition now that we know we have the noise interface available
  } else {
    // throw
  }
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::addPlain(ICleartext &operand) {
  return std::unique_ptr<AbstractCiphertext>();
}
void SimulatorCiphertext::addPlainInplace(ICleartext &operand) {

}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::subtract(AbstractCiphertext &operand) {
  return std::unique_ptr<AbstractCiphertext>();
}
void SimulatorCiphertext::subtractInplace(AbstractCiphertext &operand) {

}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::subtractPlain(ICleartext &operand) {
  return std::unique_ptr<AbstractCiphertext>();
}
void SimulatorCiphertext::subtractPlainInplace(ICleartext &operand) {

}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::rotateRows(int steps) {
  return std::unique_ptr<AbstractCiphertext>();
}
void SimulatorCiphertext::rotateRowsInplace(int steps) {

}

double SimulatorCiphertext::noiseBits() {
  noise_budget = -log2(2*this->noise_);
  return noise_budget;
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::clone() {
  return clone_impl();
}

std::unique_ptr<SimulatorCiphertext> SimulatorCiphertext::clone_impl() {
  //TODO: check
  return std::unique_ptr<SimulatorCiphertext>(this);
}

void SimulatorCiphertext::add(AbstractValue &other) {
  if (auto otherAsSimulatorCiphertext = dynamic_cast<SimulatorCiphertext *>(&other)) {  // ctxt-ctxt operation
    addInplace(*otherAsSimulatorCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<ICleartext *>(&other)) {  // ctxt-ptxt operation
    addPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation ADD only supported for (AbstractCiphertext,AbstractCiphertext) "
                             "and (SimulatorCiphertext, ICleartext).");
  }
}

void SimulatorCiphertext::subtract(AbstractValue &other) {
  if (auto otherAsSimulatorCiphertext = dynamic_cast<SimulatorCiphertext *>(&other)) {  // ctxt-ctxt operation
    subtractInplace(*otherAsSimulatorCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<ICleartext *>(&other)) {  // ctxt-ptxt operation
    subtractPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation SUBTRACT only supported for (SimulatorCiphertext,SimulatorCiphertext) "
                             "and (SimulatorCiphertext, ICleartext).");
  }
}

void SimulatorCiphertext::multiply(AbstractValue &other) {
  if (auto otherAsSimulatorCiphertext = dynamic_cast<SimulatorCiphertext *>(&other)) {  // ctxt-ctxt operation
    multiplyInplace(*otherAsSimulatorCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<ICleartext *>(&other)) {  // ctxt-ptxt operation
    multiplyPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation MULTIPLY only supported for (SimulatorCiphertext,SimulatorCiphertext) "
                             "and (SimulatorCiphertext, ICleartext).");
  }
}

void SimulatorCiphertext::divide(AbstractValue &other) {
  throw std::runtime_error("Operation divide not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::modulo(AbstractValue &other) {
  throw std::runtime_error("Operation modulo not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalAnd(AbstractValue &other) {
  throw std::runtime_error("Operation logicalAnd not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalOr(AbstractValue &other) {
  throw std::runtime_error("Operation logicalOr not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalLess(AbstractValue &other) {
  throw std::runtime_error("Operation logicalLess not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalLessEqual(AbstractValue &other) {
  throw std::runtime_error("Operation logicalLessEqual not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalGreater(AbstractValue &other) {
  throw std::runtime_error("Operation logicalGreater not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalGreaterEqual(AbstractValue &other) {
  throw std::runtime_error("Operation logicalGreaterEqual not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalEqual(AbstractValue &other) {
  throw std::runtime_error("Operation logicalEqual not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalNotEqual(AbstractValue &other) {
  throw std::runtime_error("Operation logicalNotEqual not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseAnd(AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseAnd not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseXor(AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseXor not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseOr(AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseOr not supported for (SimulatorCiphertext, ANY).");
}

SimulatorCiphertextFactory &SimulatorCiphertext::getFactory() {
  // removes const qualifier from const getFactory (https://stackoverflow.com/a/856839/3017719)
  return const_cast<SimulatorCiphertextFactory &>(const_cast<const SimulatorCiphertext *>(this)->getFactory());
}

const SimulatorCiphertextFactory &SimulatorCiphertext::getFactory() const {
  if (auto sealFactory = dynamic_cast<SimulatorCiphertextFactory *>(&factory)) {
    return *sealFactory;
  } else {
    throw std::runtime_error("Cast of AbstractFactory to SealFactory failed. SimulatorCiphertext is probably invalid.");
  }
}

void SimulatorCiphertext::logicalNot() {
  throw std::runtime_error("Operation logicalNot not supported for (SimulatorCiphertext, ANY). "
                           "For an arithmetic negation, multiply by (-1) instead.");
}

void SimulatorCiphertext::bitwiseNot() {
  throw std::runtime_error("Operation bitwiseNot not supported for (SimulatorCiphertext, ANY). "
                           "For an arithmetic negation, multiply by (-1) instead.");
}
int64_t SimulatorCiphertext::initialNoise() {
  return 0;
}

#endif
