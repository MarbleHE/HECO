#include "ast_opt/utilities/Operator.h"
#include "ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#include "ast_opt/utilities/PlaintextNorm.h"


SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertextFactory &simulatorFactory)
    : AbstractNoiseMeasuringCiphertext(
    simulatorFactory) {}

SimulatorCiphertext::SimulatorCiphertext(const SimulatorCiphertext &other)  // copy constructor
    : AbstractNoiseMeasuringCiphertext(other.factory) {
  _ciphertext = other._ciphertext;
  _noise = other._noise;
  _noise_budget = other._noise_budget;
  ciphertext_size_ = other.ciphertext_size_;
}

SimulatorCiphertext &SimulatorCiphertext::operator=(const SimulatorCiphertext &other) {  // copy assignment
  return *this = SimulatorCiphertext(other);
}

SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertext &&other) noexcept  // move constructor
    : AbstractNoiseMeasuringCiphertext(other.factory), _ciphertext(std::move(other._ciphertext)) {}

SimulatorCiphertext &SimulatorCiphertext::operator=(SimulatorCiphertext &&other) noexcept {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;
  _ciphertext = other._ciphertext;
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

const seal::Ciphertext &SimulatorCiphertext::getCiphertext() const {
  return _ciphertext;
}

seal::Ciphertext &SimulatorCiphertext::getCiphertext() {
  return _ciphertext;
}

const seal::Plaintext &SimulatorCiphertext::getPlaintext() const {
  return _plaintext;
}

seal::Plaintext &SimulatorCiphertext::getPlaintext() {
  return _plaintext;
}

double &SimulatorCiphertext::getNoise() {
  return _noise;
}

void SimulatorCiphertext::relinearize() {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();

  int destination_size = 2;
  // Determine number of relinearize_one_step calls which would be needed
  uint64_t relinearize_one_step_calls = static_cast<uint64_t>(this->ciphertext_size_ - destination_size);
  if (relinearize_one_step_calls == 0) // nothing to be done
  {
    return; //do nothing
  }
  // Noise is ~ old + 2 * min(B, 6*sigma) * t * n * (ell+1) * w * relinearize_one_step_calls
  double noise_standard_deviation = 3.2; // this is the standard value for the noise standard deviation (see SEAL: hestdparams.h)
  double noise_max_deviation = noise_standard_deviation * 6; // this is also a standard value (see SEAL: globals.h)
  int64_t poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  int64_t plain_modulus = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();



  // First t
  double result_noise = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();

  // multiply by w






  // ciphertext size is now back to 2
  this->ciphertext_size_ = 2;

  throw std::runtime_error("Not implemented yet.");
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiply(AbstractCiphertext &operand) {
  // clone current ctext
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // cast operand
  std::unique_ptr<SimulatorCiphertext> operand_ctxt = std::unique_ptr<SimulatorCiphertext>(&cast(operand));
  uint64_t
      poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
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
  double result_noise = operand_ctxt->_noise*sqrt_factor_1
      + this->_noise*sqrt_factor_2
      + sqrt_factor_total;
  result_noise *= leading_factor; // this is the resulting invariant noise
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  // update noise and noise budget of result ctxt with the new value
  r->_noise = result_noise;
  r->noiseBits();
  // calculate new ciphertext size (sum of ciphertext sizes), note multiplication increases ciphertext size.
  // this becomes relevant for relinearisation heuristics
  r->ciphertext_size_ += operand_ctxt->ciphertext_size_;
  // return ptr to the current ciphertext cast to Abstract ciphertext
  return r;
}

void SimulatorCiphertext::multiplyInplace(AbstractCiphertext &operand) {
  // clone current ctext
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // cast operand
  std::unique_ptr<SimulatorCiphertext> operand_ctxt = std::unique_ptr<SimulatorCiphertext>(&cast(operand));
  uint64_t
      poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
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
  double result_noise = operand_ctxt->_noise*sqrt_factor_1
      + this->_noise * sqrt_factor_2
      + sqrt_factor_total;
  result_noise *= leading_factor; // this is the resulting invariant noise
  // update noise and noise budget of result ctxt with the new value
  this->_noise = result_noise;
  this->noiseBits();
  // calculate new ciphertext size (sum of ciphertext sizes), note multiplication increases ciphertext size.
  // this becomes relevant for relinearisation heuristics
  this->ciphertext_size_ += operand_ctxt->ciphertext_size_;
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiplyPlain(ICleartext &operand) {
  // get plaintext from operand
  auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand);
  std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
  // note: ciphertext size does not increase
  int64_t old_noise = new_ctxt->_noise;
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  // calc noise after ptxt ctxt addition
  double result_noise = old_noise * plain_max_coeff_count * plain_max_abs_value;
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  // update noise and noise budget of result ctxt with the new value
  r->_noise = result_noise;
  r->noiseBits();
  // return ptr to the current ciphertext cast to Abstract ciphertext
  return r;
}
void SimulatorCiphertext::multiplyPlainInplace(ICleartext &operand) {
  // get plaintext from operand
  auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand);
  std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
  // note: ciphertext size does not increase
  int64_t old_noise = new_ctxt->_noise;
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  // clac noise after ptxt ctxt addition
  double result_noise = old_noise * plain_max_coeff_count * plain_max_abs_value;
  // update noise and noise budget of result ctxt with the new value
  this->_noise = result_noise;
  this->noiseBits();
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::add(AbstractCiphertext &operand) {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  std::unique_ptr<SimulatorCiphertext> operand_ctxt = std::unique_ptr<SimulatorCiphertext>(&cast(operand));
  // after addition, the noise is the sum of old noise and noise of ctext that is added
  double result_noise = new_ctxt->_noise + operand_ctxt->_noise;
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  // update noise and noise budget of result ctxt with the new value
  r->_noise = result_noise;
  r->noiseBits();
  // return ptr to the current ciphertext cast to Abstract ciphertext
  return r;
}

void SimulatorCiphertext::addInplace(AbstractCiphertext &operand) {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  std::unique_ptr<SimulatorCiphertext> operand_ctxt = std::unique_ptr<SimulatorCiphertext>(&cast(operand));
  // after addition, the noise is the sum of old noise and noise of ctext that is added
  double result_noise = new_ctxt->_noise + operand_ctxt->_noise;
  // update noise and noise budget of current ctxt with the new value (for this SimulatorCiphertext)
  this->_noise = result_noise;
  this->noiseBits();
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::addPlain(ICleartext &operand) {
  // get plaintext from operand
  auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand);
  std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
  int64_t old_noise = new_ctxt->_noise;
  int64_t rtq = new_ctxt->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  // calc noise after ptxt ctxt addition
  double result_noise = old_noise + rtq * plain_max_coeff_count * plain_max_abs_value;
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  // update noise and noise budget of result ctxt with the new value
  r->_noise = result_noise;
  r->noiseBits();
  // return ptr to the current ciphertext cast to Abstract ciphertext
  return r;
}
void SimulatorCiphertext::addPlainInplace(ICleartext &operand) {
  // get plaintext from operand
  auto cleartextInt = dynamic_cast<Cleartext<int> *>(&operand);
  std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
  int64_t old_noise = new_ctxt->_noise;
  int64_t rtq = new_ctxt->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  // calc noise after ptxt ctxt addition
  double result_noise = old_noise + rtq * plain_max_coeff_count * plain_max_abs_value;
  // update noise and noise budget of result ctxt with the new value
  this->_noise = result_noise;
  this->noiseBits();
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::subtract(AbstractCiphertext &operand) {
  // this is the same as SimulatorCiphertext::add
  return SimulatorCiphertext::add(operand);
}
void SimulatorCiphertext::subtractInplace(AbstractCiphertext &operand) {
  // this is the same as SimulatorCiphertext::addInPlace
  SimulatorCiphertext::addInplace(operand);
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::subtractPlain(ICleartext &operand) {
  // this is the same as SimulatorCiphertext::addPlain
  return SimulatorCiphertext::addPlain(operand);
}
void SimulatorCiphertext::subtractPlainInplace(ICleartext &operand) {
  // this is the same as SimulatorCiphertext::addPlainInPlace
  SimulatorCiphertext::addPlainInplace(operand);
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::rotateRows(int steps) {
  return std::unique_ptr<AbstractCiphertext>();
}
void SimulatorCiphertext::rotateRowsInplace(int steps) {

}

void SimulatorCiphertext::noiseBits() {
  this->_noise_budget = -log2(2 * this->_noise);
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::clone() {
  return clone_impl();
}

std::unique_ptr<SimulatorCiphertext> SimulatorCiphertext::clone_impl() {
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

// so far this needs as input a corresponding plaintext (that we get the "fresh" encryption from)
void SimulatorCiphertext::createFresh(std::unique_ptr<seal::Plaintext> &plaintext) {
  // set _plaintext to plaintext (needed for correct "decryption")
  _plaintext = *plaintext;
  // clone
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();

  // Noise is ~ r_t(q)*plain_max_abs_value * plain_max_coeff_count + 7 * min(B, 6*sigma)*t*n
  int64_t rtq = new_ctxt->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  double noise_standard_deviation = 3.2; // this is the standard value for the noise standard deviation (see SEAL: hestdparams.h)
  double noise_max_deviation = noise_standard_deviation * 6; // this is also a standard value (see SEAL: globals.h)
  // calculate encryption noise
  double result_noise = rtq * plain_max_abs_value * plain_max_coeff_count + 7 * noise_max_deviation *
      new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value() *
      new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  // set noise of the current object to initial noise
  this->_noise = result_noise;
  this->noiseBits();
  // freshly encrypted ciphertext has size 2
  this->ciphertext_size_ = 2;
}

#endif
