#include "ast_opt/utilities/Operator.h"
#include "ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#include "ast_opt/utilities/PlaintextNorm.h"

SimulatorCiphertext::SimulatorCiphertext(const std::reference_wrapper<const AbstractCiphertextFactory> simulatorFactory)
    : AbstractNoiseMeasuringCiphertext(
    simulatorFactory) {}

SimulatorCiphertext::SimulatorCiphertext(const SimulatorCiphertext &other)  // copy constructor
    : AbstractNoiseMeasuringCiphertext(other.factory) {
  _ciphertext = other._ciphertext;
  _plaintext = other._plaintext;
  _noise = other._noise;
  _noise_budget = other._noise_budget;
  ciphertext_size_ = other.ciphertext_size_;
}

SimulatorCiphertext &SimulatorCiphertext::operator=(const SimulatorCiphertext &other) {  // copy assignment
  return *this = SimulatorCiphertext(other);
}

SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertext &&other) noexcept  // move constructor
    : AbstractNoiseMeasuringCiphertext(other.factory), _ciphertext(std::move(other._ciphertext)) {}

SimulatorCiphertext &SimulatorCiphertext::operator=(SimulatorCiphertext &&other) {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;
  // check if factory is the same, otherwise this is invalid
  if (&factory.get()!=&(other.factory.get())) {
    throw std::runtime_error("Cannot move Ciphertext from factory A into Ciphertext created by Factory B.");
  }
  _ciphertext = std::move(other._ciphertext);
  return *this;
}

SimulatorCiphertext &cast_1(AbstractCiphertext &abstractCiphertext) {
  if (auto sealCtxt = dynamic_cast<SimulatorCiphertext *>(&abstractCiphertext)) {
    return *sealCtxt;
  } else {
    throw std::runtime_error("Cast of AbstractCiphertext to SealCiphertext failed!");
  }
}

const SimulatorCiphertext &cast_1(const AbstractCiphertext &abstractCiphertext) {
  if (auto sealCtxt = dynamic_cast<const SimulatorCiphertext *>(&abstractCiphertext)) {
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

double SimulatorCiphertext::getNoise() const {
  return _noise;
}

void SimulatorCiphertext::relinearize() {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();

  int destination_size = 2;
  // Determine number of relinearize_one_step calls which would be needed
  int64_t relinearize_one_step_calls = static_cast<int64_t>(this->ciphertext_size_ - destination_size);
  if (relinearize_one_step_calls==0) // nothing to be done
  {
    return; //do nothing
  }
  // Noise is ~ old + 2 * min(B, 6*sigma) * t * n * (ell+1) * w * relinearize_one_step_calls
  double noise_standard_deviation =
      3.2; // this is the standard value for the noise standard deviation (see SEAL: hestdparams.h)
  double noise_max_deviation = noise_standard_deviation*6; // this is also a standard value (see SEAL: globals.h)
  int64_t poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  int64_t plain_modulus = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();



  // First t
  double result_noise = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();

  // multiply_inplace by w






  // ciphertext size is now back to 2
  this->ciphertext_size_ = 2;

  throw std::runtime_error("Not implemented yet.");
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiply(const AbstractCiphertext &operand) const {
  // clone current ctext
  auto new_ctxt = this->clone_impl();
  // cast operand
  auto operand_ctxt = cast_1(operand);
  int64_t
      poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  // determine size of new ciphertext
  int result_ciphertext_size = new_ctxt->ciphertext_size_ + operand_ctxt.ciphertext_size_ - 1;
  int64_t plain_modulus = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  // compute product coeff modulus
  int64_t coeff_modulus = 1;
  for (auto mod : new_ctxt->getFactory().getContext().first_context_data()->parms().coeff_modulus()) {
    coeff_modulus *= mod.value();
  }
  double result_noise;
  /* SEAL 2.3.0 does it this way:
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
  result_noise = operand_ctxt->_noise*sqrt_factor_1
      + this->_noise*sqrt_factor_2
      + sqrt_factor_total;
  result_noise *= leading_factor; // this is the resulting invariant noise */
  /*iliashenko:*/
  result_noise = plain_modulus*sqrt(3*poly_modulus_degree + 2*pow(poly_modulus_degree, 2))
      *(new_ctxt->getNoise() + operand_ctxt.getNoise()) + 3*new_ctxt->getNoise()*operand_ctxt.getNoise() +
      plain_modulus/coeff_modulus*sqrt(3*poly_modulus_degree + 2*pow(poly_modulus_degree, 2) +
          4*pow(poly_modulus_degree, 3)/3);
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  // update noise and noise budget of result ctxt with the new value
  r->_noise = result_noise;
  r->noiseBits();
  // calculate new ciphertext size (sum of ciphertext sizes), note multiplication increases ciphertext size.
  // this becomes relevant for relinearisation heuristics
  r->ciphertext_size_ += operand_ctxt.ciphertext_size_;
  // return ptr to the current ciphertext cast to Abstract ciphertext
  return r;
}

void SimulatorCiphertext::multiplyInplace(const AbstractCiphertext &operand) {
  // clone current ctext
  auto new_ctxt = this->clone_impl();
  // cast operand
  auto operand_ctxt = cast_1(operand);
  uint64_t
      poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  // determine size of new ciphertext
  int result_ciphertext_size = new_ctxt->ciphertext_size_ + operand_ctxt.ciphertext_size_ - 1;
  uint64_t plain_modulus = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  // compute product coeff modulus
  uint64_t coeff_modulus = 1;
  for (auto mod : new_ctxt->getFactory().getContext().first_context_data()->parms().coeff_modulus()) {
    coeff_modulus *= mod.value();
  }
  double result_noise;
  /*
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
   */
  // iliashenko:
  result_noise = plain_modulus*sqrt(3*poly_modulus_degree + 2*pow(poly_modulus_degree, 2))
      *(new_ctxt->getNoise() + operand_ctxt.getNoise()) + 3*new_ctxt->getNoise()*operand_ctxt.getNoise() +
      plain_modulus/coeff_modulus*sqrt(3*poly_modulus_degree + 2*pow(poly_modulus_degree, 2) +
          4*pow(poly_modulus_degree, 3)/3);
  this->_noise = result_noise;
  this->noiseBits();
  // calculate new ciphertext size (sum of ciphertext sizes), note multiplication increases ciphertext size.
  // this becomes relevant for relinearisation heuristics
  this->ciphertext_size_ += operand_ctxt.ciphertext_size_;
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiplyPlain(const ICleartext &operand) const {
  // get plaintext from operand
  auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand);
  std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
  // note: ciphertext size does not increase
  int64_t old_noise = new_ctxt->_noise;
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  // calc noise after ptxt ctxt addition
  double result_noise = old_noise*plain_max_coeff_count*plain_max_abs_value;
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  // update noise and noise budget of result ctxt with the new value
  r->_noise = result_noise;
  r->noiseBits();
  // return ptr to the current ciphertext cast to Abstract ciphertext
  return r;
}
void SimulatorCiphertext::multiplyPlainInplace(const ICleartext &operand) {
  // get plaintext from operand
  auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand);
  auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
  auto new_ctxt = this->clone_impl();
  // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
  // note: ciphertext size does not increase
  int64_t old_noise = new_ctxt->_noise;
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  // calc noise after ptxt ctxt mult
  double result_noise = old_noise*plain_max_coeff_count*plain_max_abs_value;
  // update noise and noise budget of result ctxt with the new value
  this->_noise = result_noise;
  this->noiseBits();
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::add(const AbstractCiphertext &operand) const {
  SimulatorCiphertext operand_ctxt = cast_1(operand);
  // after addition, the noise is the sum of old noise and noise of ctxt that is added
  double result_noise = this->_noise + operand_ctxt._noise;
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  // update noise and noise budget of result ctxt with the new value
  r->_noise = result_noise;
  r->noiseBits();
  // return ptr to the current ciphertext cast to Abstract ciphertext
  return r;
}

void SimulatorCiphertext::addInplace(const AbstractCiphertext &operand) {
  auto new_ctxt = this->clone_impl();
  auto operand_ctxt = cast_1(operand);
  // after addition, the noise is the sum of old noise and noise of ctext that is added
  double result_noise = new_ctxt->_noise + operand_ctxt._noise;
  // update noise and noise budget of current ctxt with the new value (for this SimulatorCiphertext)
  this->_noise = result_noise;
  this->noiseBits();
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::addPlain(const ICleartext &operand) const {
  // get plaintext from operand
  auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand);
  auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
  auto new_ctxt = this->clone_impl();
  // noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
  int64_t old_noise = new_ctxt->_noise;
  int64_t rtq = new_ctxt->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  // calc noise after ptxt ctxt addition
  double result_noise = old_noise + rtq*plain_max_coeff_count*plain_max_abs_value;
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  // update noise and noise budget of result ctxt with the new value
  r->_noise = result_noise;
  r->noiseBits();
  // return ptr to the current ciphertext cast to Abstract ciphertext
  return r;
}
void SimulatorCiphertext::addPlainInplace(const ICleartext &operand) {
  // get plaintext from operand
  auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand);
  auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
  auto new_ctxt = this->clone_impl();
  // noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
  int64_t old_noise = new_ctxt->_noise;
  int64_t rtq = new_ctxt->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  // calc noise after ptxt ctxt addition
  double result_noise = old_noise + rtq*plain_max_coeff_count*plain_max_abs_value;
  // update noise and noise budget of result ctxt with the new value
  this->_noise = result_noise;
  this->noiseBits();
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::subtract(const AbstractCiphertext &operand) const {
  // this is the same as SimulatorCiphertext::add
  return SimulatorCiphertext::add(operand);
}
void SimulatorCiphertext::subtractInplace(const AbstractCiphertext &operand) {
  // this is the same as SimulatorCiphertext::addInPlace
  SimulatorCiphertext::addInplace(operand);
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::subtractPlain(const ICleartext &operand) const {
  // this is the same as SimulatorCiphertext::addPlain
  return SimulatorCiphertext::addPlain(operand);
}
void SimulatorCiphertext::subtractPlainInplace(const ICleartext &operand) {
  // this is the same as SimulatorCiphertext::addPlainInPlace
  SimulatorCiphertext::addPlainInplace(operand);
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::rotateRows(int steps) const {
  throw std::runtime_error("Not yet implemented.");
}
void SimulatorCiphertext::rotateRowsInplace(int steps) {
  throw std::runtime_error("Not yet implemented.");
}

double SimulatorCiphertext::noiseBits() const {
  return -log2(2*round(this->_noise));
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::clone() const {
  return clone_impl();
}

std::unique_ptr<SimulatorCiphertext> SimulatorCiphertext::clone_impl() const {
  return std::make_unique<SimulatorCiphertext>(*this);
}

void SimulatorCiphertext::add_inplace(const AbstractValue &other) {
  if (auto
      otherAsSimulatorCiphertext = dynamic_cast<const SimulatorCiphertext *>(&other)) {  // ctxt-ctxt operation
    addInplace(*otherAsSimulatorCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    addPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation ADD only supported for (AbstractCiphertext,AbstractCiphertext) "
                             "and (SimulatorCiphertext, ICleartext).");
  }
}

void SimulatorCiphertext::subtract_inplace(const AbstractValue &other) {
  if (auto otherAsSimulatorCiphertext = dynamic_cast<const SimulatorCiphertext *>(&other)) {  // ctxt-ctxt operation
    subtractInplace(*otherAsSimulatorCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    subtractPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation SUBTRACT only supported for (SimulatorCiphertext,SimulatorCiphertext) "
                             "and (SimulatorCiphertext, ICleartext).");
  }
}

void SimulatorCiphertext::multiply_inplace(const AbstractValue &other) {
  if (auto otherAsSimulatorCiphertext = dynamic_cast<const SimulatorCiphertext *>(&other)) {  // ctxt-ctxt operation
    multiplyInplace(*otherAsSimulatorCiphertext);
  } else if (auto otherAsCleartext = dynamic_cast<const ICleartext *>(&other)) {  // ctxt-ptxt operation
    multiplyPlainInplace(*otherAsCleartext);
  } else {
    throw std::runtime_error("Operation MULTIPLY only supported for (SimulatorCiphertext,SimulatorCiphertext) "
                             "and (SimulatorCiphertext, ICleartext).");
  }
}

void SimulatorCiphertext::divide_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation divide_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::modulo_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation modulo_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalAnd_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalAnd_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalOr_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalOr_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalLess_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalLess_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalLessEqual_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalLessEqual_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalGreater_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalGreater_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalGreaterEqual_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalGreaterEqual_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalEqual_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalEqual_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalNotEqual_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation logicalNotEqual_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseAnd_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseAnd_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseXor_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseXor_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseOr_inplace(const AbstractValue &other) {
  throw std::runtime_error("Operation bitwiseOr_inplace not supported for (SimulatorCiphertext, ANY).");
}

const SimulatorCiphertextFactory &SimulatorCiphertext::getFactory() const {
  if (auto sealFactory = dynamic_cast<const SimulatorCiphertextFactory *>(&factory.get())) {
    return *sealFactory;
  } else {
    throw std::runtime_error("Cast of AbstractFactory to SealFactory failed. SimulatorCiphertext is probably invalid.");
  }
}

void SimulatorCiphertext::logicalNot_inplace() {
  throw std::runtime_error("Operation logicalNot_inplace not supported for (SimulatorCiphertext, ANY). "
                           "For an arithmetic negation, multiply_inplace by (-1) instead.");
}

void SimulatorCiphertext::bitwiseNot_inplace() {
  throw std::runtime_error("Operation bitwiseNot_inplace not supported for (SimulatorCiphertext, ANY). "
                           "For an arithmetic negation, multiply_inplace by (-1) instead.");
}
int64_t SimulatorCiphertext::initialNoise() {
  return 0;
}

// so far this needs as input a corresponding plaintext (that we get the "fresh" encryption from)
void SimulatorCiphertext::createFresh(std::unique_ptr<seal::Plaintext> &plaintext) {
  // set _plaintext to plaintext (needed for correct decryption)
  _plaintext = *plaintext;
  // clone
  double result_noise;
  /*
  // SEAL 2.3.0: Noise is ~ r_t(q)*plain_max_abs_value * plain_max_coeff_count + 7 * min(B, 6*sigma)*t*n     (from SEAL 2.3.0)
  int64_t rtq = new_ctxt->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
  int64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
  int64_t plain_max_abs_value = plaintext_norm(*plaintext);
  double noise_standard_deviation = 3.2; // this is the standard value for the noise standard deviation (see SEAL: hestdparams.h)
  double noise_max_deviation = noise_standard_deviation * 6; // this is also a standard value (see SEAL: globals.h)
  // calculate encryption noise
  result_noise = rtq * plain_max_abs_value * plain_max_coeff_count + 7 * noise_max_deviation *
      new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value() *
      new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  */
  //iliashenko: noise is t/q * ( n * (t-1) / 2 + 2 * sigma sqrt(12 * n^2 + 9 * n) )
  // compute product coeff modulus
  uint64_t coeff_modulus = 1;
  for (auto mod : this->getFactory().getContext().first_context_data()->parms().coeff_modulus()) {
    coeff_modulus *= mod.value();
  }
  int64_t plain_modulus = this->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  int64_t poly_modulus = this->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  double sigma = 3.2;
  result_noise = plain_modulus/coeff_modulus*(poly_modulus*(coeff_modulus - 1)/2
      + 2*sigma*sqrt(12*pow(poly_modulus, 2) + 9*poly_modulus));
  // set noise of the current object to initial noise
  this->_noise = result_noise;
  this->noiseBits();
  // freshly encrypted ciphertext has size 2
  this->ciphertext_size_ = 2;
}

#endif
