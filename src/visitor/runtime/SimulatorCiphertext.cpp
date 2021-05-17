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
  //_ciphertext = other._ciphertext;
  _plaintext = other._plaintext;
  _noise = other._noise;
  _noise_budget = other._noise_budget;
  ciphertext_size_ = other.ciphertext_size_;
}

SimulatorCiphertext &SimulatorCiphertext::operator=(const SimulatorCiphertext &other) {  // copy assignment
  return *this = SimulatorCiphertext(other);
}

SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertext &&other) noexcept  // move constructor
    : AbstractNoiseMeasuringCiphertext(other.factory) {}//, _ciphertext(std::move(other._ciphertext)) {}

SimulatorCiphertext &SimulatorCiphertext::operator=(SimulatorCiphertext &&other) {  // move assignment
  // Self-assignment detection
  if (&other==this) return *this;
  // check if factory is the same, otherwise this is invalid
  if (&factory.get()!=&(other.factory.get())) {
    throw std::runtime_error("Cannot move Ciphertext from factory A into Ciphertext created by Factory B.");
  }
  _plaintext = std::move(other._plaintext);
  _noise = std::move(other._noise);
  _noise_budget = std::move(other._noise_budget);
  ciphertext_size_ = std::move(other.ciphertext_size_);
  //_ciphertext = std::move(other._ciphertext);
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
    throw std::runtime_error("Cast of AbstractCiphertext to SimulatorCiphertext failed!");
  }
}

const seal::Ciphertext &SimulatorCiphertext::getCiphertext() const {
//TODO: what should i return here
  throw std::runtime_error("Operation not supported for SimulatorCiphertext");
  //return _ciphertext
}

seal::Ciphertext &SimulatorCiphertext::getCiphertext() {
  //TODO: what should i return here
  throw std::runtime_error("Operation not supported for SimulatorCiphertext");
 // return _ciphertext;
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

int64_t SimulatorCiphertext::getNoiseBudget() {
  return _noise_budget;
}

int64_t SimulatorCiphertext::getCoeffModulus() {
  // THIS IS ACTUALLY NOT NEEDED: There is a SEAL fucntion total_coeff_modulus() returning a pointy=er to the product coeff modulus
  uint64_t coeff_modulus = 1;
  for (auto mod : this->getFactory().getContext().first_context_data()->parms().coeff_modulus()) {
    coeff_modulus *= mod.value();
  }
  return coeff_modulus;
}

void SimulatorCiphertext::createFresh(std::unique_ptr<seal::Plaintext> &plaintext) {
  // set _plaintext to plaintext (needed for correct decryption)
  _plaintext = *plaintext;
  uint64_t result_noise = 0;
  uint64_t plain_modulus = this->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  uint64_t poly_modulus = this->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  result_noise = plain_modulus * (poly_modulus*(plain_modulus - 1)/2
      + 2*3.2*sqrt(12*pow(poly_modulus, 2) + 9*poly_modulus));

  this->_noise = result_noise;
  this->_noise_budget = this->noiseBits();
  // freshly encrypted ciphertext has size 2
  this->ciphertext_size_ = 2;
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
  uint64_t poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  uint64_t plain_modulus = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  // First t
  uint64_t result_noise = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  // multiply_inplace by w
  // ciphertext size is now back to 2
  this->ciphertext_size_ = 2;

  throw std::runtime_error("Not implemented yet.");
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiply(const AbstractCiphertext &operand) const {
  // cast operand
  auto operand_ctxt = cast_1(operand);
  uint64_t coeff_modulus = *this->getFactory().getContext().first_context_data()->total_coeff_modulus();
  uint64_t
      poly_modulus_degree = this->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  uint64_t plain_modulus = this->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  uint64_t result_noise;
  /*iliashenko noise heuristics scaled by q */
  result_noise = plain_modulus*sqrt(3*poly_modulus_degree + 2*pow(poly_modulus_degree, 2))
      *(this->getNoise() + operand_ctxt.getNoise()) + 3*this->getNoise()*operand_ctxt.getNoise()/coeff_modulus +
      plain_modulus*sqrt(3*poly_modulus_degree + 2*pow(poly_modulus_degree, 2) +
          4*pow(poly_modulus_degree, 3)/3);
  //copy
  auto r = std::make_unique<SimulatorCiphertext>(*this);

  r->_noise = result_noise;
  r->_noise_budget = noiseBits();
  r->ciphertext_size_ += operand_ctxt.ciphertext_size_; //ciphertext size increased
  return r;
}

void SimulatorCiphertext::multiplyInplace(const AbstractCiphertext &operand) {
  // cast operand
  auto operand_ctxt = cast_1(operand);
  uint64_t coeff_modulus = *this->getFactory().getContext().first_context_data()->total_coeff_modulus();
  uint64_t
      poly_modulus_degree = this->getFactory().getContext().first_context_data()->parms().poly_modulus_degree();
  uint64_t plain_modulus = this->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  uint64_t result_noise;
  // iliashenko (scaled by q)
  result_noise =plain_modulus*sqrt(3*poly_modulus_degree + 2*pow(poly_modulus_degree, 2))
      *(this->getNoise() + operand_ctxt.getNoise()) + 3*this->getNoise()*operand_ctxt.getNoise()/coeff_modulus +
      plain_modulus*sqrt(3*poly_modulus_degree + 2*pow(poly_modulus_degree, 2) +
          4*pow(poly_modulus_degree, 3)/3);
  this->_noise = result_noise;
  this->_noise_budget = noiseBits();
  this->ciphertext_size_ += operand_ctxt.ciphertext_size_;
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiplyPlain(const ICleartext &operand) const {
  // get plaintext from operand
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    uint64_t old_noise = this->_noise;
    uint64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_max_abs_value = plaintext_norm(*plaintext);
    // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
    uint64_t result_noise = old_noise*plain_max_coeff_count*plain_max_abs_value;
    //copy
    auto r = std::make_unique<SimulatorCiphertext>(*this);
    r->_noise = result_noise;
    r->_noise_budget = noiseBits();
    return r;
  } else {
    throw std::runtime_error("Multiply(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

void SimulatorCiphertext::multiplyPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
    uint64_t old_noise = this->_noise;
    uint64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_max_abs_value = plaintext_norm(*plaintext);
    // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
    uint64_t result_noise = old_noise*plain_max_coeff_count*plain_max_abs_value;
    this->_noise = result_noise;
    this->_noise_budget = noiseBits();
  } else {
    throw std::runtime_error("Multiply(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::add(const AbstractCiphertext &operand) const {
  SimulatorCiphertext operand_ctxt = cast_1(operand);
  // noise is noise1 + noise2
  uint64_t result_noise = this->_noise + operand_ctxt._noise;
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  r->_noise = result_noise;
  r->_noise_budget = noiseBits();
  return r;
}

void SimulatorCiphertext::addInplace(const AbstractCiphertext &operand) {
  //auto new_ctxt = this->clone_impl();
  auto operand_ctxt = cast_1(operand);
  // after addition, the noise is the sum of old noise and noise of ctext that is added
  uint64_t result_noise = this->_noise + operand_ctxt._noise;
  // update noise and noise budget of current ctxt with the new value (for this SimulatorCiphertext)
  this->_noise = result_noise;
  this->_noise_budget = noiseBits();
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::addPlain(const ICleartext &operand) const {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
    double old_noise = this->_noise;
    uint64_t rtq = this->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
    uint64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_max_abs_value = plaintext_norm(*plaintext);
    // noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
    uint64_t result_noise = old_noise + rtq*plain_max_coeff_count*plain_max_abs_value;
    auto r = std::make_unique<SimulatorCiphertext>(*this);
    r->_noise = result_noise;
    r->_noise_budget = noiseBits();
    return r;
  } else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}
void SimulatorCiphertext::addPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
    double old_noise = this->_noise;
    uint64_t rtq = this->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
    uint64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_max_abs_value = plaintext_norm(*plaintext);
    // noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
    uint64_t result_noise = old_noise + rtq*plain_max_coeff_count*plain_max_abs_value;
    this->_noise = result_noise;
    this->_noise_budget = noiseBits();
  } else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
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
  auto resultCiphertext = std::make_unique<SimulatorCiphertext>(getFactory());
  return resultCiphertext;
}
void SimulatorCiphertext::rotateRowsInplace(int steps) {
  //NOOP
}
int64_t SimulatorCiphertext::noiseBits() const{
  uint64_t coeff_modulus_significant_bit_count = this->getFactory().getContext().first_context_data()->total_coeff_modulus_bit_count();
  uint64_t noise_log = round(log2(this->_noise));
  return coeff_modulus_significant_bit_count - noise_log - 1;
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

#endif
