#include "ast_opt/utilities/Operator.h"
#include "ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"
//#include <gmp.h> //TODO Fix cmakelists
#include "/usr/local/include/gmp.h"


#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>
#include "ast_opt/utilities/PlaintextNorm.h"

SimulatorCiphertext::SimulatorCiphertext(const std::reference_wrapper<const AbstractCiphertextFactory> simulatorFactory)
    : AbstractNoiseMeasuringCiphertext(
    simulatorFactory) {}

SimulatorCiphertext::SimulatorCiphertext(const SimulatorCiphertext &other)  // copy constructor
    : AbstractNoiseMeasuringCiphertext(other.factory) {
  _plaintext = other._plaintext;
//  mpz_set(_noise, other._noise); // this doesnt work. why?
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
  //_dummy_ctxt = std::move(other._dummy_ctxt);
  _plaintext = std::move(other._plaintext);
  mpz_set(_noise,other._noise); //TODO: check
  // _noise = std::move(other._noise);
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


void SimulatorCiphertext::createFresh(std::unique_ptr<seal::Plaintext> &plaintext) {
  mpz_init(_noise); // initialise noise
  // set _plaintext to plaintext (needed for correct decryption)
  _plaintext = *plaintext;
  //calc initial noise and noise budget
  mpz_t result_noise;
  mpz_init(result_noise);
  mpz_t plain_mod;
  mpz_init(plain_mod);
  mpz_init_set_ui(plain_mod, this->getFactory().getContext().first_context_data()->parms().plain_modulus().value());
  mpz_t poly_mod;
  mpz_init(poly_mod);
  mpz_init_set_ui(poly_mod, this->getFactory().getContext().first_context_data()->parms().poly_modulus_degree());
  // summand_one = n * (t-1) / 2
  mpz_t summand_one;
  mpz_init(summand_one);
  mpz_sub_ui(summand_one, plain_mod, 1);
  mpz_mul(summand_one, summand_one, poly_mod);
  mpz_div_ui(summand_one, summand_one, 2);
  // summand_two = 2 * sigma * sqrt(12 * n ^2 + 9 * n)
  mpz_t summand_two;
  mpz_init(summand_two);
  mpz_pow_ui(summand_two, poly_mod, 2);
  mpz_mul_ui(summand_two, summand_two, 12);
  mpz_t poly_mod_times_nine;
  mpz_init(poly_mod_times_nine);
  mpz_mul_ui(poly_mod_times_nine, poly_mod, 9);
  mpz_add(summand_two, summand_two, poly_mod_times_nine);
  mpz_sqrt(summand_two, summand_two);
  mpz_mul_ui(summand_two, summand_two, long(6.4)); // sigma = 3.2
  mpz_t sum;
  // sum = summand_1 + summand_2
  mpz_init(sum);
  mpz_add(sum, summand_one, summand_two);
  // result_noise = t * sum
  mpz_mul(result_noise, sum, plain_mod);
  mpz_set(this->_noise,result_noise);
  this->_noise_budget = this->noiseBits();
  // freshly encrypted ciphertext has size 2
  this->ciphertext_size_ = 2;
}

void SimulatorCiphertext::relinearize() {
  throw std::runtime_error("Not implemented yet.");
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiply(const AbstractCiphertext &operand) const {
  // cast operand
  auto operand_ctxt = cast_1(operand);
  mpz_t result_noise;
  mpz_init(result_noise);
  mpz_t plain_mod;
  mpz_init(plain_mod);
  mpz_init_set_ui(plain_mod, this->getFactory().getContext().first_context_data()->parms().plain_modulus().value());
  mpz_t poly_mod;
  mpz_init(poly_mod);
  mpz_init_set_ui(poly_mod, this->getFactory().getContext().first_context_data()->parms().poly_modulus_degree());
  mpz_t coeff_mod;
  mpz_init(coeff_mod);
  mpz_init_set_ui(coeff_mod, *this->getFactory().getContext().first_context_data()->total_coeff_modulus());
  // some powers of n and other things
  mpz_t poly_mod_squared;
  mpz_init(poly_mod_squared);
  mpz_pow_ui(poly_mod_squared, poly_mod, 2);
  mpz_t poly_mod_cubed;
  mpz_init(poly_mod_cubed);
  mpz_pow_ui(poly_mod_cubed, poly_mod, 3);
  mpz_t poly_mod_squared_times_two;
  mpz_init(poly_mod_squared_times_two);
  mpz_mul_ui(poly_mod_squared_times_two, poly_mod, 2);
  mpz_t poly_mod_cubed_times_four;
  mpz_init(poly_mod_cubed_times_four);
  mpz_mul_ui(poly_mod_cubed_times_four, poly_mod_cubed, 4);
  mpz_t poly_mod_cubed_times_four_div_three;
  mpz_init(poly_mod_cubed_times_four_div_three);
  mpz_div_ui(poly_mod_cubed_times_four_div_three, poly_mod_cubed_times_four, 3);
  // summand_one = t * sqrt(3n + 2n^2) (v1 + v2)
  mpz_t summand_one;
  mpz_init(summand_one);
  mpz_mul_ui(summand_one, poly_mod, 3);
  mpz_add(summand_one, summand_one, poly_mod_squared_times_two);
  mpz_sqrt(summand_one, summand_one);
  mpz_mul(summand_one, summand_one, plain_mod);
  mpz_t noise_sum;
  mpz_init(noise_sum);
  mpz_add(noise_sum, this->_noise, operand_ctxt._noise);
  mpz_mul(summand_one, summand_one, noise_sum);
  //summand_two = 3 * v1 * v2 / q
  mpz_t summand_two;
  mpz_init(summand_two);
  mpz_mul(summand_two, this->_noise, operand_ctxt._noise);
  mpz_mul_ui(summand_two, summand_two, 3);
  mpz_div(summand_two, summand_two, coeff_mod);
  //summand_three = t * sqrt(3d+2d^2+4d^3/3)
  mpz_t summand_three;
  mpz_init(summand_three);
  mpz_mul_ui(summand_three, poly_mod, 3);
  mpz_add(summand_three, summand_three, poly_mod_squared_times_two);
  mpz_add(summand_three, summand_three, poly_mod_cubed_times_four_div_three);
  mpz_sqrt(summand_three, summand_three);
  mpz_mul(summand_three, summand_three, plain_mod);
  // result_noise = summand_1 * summand_2 + summand_3
  mpz_add(result_noise, summand_one, summand_two);
  mpz_add(result_noise, result_noise, summand_three);
  // copy and assign correct noise and noise budget
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  mpz_set(r->_noise,result_noise);
  r->_noise_budget = r->noiseBits();
  r->ciphertext_size_ += operand_ctxt.ciphertext_size_; //ciphertext size increased
  return r;
}

void SimulatorCiphertext::multiplyInplace(const AbstractCiphertext &operand) {
  // cast operand
  auto operand_ctxt = cast_1(operand);
  mpz_t result_noise;
  mpz_init(result_noise);
  mpz_t plain_mod;
  mpz_init(plain_mod);
  mpz_init_set_ui(plain_mod, this->getFactory().getContext().first_context_data()->parms().plain_modulus().value());
  mpz_t poly_mod;
  mpz_init(poly_mod);
  mpz_init_set_ui(poly_mod, this->getFactory().getContext().first_context_data()->parms().poly_modulus_degree());
  mpz_t coeff_mod;
  mpz_init(coeff_mod);
  mpz_init_set_ui(coeff_mod, *this->getFactory().getContext().first_context_data()->total_coeff_modulus());
  // some powers of n and other things
  mpz_t poly_mod_squared;
  mpz_init(poly_mod_squared);
  mpz_pow_ui(poly_mod_squared, poly_mod, 2);
  mpz_t poly_mod_cubed;
  mpz_init(poly_mod_cubed);
  mpz_pow_ui(poly_mod_cubed, poly_mod, 3);
  mpz_t poly_mod_squared_times_two;
  mpz_init(poly_mod_squared_times_two);
  mpz_mul_ui(poly_mod_squared_times_two, poly_mod, 2);
  mpz_t poly_mod_cubed_times_four;
  mpz_init(poly_mod_cubed_times_four);
  mpz_mul_ui(poly_mod_cubed_times_four, poly_mod_cubed, 4);
  mpz_t poly_mod_cubed_times_four_div_three;
  mpz_init(poly_mod_cubed_times_four_div_three);
  mpz_div_ui(poly_mod_cubed_times_four_div_three, poly_mod_cubed_times_four, 3);
  // summand_one = t * sqrt(3n + 2n^2) (v1 + v2)
  mpz_t summand_one;
  mpz_init(summand_one);
  mpz_mul_ui(summand_one, poly_mod, 3);
  mpz_add(summand_one, summand_one, poly_mod_squared_times_two);
  mpz_sqrt(summand_one, summand_one);
  mpz_mul(summand_one, summand_one, plain_mod);
  mpz_t noise_sum;
  mpz_init(noise_sum);
  mpz_add(noise_sum, this->_noise, operand_ctxt._noise);
  mpz_mul(summand_one, summand_one, noise_sum);
  //summand_two = 3 * v1 * v2 / q
  mpz_t summand_two;
  mpz_init(summand_two);
  mpz_mul(summand_two, this->_noise, operand_ctxt._noise);
  mpz_mul_ui(summand_two, summand_two, 3);
  mpz_div(summand_two, summand_two, coeff_mod);
  //sumand_three = t * sqrt(3d+2d^2+4d^3/3)
  mpz_t summand_three;
  mpz_init(summand_three);
  mpz_mul_ui(summand_three, poly_mod, 3);
  mpz_add(summand_three, summand_three, poly_mod_squared_times_two);
  mpz_add(summand_three, summand_three, poly_mod_cubed_times_four_div_three);
  mpz_sqrt(summand_three, summand_three);
  mpz_mul(summand_three, summand_three, plain_mod);
  // result_noise = summand_1 * summand_2 + summand_3
  mpz_add(result_noise, summand_one, summand_two);
  mpz_add(result_noise, result_noise, summand_three);
  mpz_set(this->_noise,result_noise);
  this->_noise_budget = this->noiseBits();
  this->ciphertext_size_ += operand_ctxt.ciphertext_size_;
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiplyPlain(const ICleartext &operand) const {
  // get plaintext from operand
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    std::unique_ptr<seal::Plaintext> plaintext = getFactory().createPlaintext(cleartextInt->getData());
    uint64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_max_abs_value = plaintext_norm(*plaintext);
    // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
    mpz_t plain_coeff_ct;
    mpz_init(plain_coeff_ct);
    mpz_init_set_ui(plain_coeff_ct, plain_max_coeff_count);
    mpz_t plain_abs;
    mpz_init(plain_abs);
    mpz_init_set_ui(plain_abs, plain_max_abs_value);
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_mul(result_noise, plain_abs, plain_coeff_ct);
    mpz_mul(result_noise, result_noise, this->_noise);
    //copy
    auto r = std::make_unique<SimulatorCiphertext>(*this);
    mpz_set(r->_noise,result_noise);
    r->_noise_budget = r->noiseBits();
    return r;
  } else {
    throw std::runtime_error("Multiply(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

void SimulatorCiphertext::multiplyPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
    uint64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_max_abs_value = plaintext_norm(*plaintext);
    // noise is old_noise * plain_max_coeff_count * plain_max_abs_value (SEAL Manual)
    mpz_t plain_coeff_ct;
    mpz_init(plain_coeff_ct);
    mpz_init_set_ui(plain_coeff_ct, plain_max_coeff_count);
    mpz_t plain_abs;
    mpz_init(plain_abs);
    mpz_init_set_ui(plain_abs, plain_max_abs_value);
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_mul(result_noise, plain_abs, plain_coeff_ct);
    mpz_mul(result_noise, result_noise, this->_noise);
    //copy
    auto r = std::make_unique<SimulatorCiphertext>(*this);
    mpz_set(this->_noise, result_noise);
    this->_noise_budget = noiseBits();
  } else {
    throw std::runtime_error("Multiply(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::add(const AbstractCiphertext &operand) const {
  SimulatorCiphertext operand_ctxt = cast_1(operand);
  // noise is noise1 + noise2
  mpz_t result_noise;
  mpz_init(result_noise);
  mpz_add(result_noise, this->_noise, operand_ctxt._noise);
  auto r = std::make_unique<SimulatorCiphertext>(*this);
  mpz_set(r->_noise, result_noise);
  r->_noise_budget = r->noiseBits();
  return r;
}

void SimulatorCiphertext::addInplace(const AbstractCiphertext &operand) {
  auto operand_ctxt = cast_1(operand);
  // noise is noise1 + noise2
  mpz_t result_noise;
  mpz_init(result_noise);
  mpz_add(result_noise, this->_noise, operand_ctxt._noise);
  mpz_set(this->_noise,result_noise);
  this->_noise_budget = noiseBits();
}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::addPlain(const ICleartext &operand) const {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
  //  double old_noise = this->_noise;
    uint64_t rtq = this->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
    uint64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_max_abs_value = plaintext_norm(*plaintext);
    // noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_t  rt_q;
    mpz_init(rt_q);
    mpz_init_set_ui(rt_q, rtq);
    mpz_t  plain_coeff_ct;
    mpz_init(plain_coeff_ct);
    mpz_init_set_ui(plain_coeff_ct, plain_max_coeff_count);
    mpz_t  plain_abs;
    mpz_init(plain_abs);
    mpz_init_set_ui(plain_abs, plain_max_abs_value);
    mpz_mul(result_noise, rt_q, plain_coeff_ct);
    mpz_mul(result_noise, result_noise, plain_abs);
    mpz_add(result_noise, result_noise, this->_noise);
    //copy
    auto r = std::make_unique<SimulatorCiphertext>(*this);
    mpz_set(r->_noise,result_noise);
    r->_noise_budget = r->noiseBits();
    return r;
  } else {
    throw std::runtime_error("ADD(Ciphertext,Cleartext) requires a Cleartext<int> as BFV supports integers only.");
  }
}
void SimulatorCiphertext::addPlainInplace(const ICleartext &operand) {
  if (auto cleartextInt = dynamic_cast<const Cleartext<int> *>(&operand)) {
    auto plaintext = getFactory().createPlaintext(cleartextInt->getData());
   // double old_noise = this->_noise;
    uint64_t rtq = this->getFactory().getContext().first_context_data()->coeff_modulus_mod_plain_modulus();
    uint64_t plain_max_coeff_count = plaintext->nonzero_coeff_count();
    uint64_t plain_max_abs_value = plaintext_norm(*plaintext);
    // noise is old_noise + r_t(q) * plain_max_coeff_count * plain_max_abs_value
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_t  rt_q;
    mpz_init(rt_q);
    mpz_init_set_ui(rt_q, rtq);
    mpz_t  plain_coeff_ct;
    mpz_init(plain_coeff_ct);
    mpz_init_set_ui(plain_coeff_ct, plain_max_coeff_count);
    mpz_t  plain_abs;
    mpz_init(plain_abs);
    mpz_init_set_ui(plain_abs, plain_max_abs_value);
    mpz_mul(result_noise, rt_q, plain_coeff_ct);
    mpz_mul(result_noise, result_noise, plain_abs);
    mpz_add(result_noise, result_noise, this->_noise);
    //copy
    auto r = std::make_unique<SimulatorCiphertext>(*this);
    mpz_set(this->_noise,result_noise);
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
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::rotateRows(int) const {
  throw std::runtime_error("Not yet implemented.");
}
void SimulatorCiphertext::rotateRowsInplace(int) {
  throw std::runtime_error("Not yet implemented.");
}

int SimulatorCiphertext::noiseBits() const{
  size_t coeff_modulus_significant_bit_count = this->getFactory().getContext().first_context_data()->total_coeff_modulus_bit_count();
  size_t log_noise = mpz_sizeinbase(this->_noise, 2);
  return std::max(int(coeff_modulus_significant_bit_count - log_noise - 1), 0);
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

void SimulatorCiphertext::divide_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation divide_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::modulo_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation modulo_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalAnd_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalAnd_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalOr_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalOr_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalLess_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalLess_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalLessEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalLessEqual_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalGreater_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalGreater_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalGreaterEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalGreaterEqual_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalEqual_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::logicalNotEqual_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation logicalNotEqual_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseAnd_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation bitwiseAnd_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseXor_inplace(const AbstractValue &) {
  throw std::runtime_error("Operation bitwiseXor_inplace not supported for (SimulatorCiphertext, ANY).");
}

void SimulatorCiphertext::bitwiseOr_inplace(const AbstractValue &) {
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
