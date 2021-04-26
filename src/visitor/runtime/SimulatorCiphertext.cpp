
#include "ast_opt/utilities/Operator.h"
#include "ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertext.h"
#include "ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"

#ifdef HAVE_SEAL_BFV
#include <seal/seal.h>


// Constructor for a simulated ciphertext that is created given seal params, size and noise
SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertextFactory &acf,
                                         const seal::EncryptionParameters &parms,
                                         int ciphertext_size,
                                         int noise_budget) : AbstractCiphertext(acf) {

  // TODO: this is hard since BigUInt class is gone...

}

SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertextFactory &simulatorFactory) : AbstractCiphertext(simulatorFactory) {}

SimulatorCiphertext::SimulatorCiphertext(const SimulatorCiphertext &other)  // copy constructor
    : AbstractCiphertext(other.factory) {
  ciphertext = seal::Ciphertext(other.ciphertext);
}

SimulatorCiphertext &SimulatorCiphertext::operator=(const SimulatorCiphertext &other) {  // copy assignment
  return *this = SimulatorCiphertext(other);
}

SimulatorCiphertext::SimulatorCiphertext(SimulatorCiphertext &&other) noexcept  // move constructor
    : AbstractCiphertext(other.factory), ciphertext(std::move(other.ciphertext)) {}

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
  int64_t  coeff_modulus = 1;
  for (auto mod : parms.coeff_modulus())
  {
    coeff_modulus *= mod.value();
  }
  int coeff_bit_count = coeff_modulus.seal::util::get_significant_bit_count();
  int coeff_uint64_count = seal::util::divide_round_up(coeff_bit_count, seal::util::bits_per_uint64);
  int poly_modulus_degree = parms.poly_modulus_degree() - 1; // check this if it is actually the same!! (here and everywhere else)

  // Widen plain_modulus_ and noise_
  //TODO continue

  // Noise is ~ r_t(q)*plain_max_abs_value*plain_max_coeff_count + 7 * min(B, 6*sigma)*t*n
  //TODO


}

std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiply(AbstractCiphertext &operand) {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  // cast operand
  std::unique_ptr<SimulatorCiphertext> operand_ctxt = std::unique_ptr<SimulatorCiphertext>(&cast(operand));

  uint64_t poly_modulus_degree = new_ctxt->getFactory().getContext().first_context_data()->parms().poly_modulus_degree() - 1;
  uint64_t plain_modulus = new_ctxt->getFactory().getContext().first_context_data()->parms().plain_modulus().value();
  // compute product coeff modulus
  uint64_t coeff_modulus = 1;
  for (auto mod : new_ctxt->getFactory().getContext().first_context_data()->parms().coeff_modulus())
  {
    coeff_modulus *= mod.value();
  }

  // Compute Noise (this is as in Seal v2.3.0):
  // Noise is ~ t * sqrt(3n) * [ (12n)^(j1/2)*noise2 + (12n)^(j2/2)*noise1 + (12n)^((j1+j2)/2) ]
  // First compute sqrt(12n) (rounding up) and the powers needed
  uint64_t sqrt_factor_base = static_cast<uint64_t>(ceil(sqrt(static_cast<double>(12 * poly_modulus_degree))));
  uint64_t sqrt_factor_1 = seal::util::exponentiate_uint(sqrt_factor_base,
                                                         new_ctxt->getFactory().getCiphertextSlotSize() - 1);
  uint64_t sqrt_factor_2 = seal::util::exponentiate_uint(sqrt_factor_base, operand_ctxt->getFactory().getCiphertextSlotSize() - 1);
  uint64_t sqrt_factor_total = seal::util::exponentiate_uint(sqrt_factor_base,
                                                   new_ctxt->getFactory().getCiphertextSlotSize() - 1 + operand_ctxt->getFactory().getCiphertextSlotSize() - 1);
  // Compute also t * sqrt(3n)
  uint64_t leading_sqrt_factor = static_cast<uint64_t>(ceil(sqrt(static_cast<double>(3 * poly_modulus_degree))));
  uint64_t leading_factor = plain_modulus * leading_sqrt_factor;

  int64_t result_noise = operand_ctxt->_noise * sqrt_factor_1
      + this->_noise * sqrt_factor_2
      + sqrt_factor_total;
  result_noise *= leading_factor; // this is the resulting invariant noise

  //TODO: calc noise budget

  //std::unique_ptr<SimulatorCiphertext> m_1 = new_ctxt->getFactory().decryptCiphertext(new_ctxt, );
  return std::unique_ptr<AbstractCiphertext>();
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
  return noise_budget;
}
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::clone() {
  return clone_impl();
}

SimulatorCiphertextFactory &SimulatorCiphertext::getFactory() {
  return const_cast<SimulatorCiphertextFactory &>(const_cast<const SimulatorCiphertext *>(this)->getFactory());
}
const SimulatorCiphertextFactory &SimulatorCiphertext::getFactory() const {
  if (auto simFactory = dynamic_cast<SimulatorCiphertextFactory *>(&factory)) {
    return *simFactory;
  } else {
    throw std::runtime_error("Cast of AbstractFactory to SimulatorFactory failed. SimulatorCiphertext is probably invalid.");
  }
}

std::unique_ptr<SimulatorCiphertext> SimulatorCiphertext::clone_impl() {
  //TODO: check
  return std::unique_ptr<SimulatorCiphertext>(this);
}


#endif
