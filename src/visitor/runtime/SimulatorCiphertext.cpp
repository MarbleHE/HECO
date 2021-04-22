//
// Created by Alexa on 21/04/2021.
//

#include "ast_opt/visitor/runtime/SimulatorCiphertext.h"

int64_t SimulatorCiphertext::initialNoise(AbstractCiphertext &operand) {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();

  return 0;
}


std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiply(AbstractCiphertext &operand) {
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  std::unique_ptr<SimulatorCiphertext> new_ctxt2 = operand->clone();
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
  uint64_t sqrt_factor_2 = seal::util::exponentiate_uint(sqrt_factor_base, operand.ciphertext_size_ - 1);
  uint64_t sqrt_factor_total = exponentiate_uint64(sqrt_factor_base,
                                                   new_ctxt->getFactory().getCiphertextSlotSize() - 1 + operand.ciphertext_size_ - 1);


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
  auto new_noiseBits = this->noiseBits() + operand.noiseBits();
  std::unique_ptr<SimulatorCiphertext> new_ctxt = this->clone_impl();
  new_ctxt->noise_budget = new_noiseBits;
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
  //TODO: actual cloning
  return std::unique_ptr<SimulatorCiphertext>();
}

