//
// Created by Alexa on 21/04/2021.
//

#include "SimulatorCiphertext.h"
std::unique_ptr<AbstractCiphertext> SimulatorCiphertext::multiply(AbstractCiphertext &operand) {

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
AbstractCiphertextFactory &SimulatorCiphertext::getFactory() {
  return AbstractCiphertext::getFactory();
}
const AbstractCiphertextFactory &SimulatorCiphertext::getFactory() const {
  return AbstractCiphertext::getFactory();
}
std::unique_ptr<SimulatorCiphertext> SimulatorCiphertext::clone_impl() {
  //TODO: actual cloning
  return std::unique_ptr<SimulatorCiphertext>();
}
