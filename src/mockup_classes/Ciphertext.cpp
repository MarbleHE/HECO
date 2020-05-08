#include "ast_opt/mockup_classes/Ciphertext.h"
#include <iostream>

Ciphertext::Ciphertext(std::vector<double> inputData, int numCiphertextSlots)
    : numCiphertextElements(inputData.size()), offsetOfFirstElement(0) {
  if (inputData.size() > numCiphertextSlots) {
    throw std::runtime_error("");
  }
  data.insert(data.begin(), inputData.begin(), inputData.end());
  data.resize(numCiphertextSlots);
}

Ciphertext Ciphertext::operator+(const Ciphertext &ctxt) const {
  return applyBinaryOp(std::plus<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::operator+(const double plaintextScalar) const {
  Ciphertext ctxt = generateCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  return applyBinaryOp(std::plus<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::generateCiphertext(const double plaintextScalar, int fillNSlots, int totalNumCtxtSlots) const {
  Ciphertext ctxt({}, totalNumCtxtSlots);
  ctxt.offsetOfFirstElement = getOffsetOfFirstElement();
  auto targetIdx = computeCyclicEndIndex(ctxt.getOffsetOfFirstElement(), getNumCiphertextElements());
  for (auto i = offsetOfFirstElement; i!=cyclicIncrement(targetIdx, ctxt.data); i = cyclicIncrement(i, ctxt.data)) {
    ctxt.data[i] = plaintextScalar;
  }
  ctxt.numCiphertextElements = getNumCiphertextElements();
  return ctxt;
}

Ciphertext Ciphertext::operator*(const Ciphertext &ctxt) const {
  return applyBinaryOp(std::multiplies<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::operator-(const Ciphertext &ctxt) const {
  return applyBinaryOp(std::minus<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::operator-(double plaintextScalar) const {
  Ciphertext ctxt = generateCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  return applyBinaryOp(std::minus<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::operator/(const Ciphertext &ctxt) const {
  return applyBinaryOp(std::divides<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::operator/(double plaintextScalar) const {
  Ciphertext ctxt = generateCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  return applyBinaryOp(std::divides<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::operator*(const double plaintextScalar) const {
  Ciphertext ctxt = generateCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  return applyBinaryOp(std::multiplies<double>{}, *this, ctxt);
}

int Ciphertext::computeCyclicEndIndex(int startIndex, int numElements) const {
  // computes the target index
  // for example, startIndex = 32, numElement = 112, numCtxtSlots = 128 results in
  // (32 + 112) % 128 = 144 % 128 = 16, i.e., first element is in slot 32, second in slot 33, etc. and last in slot
  // 16-1=15
  return ((startIndex + numElements)%getNumCiphertextSlots()) - 1;
}

int Ciphertext::getNumCiphertextElements() const {
  return numCiphertextElements;
}

void Ciphertext::verifyNumElementsAndAlignment(const Ciphertext &ctxt) const {
  std::stringstream ss;
  if (getNumCiphertextElements()!=ctxt.getNumCiphertextElements()) {
    ss << "Cannot perform action on ciphertexts as number of elements differ: " << getNumCiphertextElements();
    ss << " vs. " << ctxt.getNumCiphertextElements() << std::endl;
    throw std::runtime_error(ss.str());
  } else if (getOffsetOfFirstElement()!=ctxt.getOffsetOfFirstElement()) {
    ss << "Cannot peform action on ciphertexts as offset differs: " << getOffsetOfFirstElement();
    ss << " vs. " << ctxt.getOffsetOfFirstElement() << std::endl;
    throw std::runtime_error(ss.str());
  }
}

int Ciphertext::getOffsetOfFirstElement() const {
  return offsetOfFirstElement;
}

int Ciphertext::getNumCiphertextSlots() const {
  return data.capacity();
}

bool Ciphertext::operator==(const Ciphertext &rhs) const {
  return data==rhs.data &&
      offsetOfFirstElement==rhs.offsetOfFirstElement &&
      numCiphertextElements==rhs.numCiphertextElements;
}

bool Ciphertext::operator!=(const Ciphertext &rhs) const {
  return !(rhs==*this);
}

Ciphertext Ciphertext::applyBinaryOp(const std::function<double(double, double)> &binaryOp,
                                     const Ciphertext &lhs,
                                     const Ciphertext &rhs) const {
  //lhs.verifyNumElementsAndAlignment(rhs);
  if (lhs.getNumCiphertextSlots()!=rhs.getNumCiphertextSlots())
    throw std::runtime_error("");

  Ciphertext result({}, lhs.getNumCiphertextSlots());
  result.offsetOfFirstElement = getOffsetOfFirstElement();

  for (auto i = 0; i < lhs.getNumCiphertextSlots(); ++i) {
    result.data[i] = binaryOp(lhs.data[i], rhs.data[i]);
  }
  result.numCiphertextElements = getNumCiphertextElements();

  return result;
}

Ciphertext::Ciphertext(const Ciphertext &ctxt)
    : data(ctxt.data),
      offsetOfFirstElement(ctxt.offsetOfFirstElement),
      numCiphertextElements(ctxt.numCiphertextElements) {

}

int Ciphertext::cyclicIncrement(int i, const std::vector<double> &vec) {
  return (i < vec.size()) ? i + 1 : 0;
}

Ciphertext Ciphertext::rotate(int n) {
  if (n==0 || getNumCiphertextElements()==0) return Ciphertext(*this);

  Ciphertext result = *this;
  auto rotTarget = (n > 0) ? (result.data.begin() + result.getNumCiphertextSlots() - n) : (result.data.begin() - n);
  std::rotate(result.data.begin(), rotTarget, result.data.end());
  result.offsetOfFirstElement = (getOffsetOfFirstElement() + n)%result.getNumCiphertextSlots();
  return result;
}

double &Ciphertext::getElementAt(int n) {
  return data.at(n);
}

Ciphertext Ciphertext::sumaAndRotateAll() {
  return sumAndRotate(getNumCiphertextSlots()/2);
}

Ciphertext Ciphertext::sumAndRotatePartially(int numElementsToSum) {
  return sumAndRotate(numElementsToSum/2);
}

Ciphertext Ciphertext::sumAndRotate(int initialRotationFactor) {
  int rotationFactor = initialRotationFactor;
  // create a copy of this ctxt as otherwise we would need to treat the first iteration differently
  auto ctxt = *this;
  // perform rotate-and-sum in total requiring log_2(#ciphertextSlots) rotations
  while (rotationFactor >= 1) {
    auto rotatedCtxt = ctxt.rotate(rotationFactor);
    ctxt = ctxt + rotatedCtxt;
    rotationFactor = rotationFactor/2;
  }
  return ctxt;
}

void Ciphertext::printCiphertextData() {
  for (double val : data) {
    std::cout << val << std::endl;
  }
}

