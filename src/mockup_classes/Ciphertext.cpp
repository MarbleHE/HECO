#include "ast_opt/mockup_classes/Ciphertext.h"

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
  Ciphertext ctxt = getCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
  return applyBinaryOp(std::plus<double>{}, *this, ctxt);
}

Ciphertext Ciphertext::getCiphertext(const double plaintextScalar, int fillNSlots, int totalNumCtxtSlots) const {
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

Ciphertext Ciphertext::operator*(const double plaintextScalar) const {
  Ciphertext ctxt = getCiphertext(plaintextScalar, getNumCiphertextElements(), getNumCiphertextSlots());
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
  lhs.verifyNumElementsAndAlignment(rhs);

  Ciphertext result({});
  result.offsetOfFirstElement = getOffsetOfFirstElement();

  auto targetIdx = computeCyclicEndIndex(offsetOfFirstElement, getNumCiphertextElements());
  for (auto i = offsetOfFirstElement; i!=cyclicIncrement(targetIdx, result.data); i = cyclicIncrement(i, result.data)) {
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
  Ciphertext result = *this;
  auto rotTarget = (n > 0) ? (result.data.begin() + result.getNumCiphertextSlots() - n) : (result.data.begin() - n);
  std::rotate(result.data.begin(), rotTarget, result.data.end());
  return result;
}

double &Ciphertext::getElementAt(int n) {
  return data.at(n);
}
