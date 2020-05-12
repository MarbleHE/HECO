#include "ast_opt/mockup_classes/Plaintext.h"

Plaintext::Plaintext(const Ciphertext &ctxt) : Ciphertext(ctxt) {}

Plaintext::Plaintext(double scalar, int numCiphertextSlots) : Ciphertext(scalar, numCiphertextSlots) {}

Plaintext::Plaintext(const std::vector<double> &data, int numCiphertextSlots) : Ciphertext(data, numCiphertextSlots) {}
