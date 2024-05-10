seal::Ciphertext encryptedLinearPolynomial_64(seal::Ciphertext v1, seal::Ciphertext v2, seal::Ciphertext v3, seal::Ciphertext v4) {
  seal::Ciphertext v5 = evaluator_multiply(v1, v3);
  seal::Ciphertext v6 = evaluator_sub(v4, v5);
  seal::Ciphertext v7 = evaluator_sub(v6, v2);
  return v7;
}


