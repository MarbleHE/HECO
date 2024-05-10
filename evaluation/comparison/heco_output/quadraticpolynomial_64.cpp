seal::Ciphertext encryptedQuadraticPolynomial_64(seal::Ciphertext v1, seal::Ciphertext v2, seal::Ciphertext v3, seal::Ciphertext v4, seal::Ciphertext v5) {
  seal::Ciphertext v6 = evaluator_multiply(v1, v4);
  seal::Ciphertext v7 = evaluator_add(v6, v2);
  seal::Ciphertext v8 = evaluator_multiply(v4, v7);
  seal::Ciphertext v9 = evaluator_add(v8, v3);
  seal::Ciphertext v10 = evaluator_sub(v5, v9);
  return v10;
}


