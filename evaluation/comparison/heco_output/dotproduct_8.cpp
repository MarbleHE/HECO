seal::Ciphertext encryptedDotProduct_8(seal::Ciphertext v1, seal::Ciphertext v2) {
  seal::Ciphertext v3 = evaluator_multiply(v1, v2);
  seal::Ciphertext v4 = evaluator_rotate(v3, 2);
  seal::Ciphertext v5 = evaluator_add(v3, v4);
  seal::Ciphertext v6 = evaluator_rotate(v5, 1);
  seal::Ciphertext v7 = evaluator_add(v5, v6);
  seal::Ciphertext v8 = evaluator_rotate(v7, 3);
  return v8;
}


