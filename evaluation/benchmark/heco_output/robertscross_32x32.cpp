seal::Ciphertext encryptedRobertsCross_32x32(seal::Ciphertext v1) {
  seal::Ciphertext v2 = evaluator_rotate(v1, 991);
  seal::Ciphertext v3 = evaluator_sub(v1, v2);
  seal::Ciphertext v4 = evaluator_rotate(v1, 993);
  seal::Ciphertext v5 = evaluator_sub(v1, v4);
  seal::Ciphertext v6 = evaluator_multiply(v3, v3);
  seal::Ciphertext v7 = evaluator_multiply(v5, v5);
  seal::Ciphertext v8 = evaluator_rotate(v6, 33);
  seal::Ciphertext v9 = evaluator_rotate(v7, 32);
  seal::Ciphertext v10 = evaluator_add(v8, v9);
  return v10;
}


