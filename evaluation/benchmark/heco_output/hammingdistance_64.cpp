seal::Ciphertext encryptedHammingDistance_64(seal::Ciphertext v1, seal::Ciphertext v2) {
  seal::Ciphertext v3 = evaluator_sub(v1, v2);
  seal::Ciphertext v4 = evaluator_multiply(v3, v3);
  seal::Ciphertext v5 = evaluator_rotate(v4, 32);
  seal::Ciphertext v6 = evaluator_add(v4, v5);
  seal::Ciphertext v7 = evaluator_rotate(v6, 16);
  seal::Ciphertext v8 = evaluator_add(v6, v7);
  seal::Ciphertext v9 = evaluator_rotate(v8, 8);
  seal::Ciphertext v10 = evaluator_add(v8, v9);
  seal::Ciphertext v11 = evaluator_rotate(v10, 4);
  seal::Ciphertext v12 = evaluator_add(v10, v11);
  seal::Ciphertext v13 = evaluator_rotate(v12, 2);
  seal::Ciphertext v14 = evaluator_add(v12, v13);
  seal::Ciphertext v15 = evaluator_rotate(v14, 1);
  seal::Ciphertext v16 = evaluator_add(v14, v15);
  seal::Ciphertext v17 = evaluator_rotate(v16, 62);
  return v17;
}


