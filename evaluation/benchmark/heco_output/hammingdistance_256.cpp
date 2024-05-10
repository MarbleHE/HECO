seal::Ciphertext encryptedHammingDistance_256(seal::Ciphertext v1, seal::Ciphertext v2) {
  seal::Ciphertext v3 = evaluator_sub(v1, v2);
  seal::Ciphertext v4 = evaluator_multiply(v3, v3);
  seal::Ciphertext v5 = evaluator_rotate(v4, 128);
  seal::Ciphertext v6 = evaluator_add(v4, v5);
  seal::Ciphertext v7 = evaluator_rotate(v6, 64);
  seal::Ciphertext v8 = evaluator_add(v6, v7);
  seal::Ciphertext v9 = evaluator_rotate(v8, 32);
  seal::Ciphertext v10 = evaluator_add(v8, v9);
  seal::Ciphertext v11 = evaluator_rotate(v10, 16);
  seal::Ciphertext v12 = evaluator_add(v10, v11);
  seal::Ciphertext v13 = evaluator_rotate(v12, 8);
  seal::Ciphertext v14 = evaluator_add(v12, v13);
  seal::Ciphertext v15 = evaluator_rotate(v14, 4);
  seal::Ciphertext v16 = evaluator_add(v14, v15);
  seal::Ciphertext v17 = evaluator_rotate(v16, 2);
  seal::Ciphertext v18 = evaluator_add(v16, v17);
  seal::Ciphertext v19 = evaluator_rotate(v18, 1);
  seal::Ciphertext v20 = evaluator_add(v18, v19);
  seal::Ciphertext v21 = evaluator_rotate(v20, 254);
  return v21;
}


