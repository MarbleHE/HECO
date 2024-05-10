seal::Ciphertext encryptedGxKernel_64x64(seal::Ciphertext v1) {
  int16_t v2 = -2;
  int16_t v3 = 2;
  int16_t v4 = -1;
  seal::Plaintext v5 = evaluator_encode(v2);
  seal::Ciphertext v6 = evaluator_multiply_plain(v1, v5);
  seal::Plaintext v7 = evaluator_encode(v4);
  seal::Ciphertext v8 = evaluator_multiply_plain(v1, v7);
  seal::Plaintext v9 = evaluator_encode(v3);
  seal::Ciphertext v10 = evaluator_multiply_plain(v1, v9);
  seal::Ciphertext v11 = evaluator_rotate(v8, 4095);
  seal::Ciphertext v12 = evaluator_rotate(v10, 63);
  seal::Ciphertext v13 = evaluator_rotate(v8, 64);
  seal::Ciphertext v14 = evaluator_rotate(v1, 65);
  seal::Ciphertext v15 = evaluator_rotate(v6, 1);
  std::vector<seal::Ciphertext> v16 = std::vector<seal::Ciphertext>();
  insert(v16, v11);
  insert(v16, v12);
  insert(v16, v1);
  insert(v16, v13);
  insert(v16, v14);
  insert(v16, v15);
  seal::Ciphertext v17 = evaluator_add_many(v16);
  return v17;
}


