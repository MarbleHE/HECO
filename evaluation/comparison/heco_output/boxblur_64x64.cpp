seal::Ciphertext encryptedBoxBlur_64x64(seal::Ciphertext v1) {
  seal::Ciphertext v2 = evaluator_rotate(v1, 64);
  seal::Ciphertext v3 = evaluator_rotate(v1, 65);
  seal::Ciphertext v4 = evaluator_rotate(v1, 1);
  std::vector<seal::Ciphertext> v5 = std::vector<seal::Ciphertext>();
  insert(v5, v1);
  insert(v5, v2);
  insert(v5, v3);
  insert(v5, v4);
  seal::Ciphertext v6 = evaluator_add_many(v5);
  return v6;
}


