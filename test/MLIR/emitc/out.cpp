#include "wrapper.cpp.inc"

seal::Ciphertext encryptedBoxBlur(seal::Ciphertext v1)
{
    std::string v2 = "glk.parms";
    std::string v3 = "foo.glk";
    seal::GaloisKeys v4 = evaluator_load_galois_keys(v3, v2);
    seal::Ciphertext v5 = evaluator_rotate(v1, 55, v4);
    seal::Ciphertext v6 = evaluator_rotate(v1, 63, v4);
    seal::Ciphertext v7 = evaluator_rotate(v1, 7, v4);
    seal::Ciphertext v8 = evaluator_rotate(v1, 56, v4);
    seal::Ciphertext v9 = evaluator_rotate(v1, 8, v4);
    seal::Ciphertext v10 = evaluator_rotate(v1, 57, v4);
    seal::Ciphertext v11 = evaluator_rotate(v1, 9, v4);
    seal::Ciphertext v12 = evaluator_rotate(v1, 1, v4);
    std::vector<seal::Ciphertext> v13 = std::vector<seal::Ciphertext>();
    insert(v13, v5);
    insert(v13, v6);
    insert(v13, v7);
    insert(v13, v8);
    insert(v13, v1);
    insert(v13, v9);
    insert(v13, v10);
    insert(v13, v11);
    insert(v13, v12);
    seal::Ciphertext v14 = evaluator_add_many(v13);
    return v14;
}

int main()
{
    // TODO: Writing a main function is up to the developer for now!
}