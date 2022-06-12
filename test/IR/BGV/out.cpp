#include "wrapper.cpp.inc"

seal::Ciphertext trace()
{
    std::string v1 = "p_1.parms";
    std::string v2 = "encrpytion_3.ctxt";
    std::string v3 = "encrpytion_2.ctxt";
    std::string v4 = "p_0.parms";
    std::string v5 = "relin_keys_1.rk";
    std::string v6 = "public_key_0.pk";
    seal::PublicKey v7 = evaluator_load_public_key(v6, v4);
    seal::RelinKeys v8 = evaluator_load_relin_keys(v5, v4);
    seal::Ciphertext v9 = evaluator_load_ctxt(v3, v1);
    seal::Ciphertext v10 = evaluator_multiply(v9, v9);
    seal::Ciphertext v11 = evaluator_relinearize(v10, v8);
    seal::Ciphertext v12 = evaluator_multiply(v11, v11);
    seal::Ciphertext v13 = evaluator_relinearize(v12, v8);
    seal::Ciphertext v14 = evaluator_multiply(v13, v13);
    seal::Ciphertext v15 = evaluator_relinearize(v14, v8);
    seal::Ciphertext v16 = evaluator_load_ctxt(v2, v1);
    seal::Ciphertext v17 = evaluator_multiply(v16, v16);
    seal::Ciphertext v18 = evaluator_relinearize(v17, v8);
    seal::Ciphertext v19 = evaluator_modswitch_to(v18, v18);
    seal::Ciphertext v20 = evaluator_multiply(v19, v19);
    seal::Ciphertext v21 = evaluator_relinearize(v20, v8);
    seal::Ciphertext v22 = evaluator_modswitch_to(v21, v21);
    seal::Ciphertext v23 = evaluator_multiply(v22, v22);
    seal::Ciphertext v24 = evaluator_relinearize(v23, v8);
    seal::Ciphertext v25 = evaluator_modswitch_to(v24, v24);
    return v22;
}
