#include "seal/ciphertext.h"
#include "seal/evaluator.h"
#include "seal/plaintext.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string.h>

std::unique_ptr<seal::SEALContext> context = nullptr;
std::unique_ptr<seal::Evaluator> evaluator = nullptr;

inline void insert(std::vector<seal::Ciphertext> &v, seal::Ciphertext &c)
{
    v.push_back(c);
}

inline seal::EncryptionParameters load_parms(std::string parms_file)
{
    std::ifstream parms_file_stream;
    parms_file_stream.open(parms_file);
    seal::EncryptionParameters parms;
    parms.load(parms_file_stream);
    return parms;
}

inline seal::PublicKey evaluator_load_public_key(std::string data_file, std::string parms_file)
{
    context = std::make_unique<seal::SEALContext>(load_parms(parms_file));
    evaluator = std::make_unique<seal::Evaluator>(*context);

    std::ifstream data_file_stream;
    data_file_stream.open(data_file);
    seal::PublicKey pk;
    pk.load(*context, data_file_stream);
    return pk;
}

inline seal::RelinKeys evaluator_load_relin_keys(std::string data_file, std::string parms_file)
{
    context = std::make_unique<seal::SEALContext>(load_parms(parms_file));
    evaluator = std::make_unique<seal::Evaluator>(*context);

    std::ifstream data_file_stream;
    data_file_stream.open(data_file);
    seal::RelinKeys rks;
    rks.load(*context, data_file_stream);
    return rks;
}

inline seal::GaloisKeys evaluator_load_galois_keys(std::string data_file, std::string parms_file)
{
    context = std::make_unique<seal::SEALContext>(load_parms(parms_file));
    evaluator = std::make_unique<seal::Evaluator>(*context);

    std::ifstream data_file_stream;
    data_file_stream.open(data_file);
    seal::GaloisKeys gks;
    gks.load(*context, data_file_stream);
    return gks;
}

inline seal::Ciphertext evaluator_load_ctxt(std::string data_file, std::string parms_file)
{
    context = std::make_unique<seal::SEALContext>(load_parms(parms_file));
    evaluator = std::make_unique<seal::Evaluator>(*context);

    std::ifstream data_file_stream;
    data_file_stream.open(data_file);
    seal::Ciphertext ctx;
    ctx.load(*context, data_file_stream);
    return ctx;
}

inline void evaluator_save_ctxt(const seal::Ciphertext &c, std::string filename)
{
    std::ofstream file;
    file.open(filename);
    c.save(file);
    file.close();
}

inline seal::Ciphertext evaluator_multiply(seal::Ciphertext &a, seal::Ciphertext &b)
{
    assert(context && evaluator);
    seal::Ciphertext result;
    evaluator->multiply(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_multiply_many(std::vector<seal::Ciphertext> &as, seal::RelinKeys &rlk)
{
    assert(context && evaluator);
    seal::Ciphertext result;
    evaluator->multiply_many(as, rlk, result);
    return result;
}

inline seal::Ciphertext evaluator_add(seal::Ciphertext &a, seal::Ciphertext &b)
{
    assert(context && evaluator);
    seal::Ciphertext result;
    evaluator->add(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_add_many(std::vector<seal::Ciphertext> &as)
{
    assert(context && evaluator);
    seal::Ciphertext result;
    evaluator->add_many(as, result);
    return result;
}

inline seal::Ciphertext evaluator_relinearize(seal::Ciphertext &a, const seal::RelinKeys &b)
{
    assert(context && evaluator);
    seal::Ciphertext result;
    evaluator->relinearize(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_modswitch_to(seal::Ciphertext &a, seal::Ciphertext &b)
{
    assert(context && evaluator);
    seal::Ciphertext result;
    evaluator->mod_switch_to(a, b.parms_id(), result);
    return result;
}

inline seal::Ciphertext evaluator_rotate(seal::Ciphertext &a, int i, const seal::GaloisKeys &k)
{
    assert(context && evaluator);
    seal::Ciphertext result;
    evaluator->rotate_rows_inplace(a, i, k);
    return result;
}