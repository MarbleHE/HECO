#include "seal/ciphertext.h"
#include "seal/evaluator.h"
#include "seal/plaintext.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string.h>

seal::Ciphertext trace();

std::unique_ptr<seal::SEALContext> context = nullptr;
std::unique_ptr<seal::Evaluator> evaluator = nullptr;

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

inline seal::RelinKeys evaluator_load_galois_keys(std::string data_file, std::string parms_file)
{
    context = std::make_unique<seal::SEALContext>(load_parms(parms_file));
    evaluator = std::make_unique<seal::Evaluator>(*context);

    std::ifstream data_file_stream;
    data_file_stream.open(data_file);
    seal::RelinKeys rks;
    rks.load(*context, data_file_stream);
    return rks;
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

inline seal::Ciphertext evaluator_relinearize(seal::Ciphertext &a, seal::RelinKeys &b)
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

int main()
{
    evaluator_save_ctxt(trace(), "trace.ctxt");
}