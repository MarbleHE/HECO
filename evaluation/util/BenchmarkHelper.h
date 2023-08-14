#ifndef HECO_EVAL_BENCHMARKHELPER_H
#define HECO_EVAL_BENCHMARKHELPER_H

#include "seal/seal.h"
#include <random>
#include <string>

std::unique_ptr<seal::Evaluator> evaluator;
std::unique_ptr<seal::RelinKeys> relinkeys;
std::unique_ptr<seal::GaloisKeys> galoiskeys;
std::unique_ptr<seal::BatchEncoder> encoder;
std::unique_ptr<seal::Encryptor> encryptor;
std::unique_ptr<seal::Decryptor> decryptor;

/// @brief Generates SEAL parameters and sets up required helper objects
/// @param poly_modulus_degree Degree of the polynomial modulus (i.e., size of the ring)
inline void keygen(size_t poly_modulus_degree)
{
    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(seal::PlainModulus::Batching(poly_modulus_degree, 20));
    seal::SEALContext context(parms);
    seal::KeyGenerator keygen(context);
    seal::SecretKey secret_key = keygen.secret_key();
    seal::PublicKey public_key;
    keygen.create_public_key(public_key);
    relinkeys = std::make_unique<seal::RelinKeys>();
    keygen.create_relin_keys(*relinkeys);
    galoiskeys = std::make_unique<seal::GaloisKeys>();
    keygen.create_galois_keys(*galoiskeys);
    encoder = std::make_unique<seal::BatchEncoder>(context);
    encryptor = std::make_unique<seal::Encryptor>(context, public_key);
    evaluator = std::make_unique<seal::Evaluator>(context);
    decryptor = std::make_unique<seal::Decryptor>(context, secret_key);
}

inline std::vector<seal::Ciphertext> encrypt_naive(std::vector<uint64_t> input)
{
    std::vector<seal::Ciphertext> ctxts(input.size());
    seal::Plaintext ptx;
    for (size_t i = 0; i < input.size(); ++i)
    {
        uint64_t value = input[i];
        ptx = seal::Plaintext(seal::util::uint_to_hex_string(&value, std::size_t(1)));
        encryptor->encrypt(ptx, ctxts[i]);
    }
    return ctxts;
}

inline std::vector<uint64_t> decrypt_naive(std::vector<seal::Ciphertext> ctxts)
{
    std::vector<uint64_t> result(ctxts.size());
    seal::Plaintext ptx;
    for (size_t i = 0; i < ctxts.size(); ++i)
    {
        decryptor->decrypt(ctxts[i], ptx);
        result[i] = *ptx.data();
    }
    return result;
}

inline uint64_t decrypt_single_naive(seal::Ciphertext ctxt)
{
    seal::Plaintext ptx;
    decryptor->decrypt(ctxt, ptx);
    return *ptx.data();
}

inline seal::Ciphertext encrypt_batched(std::vector<uint64_t> input)
{
    seal::Plaintext plain;
    encoder->encode(input, plain);
    seal::Ciphertext ctxt;
    encryptor->encrypt(plain, ctxt);
    return ctxt;
}

inline std::vector<uint64_t> decrypt_batched(seal::Ciphertext ctxt)
{
    seal::Plaintext decrypted;
    decryptor->decrypt(ctxt, decrypted);

    std::vector<uint64_t> result;
    encoder->decode(decrypted, result);

    return result;
}

// use this fixed seed to enable reproducibility of the matrix inputs
#define RAND_SEED 4673838

inline void getInputMatrix(size_t size, std::vector<std::vector<int>> &destination)
{
    // reset the RNG to make sure that every call to this method results in the same numbers
    auto randomEngine = std::default_random_engine(RAND_SEED);
    auto myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);

    // make sure we clear desination vector before, otherwise resize could end up appending elements
    destination.clear();
    destination.resize(size, std::vector<int>(size));
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            destination[i][j] = myUnifIntDist(randomEngine);
        }
    }
}

inline void getInputMatrix(size_t size, std::vector<int> &destination)
{
    // make sure we clear desination vector before, otherwise resize could end up appending elements
    destination.clear();
    std::vector<std::vector<int>> data;
    getInputMatrix(size, data);
    std::size_t total_size = 0;
    for (const auto &sub : data)
        total_size += sub.size();
    destination.reserve(total_size);
    for (const auto &sub : data)
        destination.insert(destination.end(), sub.begin(), sub.end());
}

inline void getRandomVector(std::vector<int> &destination)
{
    // reset the RNG to make sure that every call to this method results in the same numbers
    auto randomEngine = std::default_random_engine(RAND_SEED);
    auto myUnifIntDist = std::uniform_int_distribution<int>(0, 1024);

    for (int &i : destination)
    {
        i = myUnifIntDist(randomEngine);
    }
}

#endif // HECO_EVAL_BENCHMARKHELPER_H
