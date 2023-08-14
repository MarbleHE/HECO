#ifndef HECO_OUTPUT_EMITC_HELPER_H
#define HECO_OUTPUT_EMITC_HELPER_H

#include "seal/seal.h"

inline void insert(std::vector<seal::Ciphertext> &v, seal::Ciphertext &c)
{
    v.push_back(c);
}

inline seal::Plaintext evaluator_encode(int16_t value)
{
    seal::Plaintext plain;
    std::vector<uint64_t> vector(8192, value);
    encoder->encode(vector, plain);
    return plain;
}

inline seal::Ciphertext evaluator_multiply(seal::Ciphertext &a, seal::Ciphertext &b)
{
    seal::Ciphertext result;
    if (a.size() > 2)
        evaluator->relinearize_inplace(a, *relinkeys);
    if (b.size() > 2)
        evaluator->relinearize_inplace(b, *relinkeys);
    if (&a == &b)
        evaluator->square(a, result);
    else
        evaluator->multiply(a, b, result);

    return result;
}

inline seal::Ciphertext evaluator_multiply_plain(seal::Ciphertext &a, seal::Plaintext &b)
{
    seal::Ciphertext result;
    evaluator->multiply_plain(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_multiply_many(std::vector<seal::Ciphertext> &as, seal::RelinKeys &rlk)
{
    seal::Ciphertext result;
    evaluator->multiply_many(as, rlk, result);
    return result;
}

inline seal::Ciphertext evaluator_add(seal::Ciphertext &a, seal::Ciphertext &b)
{
    seal::Ciphertext result;
    evaluator->add(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_sub(seal::Ciphertext &a, seal::Ciphertext &b)
{
    seal::Ciphertext result;
    evaluator->sub(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_add_many(std::vector<seal::Ciphertext> &as)
{
    seal::Ciphertext result;
    evaluator->add_many(as, result);
    return result;
}

inline seal::Ciphertext evaluator_relinearize(seal::Ciphertext &a, const seal::RelinKeys &b)
{
    seal::Ciphertext result;
    evaluator->relinearize(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_modswitch_to(seal::Ciphertext &a, seal::Ciphertext &b)
{
    seal::Ciphertext result;
    evaluator->mod_switch_to(a, b.parms_id(), result);
    return result;
}

inline seal::Ciphertext evaluator_rotate(seal::Ciphertext &a, int i)
{
    seal::Ciphertext result;
    if (a.size() > 2)
        evaluator->relinearize_inplace(a, *relinkeys);
    evaluator->rotate_rows(a, i, *galoiskeys, result);
    return result;
}
#endif // HECO_OUTPUT_EMITC_HELPER_H