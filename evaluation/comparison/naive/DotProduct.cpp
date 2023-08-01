/// Naive version of dot product, where one ciphertext contains one value.
/// \param x a vector of size n
/// \param y a vector of size n
/// \param poly_modulus_degree FHE parameter, degree n of the polynomials. Has to larger than n.
seal::Ciphertext encryptedDotProductNaive(std::vector<seal::Ciphertext> &x, const std::vector<seal::Ciphertext> &y)
{
    seal::Plaintext sum_ptxt;
    seal::Ciphertext sum_ctxt;
    sum_ptxt.set_zero();
    encryptor->encrypt(sum_ptxt, sum_ctxt);

    // Compute differences
    for (size_t i = 0; i < x.size(); ++i)
    {
        // sum += x[i] * y[i]
        evaluator->multiply_inplace(x[i], y[i]);
        evaluator->add_inplace(sum_ctxt, x[i]);
    }
    return sum_ctxt;
}