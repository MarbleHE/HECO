/// \param a vector of size n
/// \param b vector of size n
seal::Ciphertext encryptedHammingDistanceNaive(
    std::vector<seal::Ciphertext> a_ctxt, std::vector<seal::Ciphertext> b_ctxt)
{
    seal::Plaintext sum_ptxt;
    seal::Ciphertext sum_ctxt;
    sum_ptxt.set_zero();
    encryptor->encrypt(sum_ptxt, sum_ctxt);

    // Compute differences
    // Note: We can use the fact that NEQ = XOR = (a-b)^2 for a,b \in {0,1}
    for (size_t i = 0; i < a_ctxt.size(); ++i)
    {
        evaluator->sub_inplace(a_ctxt[i], b_ctxt[i]);
        evaluator->square_inplace(a_ctxt[i]);
        evaluator->relinearize_inplace(a_ctxt[i], *relinkeys);
        evaluator->add_inplace(sum_ctxt, a_ctxt[i]);
    }

    return sum_ctxt;
}