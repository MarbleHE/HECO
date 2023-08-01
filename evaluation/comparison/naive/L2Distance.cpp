/// Output is squared to elide square root
/// Naive version of encrypted l2 distance. Each value will be it's own ciphertext
seal::Ciphertext encryptedL2DistanceNaive(std::vector<seal::Ciphertext> x_ctxt, std::vector<seal::Ciphertext> y_ctxt)
{
    seal::Plaintext sum_ptxt;
    seal::Ciphertext sum_ctxt;
    sum_ptxt.set_zero();
    encryptor->encrypt(sum_ptxt, sum_ctxt);

    for (size_t i = 0; i < x_ctxt.size(); ++i)
    {
        // sum += (x[i] - y[i])*(x[i] - y[i]);
        evaluator->sub_inplace(x_ctxt[i], y_ctxt[i]);
        evaluator->square_inplace(x_ctxt[i]);
        evaluator->add_inplace(sum_ctxt, x_ctxt[i]);
    }

    return sum_ctxt;
}