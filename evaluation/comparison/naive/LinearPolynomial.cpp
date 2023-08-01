std::vector<seal::Ciphertext> encryptedLinearPolynomialNaive(
    std::vector<seal::Ciphertext> a_ctxt, std::vector<seal::Ciphertext> b_ctxt, std::vector<seal::Ciphertext> x_ctxt,
    std::vector<seal::Ciphertext> y_ctxt)
{
    for (size_t i = 0; i < a_ctxt.size(); ++i)
    {
        evaluator->multiply_inplace(a_ctxt[i], x_ctxt[i]);
        evaluator->sub_inplace(y_ctxt[i], a_ctxt[i]);
        evaluator->sub_inplace(y_ctxt[i], b_ctxt[i]);
    }

    return y_ctxt;
}