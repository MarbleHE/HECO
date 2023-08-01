///  For a quadratic polynomial (from the porcupine paper)
/// This is the naive non-batched version
std::vector<seal::Ciphertext> encryptedQuadraticPolynomialNaive(
    std::vector<seal::Ciphertext> a_ctxt, std::vector<seal::Ciphertext> b_ctxt, std::vector<seal::Ciphertext> c_ctxt,
    std::vector<seal::Ciphertext> x_ctxt, std::vector<seal::Ciphertext> y_ctxt)
{
    for (size_t i = 0; i < a_ctxt.size(); ++i)
    {
        evaluator->multiply_inplace(a_ctxt[i], x_ctxt[i]);
        evaluator->relinearize_inplace(a_ctxt[i], *relinkeys);
        evaluator->multiply_inplace(a_ctxt[i], x_ctxt[i]);

        evaluator->multiply_inplace(b_ctxt[i], x_ctxt[i]);

        evaluator->add_inplace(c_ctxt[i], b_ctxt[i]);
        evaluator->add_inplace(c_ctxt[i], a_ctxt[i]);

        evaluator->sub_inplace(y_ctxt[i], c_ctxt[i]);
    }

    return y_ctxt;
}