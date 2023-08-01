/// For a linear polynomial (from the porcupine paper)
/// Ciphertext linear_reg(Ciphertext a, Ciphertext b,
///                       Ciphertext x, Ciphertext y)
///     Ciphertext c1 = multiply(a, x)
///     c1 = relinearize(c1)
///     Ciphertext c2 = sub(y, c1)
///     return sub(c2, b)
seal::Ciphertext encryptedLinearPolynomialPorcupine(
    seal::Ciphertext a_ctxt, seal::Ciphertext b_ctxt, seal::Ciphertext x_ctxt, seal::Ciphertext y_ctxt)
{
    // Ciphertext c1 = multiply(a, x)
    seal::Ciphertext c1;
    evaluator->multiply(a_ctxt, x_ctxt, c1);
    // c1 = relinearize(c1)
    evaluator->relinearize_inplace(c1, *relinkeys);
    // Ciphertext c2 = sub(y, c1)
    seal::Ciphertext c2;
    evaluator->sub(y_ctxt, c1, c2);
    // return sub(c2, b)
    seal::Ciphertext result_ctxt;
    evaluator->sub(c2, b_ctxt, result_ctxt);
    return result_ctxt;
}