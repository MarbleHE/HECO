///  For a quadratic polynomial (from the porcupine paper)
///  Ciphertext polynomial_reg(Ciphertext a, Ciphertext b,
///                            Ciphertext c, Ciphertext x, Ciphertext y)
///    Ciphertext c1 = multiply(a, x)
///    c1 = relinearize(c1)
///    Ciphertext c2 = add(c1, b)
///    Ciphertext c3 = multiply(x, c2)
///    c3 = relinearize(c3)
///    Ciphertext c4 = add(c3, c)
///    return sub(y, c4)
seal::Ciphertext encryptedQuadraticPolynomialPorcupine(
    seal::Ciphertext a_ctxt, seal::Ciphertext b_ctxt, seal::Ciphertext c_ctxt, seal::Ciphertext x_ctxt,
    seal::Ciphertext y_ctxt)
{
    // Ciphertext c1 = multiply(a, x)
    seal::Ciphertext c1;
    evaluator->multiply(a_ctxt, x_ctxt, c1);
    // c1 = relinearize(c1)
    evaluator->relinearize_inplace(c1, *relinkeys);
    // Ciphertext c2 = add(c1, b)
    seal::Ciphertext c2;
    evaluator->add(c1, b_ctxt, c2);
    // Ciphertext c3 = multiply(x, c2)
    seal::Ciphertext c3;
    evaluator->multiply(x_ctxt, c2, c3);
    // c3 = relinearize(c3)
    evaluator->relinearize_inplace(c3, *relinkeys);
    // Ciphertext c4 = add(c3, c)
    seal::Ciphertext c4;
    evaluator->add(c3, c_ctxt, c4);
    // return sub(y, c4)
    seal::Ciphertext result_ctxt;
    evaluator->sub(y_ctxt, c4, result_ctxt);
    return result_ctxt;
}