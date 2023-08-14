/// Porcupine version of dot product, which is based on the porcupine pseudocode. Only accepts vectors of length 8:
/// Ciphertext dot_product(Ciphertext c0, Plaintext p0)
///     Ciphertext c1 = multiply(c0, p0)
///     Ciphertext c2 = rotate(c1, 2)
///     Ciphertext c3 = add(c1, c2)
///     Ciphertext c4 = rotate(c3, 4)
///     Ciphertext c5 = add(c3, c4)
///     Ciphertext c6 = rotate(c5, 1)
///     return add(c5, c6)
/// \param x a vector of size n
/// \param y a vector of size n
seal::Ciphertext encryptedDotProductPorcupine(
    const seal::Ciphertext &x_ctxt, const seal::Ciphertext &y_ctxt, size_t size)
{
    assert(size == 8 && "Porcupine dotproduct is only valid for size 8");
    // Compute
    // Ciphertext c1 = multiply(c0, p0)
    seal::Ciphertext c1;
    evaluator->multiply(x_ctxt, y_ctxt, c1);
    evaluator->relinearize_inplace(c1, *relinkeys);
    // Ciphertext c2 = rotate(c1, 2)
    seal::Ciphertext c2;
    evaluator->rotate_rows(c1, 2, *galoiskeys, c2);
    // Ciphertext c3 = add(c1, c2)
    seal::Ciphertext c3;
    evaluator->add(c1, c2, c3);
    // Ciphertext c4 = rotate(c3, 4)
    seal::Ciphertext c4;
    evaluator->rotate_rows(c3, 4, *galoiskeys, c4);
    // Ciphertext c5 = add(c3, c4)
    seal::Ciphertext c5;
    evaluator->add(c3, c4, c5);
    // Ciphertext c6 = rotate(c5, 1)
    seal::Ciphertext c6;
    evaluator->rotate_rows(c5, 1, *galoiskeys, c6);
    // return add(c5, c6)
    seal::Ciphertext result_ctxt;
    evaluator->add(c5, c6, result_ctxt);

    return result_ctxt;
}