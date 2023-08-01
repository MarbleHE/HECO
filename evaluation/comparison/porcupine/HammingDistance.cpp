/// For 4-element hamming distance
/// Ciphertext hamming_distance(Ciphertext c0, Ciphertext c1)
///     Plaintext p0(N, 2) // N is the number of slots
///     Ciphertext c2 = add(c1, c0)
///     Ciphertext c2_ = negate(c2)
///     Ciphertext c3 = add(c2_, p0)
///     Ciphertext c4 = multiply(c3, c2)
///     c4 = relinearize(c4)
///     Ciphertext c5 = rotate(c4, 2)
///     Ciphertext c6 = add(c4, c5)
///     Ciphertext c7 = rotate(c6, 1)
///     return add(c6, c7)
seal::Ciphertext encryptedHammingDistancePorcupine(
    const seal::Ciphertext &a_ctxt, const seal::Ciphertext &b_ctxt, size_t size)

{
    assert(size == 4 && "Porcupine Hamming Distance code only valid for vectors of lenght 4");

    // Plaintext p0(N, 2) // N is the number of slots
    std::vector<long> const_vector(encoder->slot_count(), 2);
    seal::Plaintext p0;
    encoder->encode(const_vector, p0);
    // Ciphertext c2 = add(c1, c0)
    seal::Ciphertext c2;
    evaluator->add(a_ctxt, b_ctxt, c2);
    // Ciphertext c2_ = negate(c2)
    seal::Ciphertext c2_;
    evaluator->negate(c2, c2_);
    // Ciphertext c3 = add(c2_, p0)
    seal::Ciphertext c3;
    evaluator->add_plain(c2_, p0, c3);
    // Ciphertext c4 = multiply(c3, c2)
    seal::Ciphertext c4;
    evaluator->multiply(c3, c2, c4);
    // c4 = relinearize(c4)
    evaluator->relinearize(c4, *relinkeys, c4);
    // Ciphertext c5 = rotate(c4, 2)
    seal::Ciphertext c5;
    evaluator->rotate_rows(c4, 2, *galoiskeys, c5);
    // Ciphertext c6 = add(c4, c5)
    seal::Ciphertext c6;
    evaluator->add(c4, c5, c6);
    // Ciphertext c7 = rotate(c6, 1)
    seal::Ciphertext c7;
    evaluator->rotate_rows(c6, 1, *galoiskeys, c7);
    // return add(c6, c7)
    seal::Ciphertext result_ctxt;
    evaluator->add(c6, c7, result_ctxt);

    return result_ctxt;
}