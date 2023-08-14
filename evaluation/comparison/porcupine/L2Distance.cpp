/// Output is squared to elide square root
/// For 4-element distance
/// Ciphertext l2_distance(Ciphertext c0, Ciphertext c1)
///     Ciphertext c2 = sub(c1, c0)
///     Ciphertext c3 = square(c2)
///     c3 = relinearize(c3)
///     Ciphertext c4 = rotate(c3, 2)
///     Ciphertext c5 = add(c3, c4)
///     Ciphertext c6 = rotate(c4, 1)
///     return add(c5, c6)
seal::Ciphertext encryptedL2DistancePorcupine(seal::Ciphertext x_ctxt, seal::Ciphertext y_ctxt, size_t size)
{
    assert(size == 4 && "Porcupine L2Distance code only valid for vector size 4");

    // Ciphertext c2 = sub(c1, c0)
    seal::Ciphertext c2;
    evaluator->sub(y_ctxt, x_ctxt, c2);
    // Ciphertext c3 = square(c2)
    seal::Ciphertext c3;
    evaluator->square(c2, c3);
    // c3 = relinearize(c3)
    evaluator->relinearize(c3, *relinkeys, c3);

    // Ciphertext c4 = rotate(c3, 2)
    seal::Ciphertext c4;
    evaluator->rotate_rows(c3, 2, *galoiskeys, c4);
    // Ciphertext c5 = add(c3, c4)
    seal::Ciphertext c5;
    evaluator->add(c3, c4, c5);
    // Ciphertext c6 = rotate(c4, 1)
    // TODO: The above seems like a mistake in the porcupien code, so I changed it and now things work.
    seal::Ciphertext c6;
    evaluator->rotate_rows(c5, 1, *galoiskeys, c6);
    // return add(c5, c6)
    seal::Ciphertext result_ctxt;
    evaluator->add(c5, c6, result_ctxt);

    return result_ctxt;
}