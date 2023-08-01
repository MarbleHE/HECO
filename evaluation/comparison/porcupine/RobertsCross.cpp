/// Output is squared to elide square root
/// Ciphertext roberts_cross(Ciphertext c0, int h, int w)
///     Ciphertext c1 = rotate(c0, w)
///     Ciphertext c2 = rotate(c0, 1)
///     Ciphertext c3 = sub(c1, c2)
///     Ciphertext c4 = square(c4)
///     c4 = relinearize(c4)
///     Ciphertext c5 = rotate(c0, w + 1)
///     Ciphertext c6 = sub(c5, c0)
///     Ciphertext c7 = square(c6)
///     c7 = relinearize(c7)
///     return add(c4, c7)
seal::Ciphertext encryptedRobertsCrossPorcupine(seal::Ciphertext img_ctxt, size_t size)
{
    // Ciphertext c1 = rotate(c0, w)
    seal::Ciphertext c1;
    evaluator->rotate_rows(img_ctxt, 1, *galoiskeys, c1);
    // Ciphertext c2 = rotate(c0, 1)
    seal::Ciphertext c2;
    evaluator->rotate_rows(img_ctxt, -size, *galoiskeys, c2);
    // Ciphertext c3 = sub(c1, c2)
    seal::Ciphertext c3;
    evaluator->sub(c1, c2, c3);
    // Ciphertext c4 = square(c4) //TODO: There is an error in the pseudo-code here
    seal::Ciphertext c4;
    evaluator->square(c3, c4);
    // c4 = relinearize(c4)
    evaluator->relinearize_inplace(c4, *relinkeys);
    // Ciphertext c5 = rotate(c0, w + 1)
    seal::Ciphertext c5;
    evaluator->rotate_rows(img_ctxt, -size + 1, *galoiskeys, c5);
    // Ciphertext c6 = sub(c5, c0)
    seal::Ciphertext c6;
    evaluator->sub(c5, img_ctxt, c6);
    // Ciphertext c7 = square(c6)
    seal::Ciphertext c7;
    evaluator->square(c6, c7);
    // c7 = relinearize(c7)
    evaluator->relinearize_inplace(c7, *relinkeys);
    // return add(c4, c7)
    seal::Ciphertext result_ctxt;
    evaluator->add(c4, c7, result_ctxt);

    return result_ctxt;
}
