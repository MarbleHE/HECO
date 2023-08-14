/// Encrypted GxKernel, ported from Porcupine, according to:
/// Ciphertext gx(Ciphertext c0, int h, int w)
///     Ciphertext c1 = rotate(c0, w)
///     Ciphertext c2 = add(c0, c1)
///     Ciphertext c3 = rotate(c2, -w)
///     Ciphertext c4 = add(c2, c3)
///     Ciphertext c5 = rotate(c4, 1)
///     Ciphertext c6 = rotate(c4, -1)
///     return sub(c5, c6)
/// Currently, this requires the image vector to be n/2 long,
/// so we don't run into issues with rotations.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
seal::Ciphertext encryptedGxKernelPorcupine(seal::Ciphertext img_ctxt, size_t img_size)
{
    // Ciphertext c1 = rotate(c0, w)
    seal::Ciphertext c1;
    evaluator->rotate_rows(img_ctxt, -1, *galoiskeys, c1); // img_ctxt == c0
    // Ciphertext c2 = add(c0, c1)
    seal::Ciphertext c2;
    evaluator->add(img_ctxt, c1, c2);
    // evaluator->add(img_ctxt, c1, c1); // c1 == c2
    // Ciphertext c3 = rotate(c2, -w)
    seal::Ciphertext c3;
    evaluator->rotate_rows(c2, 1, *galoiskeys, c3);
    // evaluator->rotate_rows(c1, -1 * img_size, *galoiskeys, img_ctxt); // img_ctxt == c3
    // Ciphertext c4 = add(c2, c3)
    seal::Ciphertext c4;
    evaluator->add(c2, c3, c4);
    // evaluator->add(c1, img_ctxt, c1); // c1 == c4
    // Ciphertext c5 = rotate(c4, 1)
    seal::Ciphertext c5;
    evaluator->rotate_rows(c4, -1 * img_size, *galoiskeys, c5);
    // evaluator->rotate_rows(c1, 1, *galoiskeys, img_ctxt); // img_ctxt == c5
    // Ciphertext c6 = rotate(c4, -1)
    seal::Ciphertext c6;
    evaluator->rotate_rows(c4, img_size, *galoiskeys, c6);
    // evaluator->rotate_rows_inplace(c1, -1, *galoiskeys); // c1 == c6
    // return sub(c5, c6)
    seal::Ciphertext result_ctxt;
    evaluator->sub(c5, c6, result_ctxt);
    // evaluator->sub(img_ctxt, c1, img_ctxt);

    return result_ctxt;
}