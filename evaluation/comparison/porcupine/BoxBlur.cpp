/// Encrypted BoxBlur, using the pseudocode given by the porcupine paper:
/// Ciphertext boxblur(Ciphertext c0, int h, int w)
///     Ciphertext c1 = rotate(c0, -1 * w)
///     Ciphertext c2 = add(c0, c1)
///     Ciphertext c3 = rotate(c2, -1)
///     return add(c2, c3)
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
seal::Ciphertext encryptedBoxBlurPorcupine(const seal::Ciphertext &c0, size_t size)
{
    // Ciphertext c1 = rotate(c0, -1 * w)
    seal::Ciphertext c1;
    evaluator->rotate_rows(c0, -1 * size, *galoiskeys, c1);

    // Ciphertext c2 = add(c0, c1)
    seal::Ciphertext c2;
    evaluator->add(c0, c1, c2);

    // Ciphertext c3 = rotate(c2, -1)
    seal::Ciphertext c3;
    evaluator->rotate_rows(c2, -1, *galoiskeys, c3);

    // return add(c2, c3)
    seal::Ciphertext result;
    evaluator->add(c2, c3, result);

    return result;
}
