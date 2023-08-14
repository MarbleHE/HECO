std::vector<seal::Ciphertext> encryptedBoxBlurNaive(const std::vector<seal::Ciphertext> &encImg)
{
    const auto imgSize = (int)std::ceil(std::sqrt(encImg.size()));
    std::vector<seal::Ciphertext> img2(encImg.begin(), encImg.end());

    // Horizontal Kernel: for each row y
    for (int y = 0; y < imgSize; ++y)
    {
        // Get kernel for first pixel of row y, using padding
        seal::Ciphertext value;
        evaluator->add(encImg.at((-1 * imgSize + y) % encImg.size()), encImg.at(0 * imgSize + y), value);
        // Division that would usually happen here is omitted
        img2[0 * imgSize + y] = value; // Is this gonna copy or just have the reference?

        // Go through the rest of row y
        for (int x = 1; x < imgSize; ++x)
        {
            // remove the previous pixel
            evaluator->sub(value, encImg.at(((x - 2) * imgSize + y) % encImg.size()), value);
            // add the new pixel
            evaluator->add(value, encImg.at((x * imgSize + y) % encImg.size()), value);
            // save result
            img2[x * imgSize + y] = value;
        }
    }

    // Now apply the vertical kernel to img2

    // Create new output image
    std::vector<seal::Ciphertext> img3(img2.begin(), img2.end());

    // Vertical Kernel: for each column x
    for (int x = 0; x < imgSize; ++x)
    {
        // Get kernel for first pixel of column x with padding
        seal::Ciphertext value;
        evaluator->add(img2.at((x * imgSize - 1) % encImg.size()), img2.at(x * imgSize + 0), value);
        // Division that would usually happen here is omitted
        img3[x * imgSize + 0] = value;

        // Go through the rest of column x
        for (int y = 1; y < imgSize; ++y)
        {
            // remove the previous pixel
            evaluator->sub(value, img2.at((x * imgSize + y - 2) % encImg.size()), value);
            // add the new pixel
            evaluator->add(value, img2.at((x * imgSize + y) % encImg.size()), value);
            // save result
            img3[x * imgSize + y] = value;
        }
    }
    return img3;
}