/// Encrypted GxKernel, using vectors of ciphertexts.
/// \param img Pixel (x,y) = (column, row) should be at position x*imgSize + y
/// \return transformed image
std::vector<seal::Ciphertext> encryptedGxKernelNaive(const std::vector<seal::Ciphertext> &img_ctxt)
{
    int img_size = (int)std::sqrt(img_ctxt.size());

    std::vector<seal::Ciphertext> img2(img_ctxt.size());

    // First apply [+1  0  -1]
    for (int y = 0; y < img_size; ++y)
    {
        // Get kernel for first pixel of row y, using padding
        seal::Ciphertext value;
        evaluator->sub(img_ctxt.at((-1 * img_size + y) % img_ctxt.size()), img_ctxt.at(1 * img_size + y), value);
        img2[0 * img_size + y] = value;

        // Go through the rest of row y
        for (int x = 1; x < img_size; ++x)
        {
            // remove the old leftmost pixel (old weight +1, now outside kernel)
            // x = middle of current kernel, x-2 = one to the left of kernel
            evaluator->sub_inplace(value, img_ctxt.at(((x - 2) * img_size + y) % img_ctxt.size()));

            // add the left pixel (old weight 0, new weight +1)
            // x = middle kernel, x-1 = left element of kernel
            evaluator->add_inplace(value, img_ctxt.at(((x - 1) * img_size + y) % img_ctxt.size()));

            // add the middle pixel to zero it out (old weight -1, new weight 0)
            // x = right pixel of previous kernel = middle pixel of new kernel
            evaluator->add_inplace(value, img_ctxt.at(((x)*img_size + y) % img_ctxt.size()));

            // finally, subtract the right most pixel (no old weight, new weight -1)
            // x = right pixel of previous kernel, x+1 = right pixel of new kernel
            evaluator->sub_inplace(value, img_ctxt.at(((x + 1) * img_size + y) % img_ctxt.size()));

            // save result
            img2[x * img_size + y] = value;
        }
    }

    // Now apply the vertical kernel to img2
    // | +1 |
    // | +2 |
    // | +1 |

    // Create new output image
    std::vector<seal::Ciphertext> img3(img2.begin(), img2.end());

    // Vertical Kernel: for each column x
    for (int x = 0; x < img_size; ++x)
    {
        // Get kernel for first pixel of column x with padding
        seal::Ciphertext value;
        seal::Ciphertext doublePixel;
        evaluator->add(img2.at(x * img_size + 0), img2.at(x * img_size + 0), value);
        evaluator->add_inplace(value, img2.at((x * img_size - 1) % img_ctxt.size()));
        evaluator->add_inplace(value, img2.at(x * img_size + 1));
        // Division that would usually happen here is omitted
        img3[x * img_size + 0] = value;

        // Go through the rest of column x
        for (int y = 1; y < img_size; ++y)
        {
            // remove the old leftmost pixel (old weight +1, now outside kernel)
            // y = middle of current kernel, y-2 = one to the left of kernel
            evaluator->sub_inplace(value, img2.at((x * img_size + y - 2) % img_ctxt.size()));

            // subtract the left pixel (old weight +2, new weight +1)
            // x = middle kernel, x-1 = left element of kernel
            evaluator->sub_inplace(value, img2.at((x * img_size + y - 1) % img_ctxt.size()));

            // add one copy of the middle pixel (old weight +1, new weight +2)
            // y = right pixel of previous kernel = middle pixel of new kernel
            evaluator->add_inplace(value, img2.at((x * img_size + y) % img_ctxt.size()));

            // finally, add the right most pixel (no old weight, new weight +1)
            // y = right pixel of previous kernel, y+1 = right pixel of new kernel
            evaluator->add_inplace(value, img2.at((x * img_size + y + 1) % img_ctxt.size()));

            // save result
            img3[x * img_size + y] = value;
        }
    }

    return img3;
}