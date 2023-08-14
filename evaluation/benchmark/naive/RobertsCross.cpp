std::vector<seal::Ciphertext> encryptedRobertsCrossNaive(std::vector<seal::Ciphertext> img_ctxt)
{
    int img_size = (int)std::sqrt(img_ctxt.size());

    std::vector<seal::Ciphertext> result_ctxt(img_ctxt.size());

    // Each point p = img[x][y], where x is row and y is column, in the new image will equal:
    //   (img[x-1][y-1] - img[x][y])^2 + (img[x-1][y] - img[x][y-1])^2
    for (int x = 0; x < img_size; ++x)
    {
        for (int y = 0; y < img_size; ++y)
        {
            seal::Ciphertext tmp1;
            auto index1 = ((y - 1) * img_size + x + 1) % img_ctxt.size();
            auto index2 = y * img_size + x;
            evaluator->sub(img_ctxt[index1], img_ctxt[index2], tmp1);
            evaluator->square_inplace(tmp1);
            evaluator->relinearize_inplace(tmp1, *relinkeys);

            seal::Ciphertext tmp2;
            auto index3 = (y * img_size + x + 1) % img_ctxt.size();
            auto index4 = ((y - 1) * img_size + x) % img_ctxt.size();
            evaluator->sub(img_ctxt[index3], img_ctxt[index4], tmp2);
            evaluator->square_inplace(tmp2);
            evaluator->relinearize_inplace(tmp2, *relinkeys);

            evaluator->add(tmp1, tmp2, result_ctxt[y * img_size + x]);
        }
    }

    return result_ctxt;
}