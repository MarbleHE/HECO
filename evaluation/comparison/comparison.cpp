#include "seal/seal.h"
#include <cassert>
#include <memory>
#include "../util/BenchmarkHelper.h"
#include "../util/MultiTimer.h"

////////////////////////////////
// Include naive code here    //
////////////////////////////////
#include "naive/BoxBlur.cpp"
#include "naive/DotProduct.cpp"
#include "naive/GxKernel.cpp"
#include "naive/HammingDistance.cpp"
#include "naive/L2Distance.cpp"
#include "naive/LinearPolynomial.cpp"
#include "naive/QuadraticPolynomial.cpp"
#include "naive/RobertsCross.cpp"

////////////////////////////////
// Include porcupine code here//
////////////////////////////////
#include "porcupine/BoxBlur.cpp"
#include "porcupine/DotProduct.cpp"
#include "porcupine/GxKernel.cpp"
#include "porcupine/HammingDistance.cpp"
#include "porcupine/L2Distance.cpp"
#include "porcupine/LinearPolynomial.cpp"
#include "porcupine/QuadraticPolynomial.cpp"
#include "porcupine/RobertsCross.cpp"

////////////////////////////////
// Include compiled code here //
////////////////////////////////
#include "heco_output/emitc_helper.h"
// ^- defines the non-member versions of evluator.*
#include "heco_output/boxblur_64x64.cpp"
#include "heco_output/dotproduct_8.cpp"
#include "heco_output/gxkernel_64x64.cpp"
#include "heco_output/hammingdistance_4.cpp"
#include "heco_output/l2distance_4.cpp"
#include "heco_output/linearpolynomial_64.cpp"
#include "heco_output/quadraticpolynomial_64.cpp"
#include "heco_output/robertscross_64x64.cpp"

////////////////////////////////
//    BENCHMARK FUNCTIONS     //
////////////////////////////////

// Number of iterations for the benchmark
// set to 10 for actual evaluation
#define ITER_COUNT 1

void BoxBlurBench(size_t polymodulus_degree, int version)
{
    // Generate Input Image
    size_t size = 64; // 64x74 image
    std::vector<int> int_img;
    getInputMatrix(size, int_img);
    std::vector<uint64_t> img = std::vector<uint64_t>(int_img.begin(), int_img.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // BoxBlur Naive Version
        std::cout << "Running BoxBlurBench Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning BoxBlurBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (Keygen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning BoxBlurBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (Encryption)  "
                      << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxts = encrypt_naive(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning BoxBlurBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (Evaluation)  "
                      << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedBoxBlurNaive(ctxts);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning BoxBlurBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (Decryption)  "
                      << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/BoxBlur_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning BoxBlurBench Naive Version: " << ITER_COUNT << "/" << ITER_COUNT << " (DONE)        "
                  << std::endl;
        break;

    case 1:
        // BoxBlur Porcupine Version
        std::cout << "Running BoxBlurBench Porcupine Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning BoxBlurBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT << " (Keygen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning BoxBlurBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT << " (Encryption)  "
                      << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt = encrypt_batched(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning BoxBlurBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT << " (Evaluation)  "
                      << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedBoxBlurPorcupine(ctxt, size);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning BoxBlurBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT << " (Decryption)  "
                      << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/BoxBlur_Porcupine_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning BoxBlurBench Porcupine Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 2:
        // BoxBlur HECO Version
        std::cout << "Running BoxBlurBench HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning BoxBlurBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (Keygen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning BoxBlurBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (Encryption)  "
                      << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt = encrypt_batched(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning BoxBlurBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (Evaluation)  "
                      << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedBoxBlur_64x64(ctxt);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning BoxBlurBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (Decryption)  "
                      << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/BoxBlur_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning BoxBlurBench HECO Version: " << ITER_COUNT << "/" << ITER_COUNT << " (DONE)        "
                  << std::endl;
        break;

    default:
        assert(false && "invalid version number");
    }
}

void DotProductBench(size_t polymodulus_degree, int version)
{
    // Generate Input vectors
    size_t size = 8; // vectors of size 8
    std::vector<int> int_x(size);
    getRandomVector(int_x);
    std::vector<uint64_t> x = std::vector<uint64_t>(int_x.begin(), int_x.end());
    std::vector<int> int_y(size);
    getRandomVector(int_y);
    std::vector<uint64_t> y = std::vector<uint64_t>(int_y.begin(), int_y.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // DotProduct Naive Version
        std::cout << "Running DotProductBench Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning DotProductBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (Keygen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning DotProductBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto xs = encrypt_naive(x);
            auto ys = encrypt_naive(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning DotProductBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedDotProductNaive(xs, ys);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning DotProductBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_single_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/DotProduct_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning DotProductBench Naive Version: " << ITER_COUNT << "/" << ITER_COUNT << " (DONE)        "
                  << std::endl;
        break;

    case 1:
        // DotProduct Porcupine Version
        std::cout << "Running DotProductBench Porcupine Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning DotProductBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Keygen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning DotProductBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning DotProductBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedDotProductPorcupine(ctxt_x, ctxt_y, size);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning DotProductBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/DotProduct_Porcupine_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning DotProductBench Porcupine Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 2:
        // DotProduct HECO Version
        std::cout << "Running DotProductBench HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning DotProductBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (Keygen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning DotProductBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning DotProductBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedDotProduct_8(ctxt_x, ctxt_y);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning DotProductBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/DotProduct_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning DotProductBench HECO Version: " << ITER_COUNT << "/" << ITER_COUNT << " (DONE)        "
                  << std::endl;
        break;

    default:
        assert(false && "invalid version number");
    }
}

void GxKernelBench(size_t polymodulus_degree, int version)
{
    // Generate Input Image
    size_t size = 64; // 64x74 image
    std::vector<int> int_img;
    getInputMatrix(size, int_img);
    std::vector<uint64_t> img = std::vector<uint64_t>(int_img.begin(), int_img.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // GxKernel Naive Version
        std::cout << "Running GxKernelBench Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning GxKernelBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (KeyGen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning GxKernelBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (Encryption)     "
                      << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxts = encrypt_naive(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning GxKernelBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (Evaluation)     "
                      << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedGxKernelNaive(ctxts);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning GxKernelBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (Decryption)     "
                      << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/GxKernel_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning GxKernelBench Naive Version: " << ITER_COUNT << "/" << ITER_COUNT << " (DONE)        "
                  << std::endl;
        break;

    case 1:
        // GxKernel Porcupine Version
        std::cout << "Running GxKernelBench Porcupine Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning GxKernelBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT << " (KeyGen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning GxKernelBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt = encrypt_batched(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning GxKernelBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedGxKernelPorcupine(ctxt, size);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning GxKernelBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/GxKernel_Porcupine_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning GxKernelBench Porcupine Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 2:
        // GxKernel HECO Version
        std::cout << "Running GxKernelBench HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning GxKernelBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (KeyGen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning GxKernelBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (Encryption)     "
                      << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt = encrypt_batched(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning GxKernelBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (Evaluation)     "
                      << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedGxKernel_64x64(ctxt);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning GxKernelBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (Decryption)     "
                      << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/GxKernel_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning GxKernelBench HECO Version: " << ITER_COUNT << "/" << ITER_COUNT << " (DONE)        "
                  << std::endl;
        break;

    default:
        assert(false && "invalid version number");
    }
}

void HammingDistanceBench(size_t polymodulus_degree, int version)
{
    // Generate Input vectors (technically, they should be boolean, but no impact on performance)
    size_t size = 4; // vectors of size 4
    std::vector<int> int_x(size);
    getRandomVector(int_x);
    std::vector<uint64_t> x = std::vector<uint64_t>(int_x.begin(), int_x.end());
    std::vector<int> int_y(size);
    getRandomVector(int_y);
    std::vector<uint64_t> y = std::vector<uint64_t>(int_y.begin(), int_y.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // HammingDistance Naive Version
        std::cout << "Running HammingDistanceBench Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning HammingDistanceBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning HammingDistanceBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto xs = encrypt_naive(x);
            auto ys = encrypt_naive(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning HammingDistanceBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedHammingDistanceNaive(xs, ys);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning HammingDistanceBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_single_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/HammingDistance_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning HammingDistanceBench Naive Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 1:
        // HammingDistance Porcupine Version
        std::cout << "Running HammingDistanceBench Porcupine Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning HammingDistanceBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning HammingDistanceBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning HammingDistanceBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedHammingDistancePorcupine(ctxt_x, ctxt_y, size);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning HammingDistanceBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile(
            "evaluation/plotting/data/comparison/HammingDistance_Porcupine_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning HammingDistanceBench Porcupine Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 2:
        // HammingDistance HECO Version
        std::cout << "Running HammingDistanceBench HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning HammingDistanceBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning HammingDistanceBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning HammingDistanceBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedHammingDistance_4(ctxt_x, ctxt_y);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning HammingDistanceBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/HammingDistance_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning HammingDistanceBench HECO Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    default:
        assert(false && "invalid version number");
    }
}

void L2DistanceBench(size_t polymodulus_degree, int version)
{
    // Generate Input vectors
    size_t size = 4; // vectors of size 4
    std::vector<int> int_x(size);
    getRandomVector(int_x);
    std::vector<uint64_t> x = std::vector<uint64_t>(int_x.begin(), int_x.end());
    std::vector<int> int_y(size);
    getRandomVector(int_y);
    std::vector<uint64_t> y = std::vector<uint64_t>(int_y.begin(), int_y.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // L2Distance Naive Version
        std::cout << "Running L2DistanceBench Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning L2DistanceBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (KeyGen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning L2DistanceBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto xs = encrypt_naive(x);
            auto ys = encrypt_naive(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning L2DistanceBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedL2DistanceNaive(xs, ys);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning L2DistanceBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_single_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/L2Distance_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning L2DistanceBench Naive Version: " << ITER_COUNT << "/" << ITER_COUNT << " (DONE)        "
                  << std::endl;
        break;

    case 1:
        // L2Distance Porcupine Version
        std::cout << "Running L2DistanceBench Porcupine Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning L2DistanceBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning L2DistanceBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning L2DistanceBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedL2DistancePorcupine(ctxt_x, ctxt_y, size);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning L2DistanceBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/L2Distance_Porcupine_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning L2DistanceBench Porcupine Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 2:
        // L2Distance HECO Version
        std::cout << "Running L2DistanceBench HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning L2DistanceBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (KeyGen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning L2DistanceBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning L2DistanceBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedL2Distance_4(ctxt_x, ctxt_y);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning L2DistanceBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/L2Distance_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning L2DistanceBench HECO Version: " << ITER_COUNT << "/" << ITER_COUNT << " (DONE)        "
                  << std::endl;
        break;

    default:
        assert(false && "invalid version number");
    }
}

void LinearPolynomialBench(size_t polymodulus_degree, int version)
{
    // Generate Input vectors
    size_t size = 64; // vectors of length 64
    std::vector<int> int_a(size);
    getRandomVector(int_a);
    std::vector<uint64_t> a = std::vector<uint64_t>(int_a.begin(), int_a.end());
    std::vector<int> int_b(size);
    getRandomVector(int_b);
    std::vector<uint64_t> b = std::vector<uint64_t>(int_b.begin(), int_b.end());
    std::vector<int> int_x(size);
    getRandomVector(int_x);
    std::vector<uint64_t> x = std::vector<uint64_t>(int_x.begin(), int_x.end());
    std::vector<int> int_y(size);
    getRandomVector(int_y);
    std::vector<uint64_t> y = std::vector<uint64_t>(int_y.begin(), int_y.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // LinearPolynomial Naive Version
        std::cout << "Running LinearPolynomialBench Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning LinearPolynomialBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning LinearPolynomialBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto as = encrypt_naive(a);
            auto bs = encrypt_naive(b);
            auto xs = encrypt_naive(x);
            auto ys = encrypt_naive(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning LinearPolynomialBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedLinearPolynomialNaive(as, bs, xs, ys);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning LinearPolynomialBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile(
            "evaluation/plotting/data/comparison/LinearPolynomial_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning LinearPolynomialBench Naive Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 1:
        // LinearPolynomial Porcupine Version
        std::cout << "Running LinearPolynomialBench Porcupine Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning LinearPolynomialBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning LinearPolynomialBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_a = encrypt_batched(a);
            auto ctxt_b = encrypt_batched(b);
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning LinearPolynomialBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedLinearPolynomialPorcupine(ctxt_a, ctxt_b, ctxt_x, ctxt_y);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning LinearPolynomialBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile(
            "evaluation/plotting/data/comparison/LinearPolynomial_Porcupine_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning LinearPolynomialBench Porcupine Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 2:
        // LinearPolynomial HECO Version
        std::cout << "Running LinearPolynomialBench HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning LinearPolynomialBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning LinearPolynomialBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_a = encrypt_batched(a);
            auto ctxt_b = encrypt_batched(b);
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning LinearPolynomialBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedLinearPolynomial_64(ctxt_a, ctxt_b, ctxt_x, ctxt_y);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning LinearPolynomialBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/LinearPolynomial_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning LinearPolynomialBench HECO Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    default:
        assert(false && "invalid version number");
    }
}

void QuadraticPolynomialBench(size_t polymodulus_degree, int version)
{
    // Generate Input vectors
    size_t size = 64; // vectors of length 64
    std::vector<int> int_a(size);
    getRandomVector(int_a);
    std::vector<uint64_t> a = std::vector<uint64_t>(int_a.begin(), int_a.end());
    std::vector<int> int_b(size);
    getRandomVector(int_b);
    std::vector<uint64_t> b = std::vector<uint64_t>(int_b.begin(), int_b.end());
    std::vector<int> int_c(size);
    getRandomVector(int_c);
    std::vector<uint64_t> c = std::vector<uint64_t>(int_c.begin(), int_c.end());
    std::vector<int> int_x(size);
    getRandomVector(int_x);
    std::vector<uint64_t> x = std::vector<uint64_t>(int_x.begin(), int_x.end());
    std::vector<int> int_y(size);
    getRandomVector(int_y);
    std::vector<uint64_t> y = std::vector<uint64_t>(int_y.begin(), int_y.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // QuadraticPolynomial Naive Version
        std::cout << "Running QuadraticPolynomialBench Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning QuadraticPolynomialBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning QuadraticPolynomialBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto as = encrypt_naive(a);
            auto bs = encrypt_naive(b);
            auto cs = encrypt_naive(c);
            auto xs = encrypt_naive(x);
            auto ys = encrypt_naive(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning QuadraticPolynomialBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedQuadraticPolynomialNaive(as, bs, cs, xs, ys);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning QuadraticPolynomialBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile(
            "evaluation/plotting/data/comparison/QuadraticPolynomial_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning QuadraticPolynomialBench Naive Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 1:
        // QuadraticPolynomial Porcupine Version
        std::cout << "Running QuadraticPolynomialBench Porcupine Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning QuadraticPolynomialBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning QuadraticPolynomialBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_a = encrypt_batched(a);
            auto ctxt_b = encrypt_batched(b);
            auto ctxt_c = encrypt_batched(c);
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning QuadraticPolynomialBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedQuadraticPolynomialPorcupine(ctxt_a, ctxt_b, ctxt_c, ctxt_x, ctxt_y);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning QuadraticPolynomialBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile(
            "evaluation/plotting/data/comparison/QuadraticPolynomial_Porcupine_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning QuadraticPolynomialBench Porcupine Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 2:
        // QuadraticPolynomial HECO Version
        std::cout << "Running QuadraticPolynomialBench HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning QuadraticPolynomialBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning QuadraticPolynomialBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_a = encrypt_batched(a);
            auto ctxt_b = encrypt_batched(b);
            auto ctxt_c = encrypt_batched(c);
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning QuadraticPolynomialBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedQuadraticPolynomial_64(ctxt_a, ctxt_b, ctxt_c, ctxt_x, ctxt_y);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning QuadraticPolynomialBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile(
            "evaluation/plotting/data/comparison/QuadraticPolynomial_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning QuadraticPolynomialBench HECO Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    default:
        assert(false && "invalid version number");
    }
}

void RobertsCrossBench(size_t polymodulus_degree, int version)
{
    // Generate Input Image
    size_t size = 64; // 64x74 image
    std::vector<int> int_img;
    getInputMatrix(size, int_img);
    std::vector<uint64_t> img = std::vector<uint64_t>(int_img.begin(), int_img.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // RobertsCross Naive Version
        std::cout << "Running RobertsCrossBench Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning RobertsCrossBench Naive Version: " << i + 1 << "/" << ITER_COUNT << " (KeyGen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning RobertsCrossBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxts = encrypt_naive(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning RobertsCrossBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedRobertsCrossNaive(ctxts);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning RobertsCrossBench Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/RobertsCross_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning RobertsCrossBench Naive Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 1:
        // RobertsCross Porcupine Version
        std::cout << "Running RobertsCrossBench Porcupine Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning RobertsCrossBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)     " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning RobertsCrossBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt = encrypt_batched(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning RobertsCrossBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedRobertsCrossPorcupine(ctxt, size);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning RobertsCrossBench Porcupine Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile(
            "evaluation/plotting/data/comparison/RobertsCross_Porcupine_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning RobertsCrossBench Porcupine Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    case 2:
        // RobertsCross HECO Version
        std::cout << "Running RobertsCrossBench HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning RobertsCrossBench HECO Version: " << i + 1 << "/" << ITER_COUNT << " (KeyGen)     "
                      << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(polymodulus_degree);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning RobertsCrossBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)     " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt = encrypt_batched(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning RobertsCrossBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)     " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedRobertsCross_64x64(ctxt);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning RobertsCrossBench HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)     " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/comparison/RobertsCross_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning RobertsCrossBench HECO Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)        " << std::endl;
        break;

    default:
        assert(false && "invalid version number");
    }
}

int main()
{
    BoxBlurBench(8192, 0);
    BoxBlurBench(8192, 1);
    BoxBlurBench(8192, 2);

    DotProductBench(8192, 0);
    DotProductBench(8192, 1);
    DotProductBench(8192, 2);

    GxKernelBench(8192, 0);
    GxKernelBench(8192, 1);
    GxKernelBench(8192, 2);

    HammingDistanceBench(8192, 0);
    HammingDistanceBench(8192, 1);
    HammingDistanceBench(8192, 2);

    L2DistanceBench(8192, 0);
    L2DistanceBench(8192, 1);
    L2DistanceBench(8192, 2);

    LinearPolynomialBench(8192, 0);
    LinearPolynomialBench(8192, 1);
    LinearPolynomialBench(8192, 2);

    QuadraticPolynomialBench(8192, 0);
    QuadraticPolynomialBench(8192, 1);
    QuadraticPolynomialBench(8192, 2);

    RobertsCrossBench(8192, 0);
    RobertsCrossBench(8192, 1);
    RobertsCrossBench(8192, 2);
}