#include "seal/seal.h"
#include <cassert>
#include <memory>
#include "../util/BenchmarkHelper.h"
#include "../util/MultiTimer.h"

////////////////////////////////
// Include naive code here    //
////////////////////////////////
#include "naive/HammingDistance.cpp"
#include "naive/RobertsCross.cpp"

////////////////////////////////
// Include compiled code here //
////////////////////////////////
#include "heco_output/emitc_helper.h"
// ^- defines the non-member versions of evluator.*
#include "heco_output/hammingdistance_1024.cpp"
#include "heco_output/hammingdistance_16.cpp"
#include "heco_output/hammingdistance_256.cpp"
#include "heco_output/hammingdistance_4.cpp"
#include "heco_output/hammingdistance_4096.cpp"
#include "heco_output/hammingdistance_64.cpp"
#include "heco_output/robertscross_16x16.cpp"
#include "heco_output/robertscross_2x2.cpp"
#include "heco_output/robertscross_32x32.cpp"
#include "heco_output/robertscross_4x4.cpp"
#include "heco_output/robertscross_64x64.cpp"
#include "heco_output/robertscross_8x8.cpp"

////////////////////////////////
//    BENCHMARK FUNCTIONS     //
////////////////////////////////

// Number of iterations for the benchmark
// set to 10 for actual evaluation
#define ITER_COUNT 1

void HammingDistanceBench(size_t size, int version)
{
    // Generate Input vectors (technically, they should be boolean, but no impact on performance)
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
        std::cout << "Running BoxBlurBench (" << size << ") Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning BoxBlurBench (" << size << ") Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)            " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(8192);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning BoxBlurBench (" << size << ") Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)            " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto xs = encrypt_naive(x);
            auto ys = encrypt_naive(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning BoxBlurBench (" << size << ") Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)            " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedHammingDistanceNaive(xs, ys);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning BoxBlurBench (" << size << ") Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)            " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_single_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/benchmark/HammingDistance_Naive_" + std::to_string(size) + ".csv");
        // Decryption
        std::cout << "\rRunning BoxBlurBench (" << size << ") Naive Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)            " << std::endl;
        break;

    case 1:
        // HammingDistance HECO Version
        std::cout << "Running BoxBlurBench (" << size << ") HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)            " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(8192);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)            " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt_x = encrypt_batched(x);
            auto ctxt_y = encrypt_batched(y);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)            " << std::flush;
            auto evalTimer = timer.startTimer();
            seal::Ciphertext evaluated;
            switch (size)
            {
            case 4:
                evaluated = encryptedHammingDistance_4(ctxt_x, ctxt_y);
                break;
            case 16:
                evaluated = encryptedHammingDistance_16(ctxt_x, ctxt_y);
                break;
            case 64:
                evaluated = encryptedHammingDistance_64(ctxt_x, ctxt_y);
                break;
            case 256:
                evaluated = encryptedHammingDistance_256(ctxt_x, ctxt_y);
                break;
            case 1024:
                evaluated = encryptedHammingDistance_1024(ctxt_x, ctxt_y);
                break;
            case 4096:
                evaluated = encryptedHammingDistance_4096(ctxt_x, ctxt_y);
                break;

            default:
                assert(false && "Unsupported size for encrypted Hamming Distance Benchmark");
                break;
            }
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)            " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated)[0];
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/benchmark/HammingDistance_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)            " << std::endl;
        break;
    default:
        assert(false && "invalid version");
        break;
    }
}

void RobertsCrossBench(size_t size, int version)
{
    // Generate Input Image
    std::vector<int> int_img;
    getInputMatrix(size, int_img);
    std::vector<uint64_t> img = std::vector<uint64_t>(int_img.begin(), int_img.end());

    MultiTimer timer = MultiTimer();
    switch (version)
    {
    case 0:
        // RobertsCross Naive Version
        std::cout << "Running RobertsCrossBench (" << size << ") Naive Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning RobertsCrossBench (" << size << ") Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)            " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(8192);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning RobertsCrossBench (" << size << ") Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)            " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxts = encrypt_naive(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning RobertsCrossBench (" << size << ") Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)            " << std::flush;
            auto evalTimer = timer.startTimer();
            auto evaluated = encryptedRobertsCrossNaive(ctxts);
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning RobertsCrossBench (" << size << ") Naive Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)            " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_naive(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/benchmark/RobertsCross_Naive_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning RobertsCrossBench (" << size << ") Naive Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)            " << std::endl;
        break;
    case 1:
        // RobertsCross HECO Version
        std::cout << "Running RobertsCrossBench (" << size << ") HECO Version: " << std::flush;
        for (int i = 0; i < ITER_COUNT; ++i)
        {
            // KeyGen
            std::cout << "\rRunning RobertsCrossBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (KeyGen)            " << std::flush;
            auto keygenTimer = timer.startTimer();
            keygen(8192);
            timer.stopTimer(keygenTimer);

            // Encryption
            std::cout << "\rRunning RobertsCrossBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Encryption)            " << std::flush;
            auto encryptTimer = timer.startTimer();
            auto ctxt = encrypt_batched(img);
            timer.stopTimer(encryptTimer);

            // Evaluation
            std::cout << "\rRunning RobertsCrossBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Evaluation)            " << std::flush;
            auto evalTimer = timer.startTimer();
            seal::Ciphertext evaluated;
            switch (size)
            {
            case 2:
                evaluated = encryptedRobertsCross_2x2(ctxt);
                break;
            case 4:
                evaluated = encryptedRobertsCross_4x4(ctxt);
                break;
            case 8:
                evaluated = encryptedRobertsCross_8x8(ctxt);
                break;
            case 16:
                evaluated = encryptedRobertsCross_16x16(ctxt);
                break;
            case 32:
                evaluated = encryptedRobertsCross_32x32(ctxt);
                break;
            case 64:
                evaluated = encryptedRobertsCross_64x64(ctxt);
                break;

            default:
                assert(false && "Unsupported size for encrypted Hamming Distance Benchmark");
                break;
            }
            timer.stopTimer(evalTimer);

            // Decryption
            std::cout << "\rRunning RobertsCrossBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                      << " (Decryption)            " << std::flush;
            auto decryptTimer = timer.startTimer();
            auto result = decrypt_batched(evaluated);
            timer.stopTimer(decryptTimer);

            timer.addIteration();
        }
        timer.printToFile("evaluation/plotting/data/benchmark/RobertsCross_HECO_" + std::to_string(size) + ".csv");
        std::cout << "\rRunning RobertsCrossBench (" << size << ") HECO Version: " << ITER_COUNT << "/" << ITER_COUNT
                  << " (DONE)            " << std::endl;
        break;
    default:
        assert(false && "invalid version");
        break;
    }
}

int main()
{
    HammingDistanceBench(4, 0);
    HammingDistanceBench(16, 0);
    HammingDistanceBench(64, 0);
    HammingDistanceBench(256, 0);
    HammingDistanceBench(1024, 0);
    HammingDistanceBench(4096, 0);

    HammingDistanceBench(4, 1);
    HammingDistanceBench(16, 1);
    HammingDistanceBench(64, 1);
    HammingDistanceBench(256, 1);
    HammingDistanceBench(1024, 1);
    HammingDistanceBench(4096, 1);

    RobertsCrossBench(2, 0);
    RobertsCrossBench(4, 0);
    RobertsCrossBench(8, 0);
    RobertsCrossBench(16, 0);
    RobertsCrossBench(32, 0);
    RobertsCrossBench(64, 0);

    RobertsCrossBench(2, 1);
    RobertsCrossBench(4, 1);
    RobertsCrossBench(8, 1);
    RobertsCrossBench(16, 1);
    RobertsCrossBench(32, 1);
    RobertsCrossBench(64, 1);
}