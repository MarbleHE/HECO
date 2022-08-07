if (SEAL_FOUND)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")

    ## BOX BLUR ##
    set(BB_BENCH_FILES
            BoxBlur.cpp
            BoxBlurBench.cpp
            MultiTimer.cpp
            )
    add_executable(bb-bench ${BB_BENCH_FILES})
    target_include_directories(bb-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(bb-bench PUBLIC SEAL::seal)
    target_compile_definitions(bb-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(bb-bench PUBLIC -DHAVE_SEAL_CKKS)


    ## GX KERNEL ##
    set(GX_BENCH_FILES
            GxKernel.cpp
            GxKernelBench.cpp
            MultiTimer.cpp
            )
    add_executable(gx-bench ${GX_BENCH_FILES})
    target_include_directories(gx-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(gx-bench PUBLIC SEAL::seal)
    target_compile_definitions(gx-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(gx-bench PUBLIC -DHAVE_SEAL_CKKS)

    ## GY KERNEL ##
    set(GY_BENCH_FILES
            GyKernel.cpp
            GyKernelBench.cpp
            MultiTimer.cpp
            )
    add_executable(gy-bench ${GY_BENCH_FILES})
    target_include_directories(gy-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(gy-bench PUBLIC SEAL::seal)
    target_compile_definitions(gy-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(gy-bench PUBLIC -DHAVE_SEAL_CKKS)

    ## HAMMING DISTANCE ##
    set(HAM_BENCH_FILES
            HammingDistance.cpp
            HammingDistanceBench.cpp
            MultiTimer.cpp
            )
    add_executable(ham-bench ${HAM_BENCH_FILES})
    target_include_directories(ham-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(ham-bench PUBLIC SEAL::seal)
    target_compile_definitions(ham-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(ham-bench PUBLIC -DHAVE_SEAL_CKKS)

    ## ROBERTS CROSS ##
    set(RC_BENCH_FILES
            RobertsCross.cpp
            RobertsCrossBench.cpp
            MultiTimer.cpp
            )
    add_executable(rc-bench ${RC_BENCH_FILES})
    target_include_directories(rc-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(rc-bench PUBLIC SEAL::seal)
    target_compile_definitions(rc-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(rc-bench PUBLIC -DHAVE_SEAL_CKKS)

    ## LINEAR POLY ##
    set(LP_BENCH_FILES
            LinearPolynomial.cpp
            LinearPolynomialBench.cpp
            MultiTimer.cpp
    )
    add_executable(lp-bench ${LP_BENCH_FILES})
    target_include_directories(lp-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(lp-bench PUBLIC SEAL::seal)
    target_compile_definitions(lp-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(lp-bench PUBLIC -DHAVE_SEAL_CKKS)

    ## QUADRATIC POLY ##
    set(QP_BENCH_FILES
            QuadraticPolynomial.cpp
            QuadraticPolynomialBench.cpp
            MultiTimer.cpp
    )
    add_executable(qp-bench ${QP_BENCH_FILES})
    target_include_directories(qp-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(qp-bench PUBLIC SEAL::seal)
    target_compile_definitions(qp-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(qp-bench PUBLIC -DHAVE_SEAL_CKKS)

    ## LAPLACE SHARPENING ##
    set(RC_BENCH_FILES
            LaplaceSharpening.cpp
            LaplaceSharpeningBench.cpp
            MultiTimer.cpp
            )
    add_executable(ls-bench ${RC_BENCH_FILES})
    target_include_directories(ls-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(ls-bench PUBLIC SEAL::seal)
    target_compile_definitions(ls-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(ls-bench PUBLIC -DHAVE_SEAL_CKKS)

    ## L2 DISTANCE ##
    set(L2_BENCH_FILES
            L2Distance.cpp
            L2DistanceBench.cpp
            MultiTimer.cpp
            )
    add_executable(l2-bench ${L2_BENCH_FILES})
    target_include_directories(l2-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(l2-bench PUBLIC SEAL::seal)
    target_compile_definitions(l2-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(l2-bench PUBLIC -DHAVE_SEAL_CKKS)

    ## DOT PRODUCT ##
    set(DP_BENCH_FILES
            DotProduct.cpp
            DotProductBench.cpp
            MultiTimer.cpp
            )
    add_executable(dp-bench ${DP_BENCH_FILES})
    target_include_directories(dp-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(dp-bench PUBLIC SEAL::seal)
    target_compile_definitions(dp-bench PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(dp-bench PUBLIC -DHAVE_SEAL_CKKS)


endif (SEAL_FOUND)