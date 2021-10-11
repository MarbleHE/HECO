cmake_minimum_required(VERSION 3.11.0)

## BOX BLUR ##
if (SEAL_FOUND)
    set(BB_BENCH_FILES
            BoxBlur.cpp
            BoxBlurBench.cpp
            MultiTimer.cpp
            )
    add_executable(bb-bench ${BB_BENCH_FILES})
    target_include_directories(bb-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(bb-bench PUBLIC SEAL::seal)


    ## GX KERNEL ##
    set(GX_BENCH_FILES
            GxKernel.cpp
            GxKernelBench.cpp
            MultiTimer.cpp
            )
    add_executable(gx-bench ${GX_BENCH_FILES})
    target_include_directories(gx-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(gx-bench PUBLIC SEAL::seal)

    ## HAMMING DISTANCE ##
    set(HAM_BENCH_FILES
            HammingDistance.cpp
            HammingDistanceBench.cpp
            MultiTimer.cpp
            )
    add_executable(ham-bench ${HAM_BENCH_FILES})
    target_include_directories(ham-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(ham-bench PUBLIC SEAL::seal)

    ## ROBERTS CROSS ##
    set(RC_BENCH_FILES
            RobertsCross.cpp
            RobertsCrossBench.cpp
            MultiTimer.cpp
            )
    add_executable(rc-bench ${RC_BENCH_FILES})
    target_include_directories(rc-bench PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(rc-bench PUBLIC SEAL::seal)

endif (SEAL_FOUND)