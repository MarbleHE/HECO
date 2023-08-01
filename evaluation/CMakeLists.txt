add_executable(comparison 
    comparison/comparison.cpp
    util/MultiTimer.cpp
)

target_link_libraries(comparison SEAL::seal)


add_executable(benchmark 
    benchmark/benchmark.cpp
    util/MultiTimer.cpp
)

target_link_libraries(benchmark SEAL::seal)