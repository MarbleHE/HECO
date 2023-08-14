OVERVIEW: HECO optimizer driver

Available Dialects: affine, arith, bfv, builtin, emitc, fhe, func, linalg, poly, tensor
USAGE: heco [options] <input file>

OPTIONS:

Color Options:

  --color                                              - Use colors in output (default=autodetect)

General options:

  --allow-unregistered-dialect                         - Allow operation with no registered dialects
  Compiler passes to run
    --pass-pipeline                                    -   A textual description of a pass pipeline to run
    Passes:
      --affine-loop-unroll                             -   Unroll affine loops
        --unroll-factor=<uint>                         - Use this unroll factor for all loops being unrolled
        --unroll-full                                  - Fully unroll loops
        --unroll-full-threshold=<uint>                 - Unroll all loops with trip count less than or equal to this
        --unroll-num-reps=<uint>                       - Unroll innermost loops repeatedly this many times
        --unroll-up-to-factor                          - Allow unrolling up to the factor specified
      --batching                                       -   
      --bfv2emitc                                      -   
      --bfv2llvm                                       -   
      --canonicalize                                   -   Canonicalize operations
        --disable-patterns=<string>                    - Labels of patterns that should be filtered out during application
        --enable-patterns=<string>                     - Labels of patterns that should be used during application, all other patterns are filtered out
        --max-iterations=<long>                        - Seed the worklist in general top-down order
        --region-simplify                              - Seed the worklist in general top-down order
        --top-down                                     - Seed the worklist in general top-down order
      --combine-simplify                               -   
      --cse                                            -   Eliminate common sub-expressions
      --fhe2bfv                                        -   
        --galois_keys_file=<string>                    - Name of the file containing the Galois keys.
        --params_file=<string>                         - Name of the paramter file defining, among other things, the ciphertext moduli
        --poly_mod_degree=<int>                        - Polynomial Degree of the Ciphertexts to assume.
        --relin_keys_file=<string>                     - Name of the file containing the relineriaztion keys.
      --internal-batching                              -   
      --lower-virtual                                  -   
      --nary                                           -   
      --scalar-batching                                -   
      --tensor2fhe                                     -   
      --unroll-loops                                   -   
    Pass Pipelines:
      --full-pass                                      -   Run all passes
      --hir-pass                                       -   Run HIR passes (including preprocessing)
      --preprocess                                     -   Run the preprocessing passes
  --mlir-debug-counter=<string>                        - Comma separated list of debug counter skip and count arguments
  --mlir-disable-threading                             - Disable multi-threading within MLIR, overrides any further call to MLIRContext::enableMultiThreading()
  --mlir-elide-elementsattrs-if-larger=<uint>          - Elide ElementsAttrs with "..." that have more elements than the given upper limit
  --mlir-pass-pipeline-crash-reproducer=<string>       - Generate a .mlir reproducer file at the given output path if the pass manager crashes or fails
  --mlir-pass-pipeline-local-reproducer                - When generating a crash reproducer, attempt to generated a reproducer with the smallest pipeline.
  --mlir-pass-statistics                               - Display the statistics of each pass
  --mlir-pass-statistics-display=<value>               - Display method for pass statistics
    =list                                              -   display the results in a merged list sorted by pass name
    =pipeline                                          -   display the results with a nested pipeline view
  --mlir-pretty-debuginfo                              - Print pretty debug info in MLIR output
  --mlir-print-debug-counter                           - Print out debug counter information after all counters have been accumulated
  --mlir-print-debuginfo                               - Print debug info in MLIR output
  --mlir-print-elementsattrs-with-hex-if-larger=<long> - Print DenseElementsAttrs with a hex string that have more elements than the given upper limit (use -1 to disable)
  --mlir-print-ir-after=<pass-arg>                     - Print IR after specified passes
  --mlir-print-ir-after-all                            - Print IR after each pass
  --mlir-print-ir-after-change                         - When printing the IR after a pass, only print if the IR changed
  --mlir-print-ir-after-failure                        - When printing the IR after a pass, only print if the pass failed
  --mlir-print-ir-before=<pass-arg>                    - Print IR before specified passes
  --mlir-print-ir-before-all                           - Print IR before each pass
  --mlir-print-ir-module-scope                         - When printing IR for print-ir-[before|after]{-all} always print the top-level operation
  --mlir-print-local-scope                             - Print with local scope and inline information (eliding aliases for attributes, types, and locations
  --mlir-print-op-on-diagnostic                        - When a diagnostic is emitted on an operation, also print the operation as an attached note
  --mlir-print-stacktrace-on-diagnostic                - When a diagnostic is emitted, also print the stack trace as an attached note
  --mlir-print-value-users                             - Print users of operation results and block arguments as a comment
  --mlir-timing                                        - Display execution times
  --mlir-timing-display=<value>                        - Display method for timing data
    =list                                              -   display the results in a list sorted by total time
    =tree                                              -   display the results ina with a nested tree view
  -o <filename>                                        - Output filename
  --show-dialects                                      - Print the list of registered dialects
  --split-input-file                                   - Split the input file into pieces and process each chunk independently
  --verify-diagnostics                                 - Check that emitted diagnostics match expected-* lines on the corresponding line
  --verify-each                                        - Run the verifier after each transformation pass

Generic Options:

  --help                                               - Display available options (--help-hidden for more)
  --help-list                                          - Display list of available options (--help-list-hidden for more)
  --version                                            - Display the version of this program
