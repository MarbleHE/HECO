#!/bin/bash

input_dir="evaluation/compile_time/heco_input"
output_dir="evaluation/compile_time/heco_output"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Loop through all files in the input directory
for input_file in "$input_dir"/*; do
    # Get the filename without the path
    filename=$(basename "$input_file")

    # Set the output file path
    output_file="$output_dir/$filename"

    # Run heco with the input file and store the output in the corresponding output file
    ./build/bin/heco --full-pass  -mlir-timing -mlir-timing-display=list "$input_file" > "$output_file"

    echo "Optimized: $input_file -> $output_file"
done