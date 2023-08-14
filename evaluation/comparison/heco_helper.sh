#!/bin/bash

input_dir="evaluation/comparison/heco_input"
output_dir="evaluation/comparison/heco_output"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Loop through all files in the input directory
for input_file in "$input_dir"/*; do
    # Get the filename without the path
    filename=$(basename "$input_file")

    # Set the output file path
    output_file="$output_dir/$filename"

    # Run heco with the input file and store the output in the corresponding output file
    ./build/bin/heco --full-pass "$input_file" > "$output_file"

    echo "Optimized: $input_file -> $output_file"
done

# Loop through all *.mlir files in the output directory
for output_file in "$output_dir"/*; do
    # Get the filename without the path
    filename=$(basename "$output_file")
    extension="${filename##*.}"
    filename="${filename%.*}"

    if [[ $extension == mlir ]]
    then
        # Set the output file path
        cpp_file="$output_dir/$filename.cpp"

        # Run emitc-translate with the optimized heco file and store the output in the corresponding *.cpp file
        ./build/bin/emitc-translate --mlir-to-cpp "$output_file" > "$cpp_file"

        echo "Translated: $output_file -> $cpp_file"
    fi
done