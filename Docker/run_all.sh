#!/bin/bash

#!/bin/bash

upload_file() {
    TARGET_DIR=$1
    shift
    for item in "$@"; do
        aws s3 cp $item ${S3_URL}/${S3_FOLDER}/${TARGET_DIR}/
    done  
}

# build ABC
mkdir /ABC/build && cd /ABC/build && cmake .. && make -j$(nproc)

# run the benchmark by passing the bench name as arg to program, then upload results to S3 bucket
echo "Running ABC demo..."
target_filename=demo_values.csv
/ABC/build/ast_demo $1 $target_filename \
    && upload_file demo $target_filename
