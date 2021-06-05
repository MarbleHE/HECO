#!/bin/bash

# check out the ABC repo
git clone https://github.com/MarbleHE/ABC.git /ABC \
    && cd /ABC/Docker \
    && chmod +x run_all.sh \
    && ./run_all.sh

