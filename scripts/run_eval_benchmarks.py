#!/usr/bin/python3

import subprocess
import json
import os 
from datetime import datetime

output_directory = "/home/ubuntu/benchmark_results"
json_configs = "/home/ubuntu/benchmark_configs"

configs = [
    # {
    #     "program": "OPT_RV",
    #     "imgSizes": [8, 16, 32, 64, 96, 128],
    #     "testruns": 10
    # },
    # {
    #     "program": "OPT_SEAL",
    #     "imgSizes": [8, 16, 32, 64, 96, 128],
    #     "testruns": 10
    # },
    {
        "program": "UNOPT_SEAL",
        "imgSizes": [96],
        "testruns": 1
    },
    {
        "program": "UNOPT_SEAL",
        "imgSizes": [128],
        "testruns": 1
    },
    {
        "program": "UNOPT_SEAL",
        "imgSizes": [8, 16, 32, 64, 96, 128],
        "testruns": 1
    }
]

benchmarkJsonTemplate = {
    "num_testruns": 1,
    "image_sizes" : [
    ],
    "programs" : [
    ],
    "result_files_directory": output_directory
}

__curDir__ = os.path.dirname(os.path.realpath(__file__))

for c in configs:
    for imgSize in c["imgSizes"]:
        testruns = c["testruns"]
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y%m%d_%H%M%S")

        config_file_path = os.path.join(json_configs, timestampStr + '_benchmark_config.json')

        benchmarkJsonTemplate["programs"] = [c["program"]]
        benchmarkJsonTemplate["image_sizes"] = [imgSize]

        with open(config_file_path, 'w') as outfile:
            json.dump(benchmarkJsonTemplate, outfile)

        print(config_file_path)

        while testruns > 0:
            os.chdir("/home/ubuntu/AST-Optimizer/build")
            cmakeCmd = ['./ast_benchmarkRuntime', config_file_path]
            retCode = subprocess.check_call(cmakeCmd, stderr=subprocess.STDOUT)
            
            testruns = testruns-1