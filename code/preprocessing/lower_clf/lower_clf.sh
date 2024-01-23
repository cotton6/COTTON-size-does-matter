#!/bin/bash

BASEDIR=$(dirname "$0")
# echo "$BASEDIR"
cd $BASEDIR

python main.py \
--mode test \
--input_dir ../../$1 \
--output_dir ../../$1"_info" 

cd ../../
# echo "Finish ATR parsing"