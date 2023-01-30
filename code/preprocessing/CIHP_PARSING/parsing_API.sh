#!/bin/bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR"

cd $BASEDIR

python ./test_API.py $1

cd ../../
echo "Finish"