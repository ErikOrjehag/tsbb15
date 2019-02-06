#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CVL_LAB_IMAGEDIR=$DIR/images
export PYTHONPATH=$PYTHONPATH:$DIR/cvl_labs
