#!/bin/bash
set -euo pipefail
# export WEIGHT=/nas_cv/walter_stuff/git/yolov5-master/auto_anno/countdown_anno_2/weights/best.pt
# export INPUT_DATASET=/big_daddy/nigel/modular_data_processing/ams/ams_od_1723_processed
# export RESULT_DIR=runs/detect/ams/ams_od_1723_processed
INPUT_DIR=$INPUT_DATASET
MODEL_WEIGHT=$WEIGHT
OUTPUT_DIR=$RESULT_DIR

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "$INPUT_DIR must be a directory"
    exit 1
fi


for i in "$INPUT_DIR"/*/*/; do
    python detect_imagr.py --weights "$MODEL_WEIGHT" --conf-thres 0.5 --save-crop --img-size 960 --source "$i/*_1920:1080.bayer_8.jpg" --name "$(basename "$i")" --project "$OUTPUT_DIR/$(basename "$(dirname "$i")")"
done