#!/bin/bash

LOG_DIR="run-log"

if [ ! -d "$LOG_DIR" ]; then
    mkdir "$LOG_DIR"
fi

for cid in $(seq 1 20); do
    for emb in 1 2 4; do
        python prompt_learning.py --class-index $cid --n-emb $emb | tee $LOG_DIR/class-$cid-n-emb-$emb.log
    done
done
