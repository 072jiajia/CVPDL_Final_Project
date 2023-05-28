#!/bin/bash

PRETRAIN_DIR="pretrain-weights"

# Create the pretrain weights directory if it doesn't exist
if [ ! -d "$PRETRAIN_DIR" ]; then
    mkdir "$PRETRAIN_DIR"
fi

# Download pretrain weights if they don't exist
if [ ! -f "$PRETRAIN_DIR/sam_vit_b_01ec64.pth" ] && [ "$1" = "vit_b" ]; then
    wget -O "$PRETRAIN_DIR/sam_vit_b_01ec64.pth" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

if [ ! -f "$PRETRAIN_DIR/sam_vit_l_0b3195.pth" ] && [ "$1" = "vit_l" ]; then
    wget -O "$PRETRAIN_DIR/sam_vit_l_0b3195.pth" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
fi

if [ ! -f "$PRETRAIN_DIR/sam_vit_h_4b8939.pth" ] && ([ "$1" = "vit_h" ] || [ -z "$1" ]); then
    wget -O "$PRETRAIN_DIR/sam_vit_h_4b8939.pth" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# Run preprocess.py script with appropriate arguments
if [ "$1" = "vit_b" ]; then
    python preprocess.py --model-type vit_b --checkpoint-path "$PRETRAIN_DIR/sam_vit_b_01ec64.pth" --data_dir COCO2017/ --dataset COCO
elif [ "$1" = "vit_l" ]; then
    python preprocess.py --model-type vit_l --checkpoint-path "$PRETRAIN_DIR/sam_vit_l_0b3195.pth" --data_dir COCO2017/ --dataset COCO
else
    python preprocess.py --model-type vit_h --checkpoint-path "$PRETRAIN_DIR/sam_vit_h_4b8939.pth" --data_dir COCO2017/ --dataset COCO
fi
