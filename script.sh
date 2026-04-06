#!/bin/bash

# Validate the model
python train.py \
    -c configs/deimv2/deimv2_dinov3_l_coco_custom_test.yml \
    -r outputs/deimv2_dinov3_l_coco_custom/best_stg2.pth \
    --test-only


# Test the model on single image
python tools/inference/torch_inf.py \
    -c configs/deimv2/deimv2_dinov3_l_coco_custom_test.yml \
    -r outputs/deimv2_dinov3_l_coco_custom/best_stg2.pth \
    --input images/pod-140.jpg \
    --device cuda:0


# Export tflite
python tools/deployment/export_tflite.py \
    -c configs/deimv2/deimv2_dinov3_l_coco_custom.yml \
    -r weights/deimv2_dinov3_l_coco_custom_best_stg2.pth \
    -o weights/deimv2_l.tflite