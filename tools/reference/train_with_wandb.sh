#!/bin/bash

# Example script for training with wandb logging
# This script demonstrates how to use wandb with DEIMv2 training

# Configuration
MODEL_SIZE="s"  # Options: s, m, l, x (for DINOv3) or n, s, m, l, x (for HGNetv2)
BACKBONE="dinov3"  # Options: dinov3, hgnetv2
DATASET="coco"
NUM_GPUS=4
MASTER_PORT=7777

# wandb Configuration
USE_WANDB=true
WANDB_PROJECT="deimv2-${DATASET}"
WANDB_ENTITY=""  # Set your wandb username or team name here
WANDB_RUN_NAME="${BACKBONE}_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"
WANDB_TAGS="baseline,${BACKBONE},${MODEL_SIZE}"
WANDB_NOTES="Training ${BACKBONE}-${MODEL_SIZE} on ${DATASET}"

# Training Configuration
CONFIG_FILE="configs/deimv2/deimv2_${BACKBONE}_${MODEL_SIZE}_${DATASET}.yml"
OUTPUT_DIR="output/${BACKBONE}_${MODEL_SIZE}_${DATASET}"
USE_AMP=true
SEED=0

# Construct GPU string
GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Model: ${BACKBONE}-${MODEL_SIZE}"
echo "Dataset: ${DATASET}"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: ${GPU_IDS} (${NUM_GPUS} GPUs)"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo "wandb Configuration"
echo "=========================================="
echo "Enabled: ${USE_WANDB}"
echo "Project: ${WANDB_PROJECT}"
echo "Run Name: ${WANDB_RUN_NAME}"
echo "Tags: ${WANDB_TAGS}"
echo "=========================================="

# Construct the training command
TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --master_port=${MASTER_PORT} --nproc_per_node=${NUM_GPUS} train.py"
TRAIN_CMD="${TRAIN_CMD} -c ${CONFIG_FILE}"
TRAIN_CMD="${TRAIN_CMD} --seed=${SEED}"
TRAIN_CMD="${TRAIN_CMD} --output-dir ${OUTPUT_DIR}"

# Add AMP if enabled
if [ "$USE_AMP" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use-amp"
fi

# Add wandb arguments if enabled
if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use-wandb"
    TRAIN_CMD="${TRAIN_CMD} --wandb-project ${WANDB_PROJECT}"
    TRAIN_CMD="${TRAIN_CMD} --wandb-run-name ${WANDB_RUN_NAME}"
    TRAIN_CMD="${TRAIN_CMD} --wandb-tags ${WANDB_TAGS}"
    
    if [ -n "$WANDB_ENTITY" ]; then
        TRAIN_CMD="${TRAIN_CMD} --wandb-entity ${WANDB_ENTITY}"
    fi
    
    if [ -n "$WANDB_NOTES" ]; then
        TRAIN_CMD="${TRAIN_CMD} --wandb-notes \"${WANDB_NOTES}\""
    fi
fi

echo ""
echo "Running command:"
echo "${TRAIN_CMD}"
echo ""

# Run the training command
eval ${TRAIN_CMD}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code $?"
    echo "=========================================="
    exit 1
fi

