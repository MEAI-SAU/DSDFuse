#!/bin/bash
# MDiff-SFFuse Ablation Experiment Script for PET-MRI Dataset
# Usage: bash scripts/run_ablation_pet.sh

echo "Starting PET-MRI ablation experiments..."
echo "========================================"

# Set environment
export CUDA_VISIBLE_DEVICES=0

# Define ablation modes
ABLATION_MODES=("baseline" "wo_backbone" "wo_bank" "wo_fusion_head" "wo_refinement" "full")
CONFIG_PREFIX="config/train_pet_mri_ablation_"

for mode in "${ABLATION_MODES[@]}"; do
    echo ""
    echo "========================================"
    echo "Running ablation mode: $mode"
    echo "========================================"
    
    CONFIG_FILE="${CONFIG_PREFIX}${mode}.json"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file $CONFIG_FILE not found!"
        exit 1
    fi
    
    echo "Config file: $CONFIG_FILE"
    
    # Train the model
    echo "Starting training..."
    python train.py -c "$CONFIG_FILE"
    
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for mode $mode"
        exit 1
    fi
    
    # Test the model
    echo "Training completed. Starting testing..."
    
    # Find the best model checkpoint
    CHECKPOINT_DIR="experiments/ablation/PET_MRI/${mode}/checkpoint"
    BEST_MODEL=$(ls -t "${CHECKPOINT_DIR}"/*best_gen_G.pth 2>/dev/null | head -1)
    
    if [ -z "$BEST_MODEL" ]; then
        echo "Warning: No best model found in $CHECKPOINT_DIR"
        echo "Looking for any checkpoint..."
        BEST_MODEL=$(ls -t "${CHECKPOINT_DIR}"/*_gen_G.pth 2>/dev/null | head -1)
    fi
    
    if [ -n "$BEST_MODEL" ]; then
        echo "Testing with model: $BEST_MODEL"
        python test-med-PET.py --model_path "$BEST_MODEL" --output_dir "experiments/ablation/PET_MRI/${mode}/fused_images" --save_metrics "experiments/ablation/PET_MRI/${mode}/metrics.csv"
    else
        echo "Error: No checkpoint found for mode $mode"
    fi
    
    echo "Completed ablation mode: $mode"
    echo "========================================"
done

echo ""
echo "========================================"
echo "PET-MRI ablation experiments completed!"
echo "========================================"