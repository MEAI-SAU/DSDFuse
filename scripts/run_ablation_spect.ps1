<#
.SYNOPSIS
MDiff-SFFuse Ablation Experiment Script for SPECT-MRI Dataset
.DESCRIPTION
Runs all ablation experiments for SPECT-MRI dataset
#>

Write-Host "========================================"
Write-Host "Starting SPECT-MRI ablation experiments..."
Write-Host "========================================"

# Define ablation modes
$ablationModes = @("baseline", "wo_backbone", "wo_bank", "wo_fusion_head", "wo_refinement", "full")
$configPrefix = "config/train_spect_mri_ablation_"

foreach ($mode in $ablationModes) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Running ablation mode: $mode"
    Write-Host "========================================"
    
    $configFile = "${configPrefix}${mode}.json"
    
    if (-not (Test-Path $configFile)) {
        Write-Host "Error: Config file $configFile not found!"
        exit 1
    }
    
    Write-Host "Config file: $configFile"
    
    # Train the model using FDFM conda environment
    Write-Host "Starting training..."
    $env:PYTHONUNBUFFERED = 1
    conda run -n FDFM python -u train.py -c $configFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Training failed for mode $mode"
        exit 1
    }
    
    # Test the model
    Write-Host "Training completed. Starting testing..."
    
    # Find the best model checkpoint
    $checkpointDir = "experiments/ablation/SPECT_MRI/${mode}/checkpoint"
    $bestModel = Get-ChildItem -Path $checkpointDir -Filter "*best_gen_G.pth" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if (-not $bestModel) {
        Write-Host "Warning: No best model found in $checkpointDir"
        Write-Host "Looking for any checkpoint..."
        $bestModel = Get-ChildItem -Path $checkpointDir -Filter "*_gen_G.pth" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    
    if ($bestModel) {
        $modelPath = $bestModel.FullName
        Write-Host "Testing with model: $modelPath"
        conda run -n FDFM python -u test-med-PET.py --model_path "$modelPath" --output_dir "experiments/ablation/SPECT_MRI/${mode}/fused_images" --save_metrics "experiments/ablation/SPECT_MRI/${mode}/metrics.csv" --dataset "SPECT-MRI"
    }
    else {
        Write-Host "Error: No checkpoint found for mode $mode"
    }
    
    Write-Host "Completed ablation mode: $mode"
    Write-Host "========================================"
}

Write-Host ""
Write-Host "========================================"
Write-Host "SPECT-MRI ablation experiments completed!"
Write-Host "========================================"