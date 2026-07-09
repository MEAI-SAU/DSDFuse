#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSDFuse Ablation Experiment Summary Generator
Generates CSV and LaTeX tables from ablation experiment results.
"""

import os
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate ablation experiment summary')
    parser.add_argument('--pet_mri_dir', type=str, default='experiments/ablation/PET_MRI',
                        help='Path to PET-MRI ablation results')
    parser.add_argument('--spect_mri_dir', type=str, default='experiments/ablation/SPECT_MRI',
                        help='Path to SPECT-MRI ablation results')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for summary files')
    return parser.parse_args()

def read_metrics(csv_path):
    """Read metrics from a CSV file."""
    metrics = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                variant = row.get('Variant', row.get('variant', 'Unknown'))
                metrics[variant] = {
                    'MI': float(row.get('MI', row.get('mi', 0))),
                    'VIF': float(row.get('VIF', row.get('vif', 0))),
                    'Qabf': float(row.get('Qabf', row.get('qabf', 0))),
                    'SSIM': float(row.get('SSIM', row.get('ssim', 0))),
                    'FMI': float(row.get('FMI', row.get('FMI_pixel', row.get('fmi_pixel', row.get('fmi', 0))))),
                    'SCD': float(row.get('SCD', row.get('scd', 0))),
                }
    return metrics

def collect_ablation_results(base_dir, ablation_modes):
    """Collect results for all ablation modes."""
    results = {}
    for mode in ablation_modes:
        mode_dir = os.path.join(base_dir, mode)
        csv_path = os.path.join(mode_dir, 'metrics.csv')
        
        if os.path.exists(csv_path):
            metrics = read_metrics(csv_path)
            # Use the mode name as the variant
            if metrics:
                # Find the matching variant or use mode name
                variant_names = list(metrics.keys())
                if mode.lower() in [v.lower() for v in variant_names]:
                    for v in variant_names:
                        if v.lower() == mode.lower():
                            results[mode] = metrics[v]
                            break
                elif variant_names:
                    results[mode] = metrics[variant_names[0]]
                else:
                    results[mode] = None
            else:
                results[mode] = None
        else:
            results[mode] = None
    return results

def write_csv(results, output_path, dataset_name):
    """Write results to CSV file."""
    # Define the order of variants
    variant_order = [
        ('baseline', 'Baseline'),
        ('wo_backbone', 'w/o Multi-domain Backbone'),
        ('wo_bank', 'w/o Diffusion Process Feature Bank'),
        ('wo_fusion_head', 'w/o Step-wise Fusion Head'),
        ('wo_refinement', 'w/o Reliability Refinement'),
        ('full', 'Full DSDFuse'),
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant', 'MI', 'VIF', 'Qabf', 'SSIM', 'FMI', 'SCD'])
        
        for mode, display_name in variant_order:
            if mode in results and results[mode] is not None:
                m = results[mode]
                writer.writerow([
                    display_name,
                    f"{m['MI']:.4f}",
                    f"{m['VIF']:.4f}",
                    f"{m['Qabf']:.4f}",
                    f"{m['SSIM']:.4f}",
                    f"{m['FMI']:.4f}",
                    f"{m['SCD']:.4f}",
                ])
            else:
                writer.writerow([display_name, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
    
    print(f"Generated CSV: {output_path}")

def write_latex_table(results_pet, results_spect, output_path):
    """Write LaTeX tables to file."""
    # Define the order of variants
    variant_order = [
        ('baseline', 'Baseline'),
        ('wo_backbone', 'w/o Multi-domain Backbone'),
        ('wo_bank', 'w/o Diffusion Process Feature Bank'),
        ('wo_fusion_head', 'w/o Step-wise Fusion Head'),
        ('wo_refinement', 'w/o Reliability Refinement'),
        ('full', 'Full DSDFuse'),
    ]
    
    def get_best_values(results):
        """Find best values for each metric."""
        best = {
            'MI': (None, float('-inf')),
            'VIF': (None, float('-inf')),
            'Qabf': (None, float('-inf')),
            'SSIM': (None, float('-inf')),
            'FMI': (None, float('-inf')),
            'SCD': (None, float('-inf')),  # Assuming higher is better for SCD
        }
        
        for mode, metrics in results.items():
            if metrics is None:
                continue
            for metric, value in metrics.items():
                if value > best[metric][1]:
                    best[metric] = (mode, value)
        
        return best
    
    best_pet = get_best_values(results_pet)
    best_spect = get_best_values(results_spect)
    
    def format_value(results, mode, metric, best):
        """Format value with bold if it's the best."""
        if mode not in results or results[mode] is None:
            return '--'
        
        value = results[mode][metric]
        if best[metric][0] == mode:
            return f'\\textbf{{{value:.4f}}}'
        else:
            return f'{value:.4f}'
    
    with open(output_path, 'w') as f:
        f.write('% DSDFuse Ablation Study Results\n')
        f.write('% Generated by scripts/generate_ablation_summary.py\n')
        f.write('\n')
        
        # PET-MRI Table
        f.write('\\begin{table*}[ht]\n')
        f.write('    \\centering\n')
        f.write('    \\caption{Ablation study on PET-MRI dataset.}\n')
        f.write('    \\label{tab:ablation_pet_mri}\n')
        f.write('    \\begin{tabular}{l|cccccc}\n')
        f.write('        \\toprule\n')
        f.write('        Variant & MI & VIF & Qabf & SSIM & FMI & SCD \\\\\n')
        f.write('        \\midrule\n')
        
        for mode, display_name in variant_order:
            row = [display_name]
            for metric in ['MI', 'VIF', 'Qabf', 'SSIM', 'FMI', 'SCD']:
                row.append(format_value(results_pet, mode, metric, best_pet))
            f.write('        ' + ' & '.join(row) + ' \\\\\n')
        
        f.write('        \\bottomrule\n')
        f.write('    \\end{tabular}\n')
        f.write('\\end{table*}\n')
        f.write('\n')
        
        # SPECT-MRI Table
        f.write('\\begin{table*}[ht]\n')
        f.write('    \\centering\n')
        f.write('    \\caption{Ablation study on SPECT-MRI dataset.}\n')
        f.write('    \\label{tab:ablation_spect_mri}\n')
        f.write('    \\begin{tabular}{l|cccccc}\n')
        f.write('        \\toprule\n')
        f.write('        Variant & MI & VIF & Qabf & SSIM & FMI & SCD \\\\\n')
        f.write('        \\midrule\n')
        
        for mode, display_name in variant_order:
            row = [display_name]
            for metric in ['MI', 'VIF', 'Qabf', 'SSIM', 'FMI', 'SCD']:
                row.append(format_value(results_spect, mode, metric, best_spect))
            f.write('        ' + ' & '.join(row) + ' \\\\\n')
        
        f.write('        \\bottomrule\n')
        f.write('    \\end{tabular}\n')
        f.write('\\end{table*}\n')
    
    print(f"Generated LaTeX table: {output_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define ablation modes
    ablation_modes = ['baseline', 'wo_backbone', 'wo_bank', 'wo_fusion_head', 'wo_refinement', 'full']
    
    # Collect results
    print("Collecting PET-MRI results...")
    results_pet = collect_ablation_results(args.pet_mri_dir, ablation_modes)
    
    print("Collecting SPECT-MRI results...")
    results_spect = collect_ablation_results(args.spect_mri_dir, ablation_modes)
    
    # Write CSV files
    pet_csv_path = os.path.join(args.output_dir, 'ablation_pet_mri.csv')
    write_csv(results_pet, pet_csv_path, 'PET-MRI')
    
    spect_csv_path = os.path.join(args.output_dir, 'ablation_spect_mri.csv')
    write_csv(results_spect, spect_csv_path, 'SPECT-MRI')
    
    # Write LaTeX table
    latex_path = os.path.join(args.output_dir, 'ablation_tables.tex')
    write_latex_table(results_pet, results_spect, latex_path)
    
    print("\nSummary generation completed!")
    print(f"Results saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()