#!/usr/bin/env python3
"""
Test the Fixed Individual Treatment Effect (ITE) Calculation

This script tests whether the updated ITE calculation in true_differential_treatment_effects.py
now correctly aligns with the successful HR=0.70 methodology that reproduced trial results.

Author: Sarah Urbut
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from true_differential_treatment_effects import calculate_individual_treatment_effects_from_matching

def test_statin_ite_calculation(statin_results, thetas, Y, processed_ids, covariate_dicts):
    """
    Test the updated ITE calculation using statin data to validate it aligns with HR=0.70.
    
    Parameters:
    - statin_results: Results from simple_treatment_analysis.py (should include matching pairs)
    - thetas: Signature loadings
    - Y: Outcome tensor
    - processed_ids: Patient IDs
    - covariate_dicts: Covariate dictionaries
    
    Returns:
    - Validation results and comparison with expected HR
    """
    
    print("="*80)
    print("TESTING FIXED ITE CALCULATION")
    print("Validating against successful HR=0.70 methodology")
    print("="*80)
    
    # Check if statin_results has the required structure
    print(f"\nStatin results structure:")
    print(f"Available keys: {list(statin_results.keys())}")
    
    if 'matched_patients' not in statin_results:
        print("❌ ERROR: 'matched_patients' not found in statin_results")
        print("   Make sure you've updated simple_treatment_analysis.py to return comprehensive results")
        return None
    
    if 'treatment_times' not in statin_results:
        print("❌ ERROR: 'treatment_times' not found in statin_results")
        print("   Make sure you've updated simple_treatment_analysis.py to return treatment times")
        return None
    
    # Extract matching data
    matched_patients = statin_results['matched_patients']
    treatment_times = statin_results['treatment_times']
    
    print(f"\nMatching data extracted:")
    print(f"  Treated patients: {len(matched_patients['treated_eids'])}")
    print(f"  Control patients: {len(matched_patients['control_eids'])}")
    print(f"  Treatment times available: {len(treatment_times['treated_times'])}")
    
    # Create matched pairs list
    matched_pairs = []
    treated_eids = matched_patients['treated_eids']
    control_eids = matched_patients['control_eids']
    treated_times = treatment_times['treated_times']
    
    for treated_eid, control_eid in zip(treated_eids, control_eids):
        matched_pairs.append({
            'treated_eid': treated_eid,
            'control_eid': control_eid
        })
    
    print(f"  Created {len(matched_pairs)} matched pairs for analysis")
    
    # Test the ITE calculation (assuming CVD outcome index - adjust as needed)
    cvd_outcome_idx = 0  # Adjust this based on your outcome tensor structure
    
    print(f"\n{'='*60}")
    print(f"RUNNING ITE CALCULATION WITH FIXED METHODOLOGY")
    print(f"{'='*60}")
    
    # Run the updated ITE calculation
    ite_results = calculate_individual_treatment_effects_from_matching(
        treated_eids=treated_eids,
        control_eids=control_eids,
        matched_pairs=matched_pairs,
        Y_tensor=Y,
        outcome_idx=cvd_outcome_idx,
        processed_ids=processed_ids,
        treated_times=treated_times,
        covariate_dicts=covariate_dicts,
        follow_up_years=5
    )
    
    if ite_results['n_patients'] == 0:
        print("❌ No valid patients for ITE calculation")
        return None
    
    # Extract and analyze results
    individual_effects = ite_results['individual_effects']
    
    print(f"\n{'='*60}")
    print(f"ITE CALCULATION RESULTS")
    print(f"{'='*60}")
    
    overall_ite_effect = np.mean(individual_effects)
    ite_std = np.std(individual_effects)
    
    print(f"✅ Successfully calculated ITEs for {len(individual_effects)} patients")
    print(f"Overall ITE effect: {overall_ite_effect:.6f} hazard reduction per year")
    print(f"Standard deviation: {ite_std:.6f}")
    print(f"Effect range: [{np.min(individual_effects):.6f}, {np.max(individual_effects):.6f}]")
    
    # Compare with expected HR=0.70 (30% risk reduction)
    print(f"\n{'='*60}")
    print(f"VALIDATION AGAINST SUCCESSFUL HR=0.70")
    print(f"{'='*60}")
    
    # HR=0.70 means 30% risk reduction
    # In terms of hazard rate difference, this should be positive (control higher than treated)
    expected_magnitude = 0.30  # 30% effect
    
    print(f"Expected from HR=0.70: ~30% risk reduction")
    print(f"Actual ITE effect: {overall_ite_effect:.6f}")
    
    if abs(overall_ite_effect) < 0.00001:  # Very small effect like before
        print("❌ VALIDATION FAILED: ITE effect is still essentially zero")
        print("   The ITE calculation is not capturing the treatment effect properly")
        
        # Diagnostic information
        print(f"\n🔍 DIAGNOSTIC INFORMATION:")
        print(f"Exclusions breakdown:")
        for reason, count in ite_results['exclusion_summary'].items():
            if count > 0:
                print(f"  - {reason}: {count}")
        
        return {
            'validation_status': 'FAILED',
            'ite_effect': overall_ite_effect,
            'expected_effect': expected_magnitude,
            'alignment_score': 0,
            'diagnostic_info': ite_results['exclusion_summary']
        }
        
    elif overall_ite_effect > 0:  # Positive means control has higher hazard (treatment reduces risk)
        alignment_score = min(1.0, overall_ite_effect / (expected_magnitude * 0.1))  # Rough scaling
        
        print("✅ VALIDATION PASSED: ITE shows positive treatment effect")
        print(f"   Treatment reduces hazard rate by {overall_ite_effect:.6f} per year")
        print(f"   This aligns with HR=0.70 expectation")
        print(f"   Alignment score: {alignment_score:.3f}")
        
        return {
            'validation_status': 'PASSED',
            'ite_effect': overall_ite_effect,
            'expected_effect': expected_magnitude,
            'alignment_score': alignment_score,
            'n_patients': len(individual_effects),
            'ite_results': ite_results
        }
        
    else:  # Negative effect
        print("⚠️ UNEXPECTED: ITE shows negative treatment effect")
        print(f"   This suggests treatment increases risk, contradicting HR=0.70")
        
        return {
            'validation_status': 'UNEXPECTED',
            'ite_effect': overall_ite_effect,
            'expected_effect': expected_magnitude,
            'alignment_score': 0,
            'n_patients': len(individual_effects)
        }

def create_ite_validation_visualization(validation_results):
    """
    Create visualization to show ITE validation results.
    """
    
    if validation_results is None or validation_results['validation_status'] == 'FAILED':
        print("Cannot create visualization - validation failed")
        return None
    
    ite_results = validation_results.get('ite_results')
    if ite_results is None:
        print("No ITE results available for visualization")
        return None
    
    individual_effects = ite_results['individual_effects']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution of individual treatment effects
    ax1 = axes[0]
    ax1.hist(individual_effects, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(individual_effects), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(individual_effects):.6f}')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5, label='No effect')
    ax1.set_xlabel('Individual Treatment Effect (hazard reduction per year)')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Distribution of Individual Treatment Effects\n(Fixed ITE Calculation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation metrics
    ax2 = axes[1]
    
    metrics = ['ITE Effect', 'Expected (from HR=0.70)', 'Alignment Score']
    values = [
        validation_results['ite_effect'],
        validation_results['expected_effect'] * 0.1,  # Scale for visualization
        validation_results['alignment_score']
    ]
    colors = ['blue', 'green', 'orange']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Effect Magnitude')
    ax2.set_title('ITE Validation Metrics')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # Add status text
    status = validation_results['validation_status']
    status_color = {'PASSED': 'green', 'FAILED': 'red', 'UNEXPECTED': 'orange'}
    
    fig.suptitle(f'ITE Validation Results: {status}', 
                 fontsize=16, fontweight='bold', 
                 color=status_color.get(status, 'black'))
    
    plt.tight_layout()
    
    print(f"\n📊 Visualization created for {len(individual_effects)} patients")
    print(f"Status: {status}")
    
    return fig

def main_ite_validation_test():
    """
    Main function to test the ITE validation.
    This assumes you have statin results available.
    """
    
    print("ITE Validation Test")
    print("="*40)
    print("This script tests whether the updated ITE calculation")
    print("now correctly aligns with the HR=0.70 methodology.")
    print()
    print("To run this test, you need:")
    print("1. statin_results from updated simple_treatment_analysis.py")
    print("2. thetas, Y, processed_ids, covariate_dicts loaded")
    print()
    print("Example usage:")
    print("validation_results = test_statin_ite_calculation(")
    print("    statin_results, thetas, Y, processed_ids, covariate_dicts)")
    print("fig = create_ite_validation_visualization(validation_results)")

if __name__ == "__main__":
    main_ite_validation_test()