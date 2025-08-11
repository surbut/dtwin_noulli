#!/usr/bin/env python3
"""
TRUE Differential Treatment Effects Analysis

This script calculates ACTUAL differential treatment effects using matched controls.
Unlike the previous approach which just looked at baseline risk differences,
this calculates individual treatment effects and tests for heterogeneity.

Author: Sarah Urbut
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_individual_treatment_effects_from_matching(
    treated_eids, control_eids, matched_pairs, 
    Y_tensor, outcome_idx, processed_ids, 
    treated_times, covariate_dicts, follow_up_years=5):
    """
    Calculate individual treatment effects using matched controls.
    
    Parameters:
    - treated_eids: List of treated patient IDs
    - control_eids: List of control patient IDs  
    - matched_pairs: List of matched pairs from your matching algorithm
    - Y_tensor: Outcome tensor [patients, events, time]
    - outcome_idx: Index of the outcome to analyze
    - processed_ids: Array of patient IDs corresponding to tensor indices
    - treated_times: Treatment times for treated patients
    - covariate_dicts: Covariate dictionaries
    - follow_up_years: Years of follow-up to consider
    
    Returns:
    - Dictionary with individual treatment effects and patient info
    """
    
    print(f"\n=== CALCULATING TRUE INDIVIDUAL TREATMENT EFFECTS ===")
    
    # Convert to numpy if PyTorch tensor
    if hasattr(Y_tensor, 'detach'):
        Y_np = Y_tensor.detach().cpu().numpy()
    else:
        Y_np = Y_tensor
        
    # Create EID to index mapping
    eid_to_idx = {eid: idx for idx, eid in enumerate(processed_ids)}
    
    individual_effects = []
    patient_signatures = []
    patient_info = []
    
    print(f"Processing {len(matched_pairs)} matched pairs...")
    
    for pair in matched_pairs:
        treated_eid = pair['treated_eid']
        control_eid = pair['control_eid']
        
        # Get indices in the tensor
        if treated_eid not in eid_to_idx or control_eid not in eid_to_idx:
            continue
            
        treated_idx = eid_to_idx[treated_eid]
        control_idx = eid_to_idx[control_eid]
        
        # Get treatment time for treated patient
        treatment_time = None
        for i, eid in enumerate(treated_eids):
            if eid == treated_eid:
                treatment_time = treated_times[i]
                break
        
        if treatment_time is None:
            continue
            
        # Get control "treatment" time (enrollment age)
        control_age = covariate_dicts['age_at_enroll'].get(int(control_eid))
        if control_age is None or np.isnan(control_age):
            continue
        control_time = int(control_age - 30)  # Convert to time index
        
        # Ensure we have enough follow-up
        treatment_time_idx = int(treatment_time)
        max_follow_up_treated = min(follow_up_years, Y_np.shape[2] - treatment_time_idx)
        max_follow_up_control = min(follow_up_years, Y_np.shape[2] - control_time)
        
        if max_follow_up_treated < 1 or max_follow_up_control < 1:
            continue
            
        # Calculate outcomes during follow-up period
        # For treated patient: events after treatment
        treated_outcomes = Y_np[treated_idx, outcome_idx, 
                                treatment_time_idx:treatment_time_idx+int(max_follow_up_treated)]
        
        # For control patient: events after "control time"
        control_outcomes = Y_np[control_idx, outcome_idx, 
                               control_time:control_time+int(max_follow_up_control)]
        
        # Calculate event rates (events per year)
        treated_event_rate = np.sum(treated_outcomes) / max_follow_up_treated
        control_event_rate = np.sum(control_outcomes) / max_follow_up_control
        
        # Individual treatment effect = Control rate - Treated rate
        # (Positive = beneficial treatment effect)
        individual_treatment_effect = control_event_rate - treated_event_rate
        
        individual_effects.append(individual_treatment_effect)
        patient_info.append({
            'treated_eid': treated_eid,
            'control_eid': control_eid,
            'treated_idx': treated_idx,
            'control_idx': control_idx,
            'treatment_time': treatment_time,
            'control_time': control_time,
            'treated_events': np.sum(treated_outcomes),
            'control_events': np.sum(control_outcomes),
            'treated_follow_up': max_follow_up_treated,
            'control_follow_up': max_follow_up_control,
            'treated_event_rate': treated_event_rate,
            'control_event_rate': control_event_rate,
            'individual_treatment_effect': individual_treatment_effect
        })
    
    print(f"Successfully calculated treatment effects for {len(individual_effects)} patients")
    
    return {
        'individual_effects': np.array(individual_effects),
        'patient_info': patient_info,
        'n_patients': len(individual_effects)
    }

def test_differential_treatment_effects_by_signatures(
    individual_effects_data, thetas, processed_ids, signature_names=None):
    """
    Test if treatment effects differ across signature levels.
    
    Parameters:
    - individual_effects_data: Output from calculate_individual_treatment_effects_from_matching
    - thetas: Signature loadings [patients, signatures, time]
    - processed_ids: Patient IDs corresponding to theta indices
    - signature_names: Optional names for signatures
    
    Returns:
    - Dictionary with heterogeneity test results
    """
    
    print(f"\n=== TESTING DIFFERENTIAL TREATMENT EFFECTS BY SIGNATURES ===")
    
    individual_effects = individual_effects_data['individual_effects']
    patient_info = individual_effects_data['patient_info']
    
    if len(individual_effects) < 10:
        print("Warning: Too few patients for meaningful analysis")
        return None
        
    # Extract signature patterns for treated patients at treatment time
    treated_signatures = []
    
    for info in patient_info:
        treated_idx = info['treated_idx']
        treatment_time = int(info['treatment_time'])
        
        # Get signature pattern in the year before treatment
        if treatment_time > 0 and treated_idx < thetas.shape[0]:
            # Use signature at treatment time (or slightly before)
            sig_pattern = thetas[treated_idx, :, treatment_time-1 if treatment_time > 0 else 0]
            treated_signatures.append(sig_pattern)
        else:
            # Skip this patient
            treated_signatures.append(None)
    
    # Filter out None values
    valid_indices = [i for i, sig in enumerate(treated_signatures) if sig is not None]
    valid_signatures = np.array([treated_signatures[i] for i in valid_indices])
    valid_effects = individual_effects[valid_indices]
    
    print(f"Valid patients for signature analysis: {len(valid_effects)}")
    
    if len(valid_effects) < 10:
        print("Warning: Too few valid patients")
        return None
    
    # Test each signature
    n_signatures = valid_signatures.shape[1]
    if signature_names is None:
        signature_names = [f"Signature_{i}" for i in range(n_signatures)]
    
    results = {}
    
    for sig_idx in range(n_signatures):
        sig_name = signature_names[sig_idx]
        sig_values = valid_signatures[:, sig_idx]
        
        # Test correlation between signature and treatment effect
        correlation, corr_pval = stats.pearsonr(sig_values, valid_effects)
        
        # Split into high/low groups
        median_sig = np.median(sig_values)
        high_mask = sig_values > median_sig
        low_mask = sig_values <= median_sig
        
        high_effects = valid_effects[high_mask]
        low_effects = valid_effects[low_mask]
        
        # Test difference between high and low groups
        if len(high_effects) > 5 and len(low_effects) > 5:
            t_stat, t_pval = stats.ttest_ind(high_effects, low_effects)
            effect_size = (np.mean(high_effects) - np.mean(low_effects)) / np.std(valid_effects)
        else:
            t_stat, t_pval, effect_size = np.nan, np.nan, np.nan
        
        results[sig_name] = {
            'signature_index': sig_idx,
            'correlation': correlation,
            'correlation_pval': corr_pval,
            'high_group_mean_effect': np.mean(high_effects) if len(high_effects) > 0 else np.nan,
            'low_group_mean_effect': np.mean(low_effects) if len(low_effects) > 0 else np.nan,
            'high_vs_low_tstat': t_stat,
            'high_vs_low_pval': t_pval,
            'effect_size': effect_size,
            'n_high': len(high_effects),
            'n_low': len(low_effects),
            'significant_correlation': corr_pval < 0.05 if not np.isnan(corr_pval) else False,
            'significant_difference': t_pval < 0.05 if not np.isnan(t_pval) else False,
            'clinically_meaningful': abs(effect_size) > 0.2 if not np.isnan(effect_size) else False
        }
    
    return results

def interpret_differential_effects(results, individual_effects_data):
    """
    Interpret the differential treatment effect results in clinical terms.
    """
    
    print(f"\n{'='*80}")
    print(f"INTERPRETATION: DIFFERENTIAL TREATMENT EFFECTS")
    print(f"{'='*80}")
    
    individual_effects = individual_effects_data['individual_effects']
    
    # Overall treatment effect
    overall_effect = np.mean(individual_effects)
    overall_std = np.std(individual_effects)
    
    print(f"\nOVERALL TREATMENT EFFECT:")
    print(f"  Mean individual treatment effect: {overall_effect:.4f} events/year")
    print(f"  Standard deviation: {overall_std:.4f}")
    
    if overall_effect > 0:
        print(f"  → On average, treatment REDUCES event rate by {overall_effect:.4f} events/year")
    else:
        print(f"  → On average, treatment INCREASES event rate by {abs(overall_effect):.4f} events/year")
    
    # Find most significant signatures
    significant_signatures = []
    for sig_name, result in results.items():
        if result['significant_difference'] and result['clinically_meaningful']:
            significant_signatures.append((sig_name, result))
    
    if len(significant_signatures) == 0:
        print(f"\n❌ NO DIFFERENTIAL TREATMENT EFFECTS FOUND")
        print(f"  All patients appear to benefit similarly from treatment")
        return
    
    # Sort by effect size
    significant_signatures.sort(key=lambda x: abs(x[1]['effect_size']), reverse=True)
    
    print(f"\n✅ DIFFERENTIAL TREATMENT EFFECTS FOUND:")
    print(f"  {len(significant_signatures)} signatures show significant heterogeneity")
    
    for sig_name, result in significant_signatures[:5]:  # Top 5
        print(f"\n{sig_name}:")
        print(f"  High {sig_name} patients: {result['high_group_mean_effect']:.4f} treatment effect")
        print(f"  Low {sig_name} patients: {result['low_group_mean_effect']:.4f} treatment effect")
        
        diff = result['high_group_mean_effect'] - result['low_group_mean_effect']
        print(f"  Difference: {diff:.4f} events/year (p={result['high_vs_low_pval']:.2e})")
        
        if diff > 0:
            print(f"  → High {sig_name} patients benefit MORE from treatment")
        else:
            print(f"  → Low {sig_name} patients benefit MORE from treatment")
        
        print(f"  Effect size: {result['effect_size']:.3f} (Cohen's d)")

def create_differential_effects_visualization(results, individual_effects_data, 
                                            outcome_name="Outcome", drug_name="Treatment"):
    """
    Create comprehensive visualizations for differential treatment effects.
    """
    
    individual_effects = individual_effects_data['individual_effects']
    
    # Find significant results
    significant_results = {name: result for name, result in results.items() 
                          if result['significant_difference']}
    
    if len(significant_results) == 0:
        print("No significant differential effects to visualize")
        return None
    
    # Create figure
    n_significant = min(6, len(significant_results))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Differential Treatment Effects: {drug_name} on {outcome_name}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Overall treatment effect distribution
    ax1 = axes[0, 0]
    ax1.hist(individual_effects, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(individual_effects), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(individual_effects):.4f}')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5, label='No effect')
    ax1.set_xlabel('Individual Treatment Effect (events/year)')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Distribution of Individual Treatment Effects')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot top significant signatures
    sorted_sigs = sorted(significant_results.items(), 
                        key=lambda x: abs(x[1]['effect_size']), reverse=True)
    
    for i, (sig_name, result) in enumerate(sorted_sigs[:5]):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]
        
        # Create bar plot for high vs low groups
        groups = ['High\n' + sig_name, 'Low\n' + sig_name]
        effects = [result['high_group_mean_effect'], result['low_group_mean_effect']]
        colors = ['red' if e > np.mean(individual_effects) else 'blue' for e in effects]
        
        bars = ax.bar(groups, effects, color=colors, alpha=0.7, edgecolor='black')
        
        # Add overall mean line
        ax.axhline(np.mean(individual_effects), color='gray', linestyle='--', 
                   label=f'Overall mean: {np.mean(individual_effects):.4f}')
        
        # Add p-value and effect size
        ax.text(0.5, max(effects) * 1.1, 
                f'p = {result["high_vs_low_pval"]:.2e}\nCohen\'s d = {result["effect_size"]:.3f}',
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        ax.set_ylabel('Treatment Effect (events/year)')
        ax.set_title(f'{sig_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add sample sizes
        ax.text(0, min(effects) * 0.9, f'n={result["n_high"]}', ha='center', fontsize=10)
        ax.text(1, min(effects) * 0.9, f'n={result["n_low"]}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total patients analyzed: {len(individual_effects)}")
    print(f"Signatures with differential effects: {len(significant_results)}")
    print(f"Overall treatment effect: {np.mean(individual_effects):.4f} ± {np.std(individual_effects):.4f}")
    
    return fig

def run_true_differential_analysis(matched_pairs, treated_eids, treated_times, 
                                 Y_tensor, outcome_idx, outcome_name,
                                 thetas, processed_ids, covariate_dicts, 
                                 drug_name="Treatment"):
    """
    Run complete differential treatment effects analysis.
    
    Parameters:
    - matched_pairs: Output from your matching algorithm (list of dicts with 'treated_eid' and 'control_eid')
    - treated_eids: List of treated patient IDs
    - treated_times: List of treatment times
    - Y_tensor: Outcome tensor
    - outcome_idx: Index of outcome to analyze
    - outcome_name: Name of outcome for reporting
    - thetas: Signature loadings
    - processed_ids: Patient IDs
    - covariate_dicts: Covariate dictionaries
    - drug_name: Name of treatment
    
    Returns:
    - Complete results dictionary
    """
    
    print(f"{'='*80}")
    print(f"TRUE DIFFERENTIAL TREATMENT EFFECTS ANALYSIS")
    print(f"Drug: {drug_name}, Outcome: {outcome_name}")
    print(f"{'='*80}")
    
    # Step 1: Calculate individual treatment effects
    individual_effects_data = calculate_individual_treatment_effects_from_matching(
        treated_eids=treated_eids,
        control_eids=[],  # Will extract from matched_pairs
        matched_pairs=matched_pairs,
        Y_tensor=Y_tensor,
        outcome_idx=outcome_idx,
        processed_ids=processed_ids,
        treated_times=treated_times,
        covariate_dicts=covariate_dicts
    )
    
    if individual_effects_data['n_patients'] < 10:
        print("❌ Insufficient data for differential analysis")
        return None
    
    # Step 2: Test for differential effects by signatures
    differential_results = test_differential_treatment_effects_by_signatures(
        individual_effects_data=individual_effects_data,
        thetas=thetas,
        processed_ids=processed_ids
    )
    
    if differential_results is None:
        print("❌ Could not perform signature analysis")
        return None
    
    # Step 3: Interpret results
    interpret_differential_effects(differential_results, individual_effects_data)
    
    # Step 4: Create visualizations
    fig = create_differential_effects_visualization(
        results=differential_results,
        individual_effects_data=individual_effects_data,
        outcome_name=outcome_name,
        drug_name=drug_name
    )
    
    return {
        'individual_effects_data': individual_effects_data,
        'differential_results': differential_results,
        'visualization': fig,
        'summary': {
            'n_patients': individual_effects_data['n_patients'],
            'overall_treatment_effect': np.mean(individual_effects_data['individual_effects']),
            'n_signatures_with_differential_effects': sum(1 for r in differential_results.values() 
                                                         if r['significant_difference']),
            'most_differential_signature': max(differential_results.items(), 
                                             key=lambda x: abs(x[1]['effect_size']) if not np.isnan(x[1]['effect_size']) else 0)
        }
    }

if __name__ == "__main__":
    print("True Differential Treatment Effects Analysis")
    print("Use run_true_differential_analysis() function")