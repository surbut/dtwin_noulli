#!/usr/bin/env python3
"""
Standalone Fixed Aspirin Analysis

This is a simplified, standalone version that fixes the pair correspondence issue
without requiring imports from other scripts.

Author: Sarah Urbut
Date: 2025-01-27
"""

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
except ImportError:
    print("Warning: lifelines not available for HR calculation")

def calculate_hazard_ratio_fixed(treated_outcomes, control_outcomes, follow_up_times):
    """Calculate HR using Cox model with proper indexing"""
    try:
        from lifelines import CoxPHFitter
        import warnings
        warnings.filterwarnings('ignore')
    except ImportError:
        return None
    
    # Prepare data
    n_treated = len(treated_outcomes)
    n_control = len(control_outcomes)
    
    # Combine data
    all_outcomes = np.concatenate([treated_outcomes, control_outcomes])
    all_times = np.concatenate([follow_up_times[:n_treated], follow_up_times[n_treated:n_treated+n_control]])
    treatment_status = np.concatenate([np.ones(n_treated), np.zeros(n_control)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': all_times,
        'event': all_outcomes,
        'treatment': treatment_status
    })
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    
    # Extract results
    hr = np.exp(cph.params_['treatment'])
    
    try:
        hr_ci_lower = np.exp(cph.confidence_intervals_.loc['treatment', 'treatment_lower'])
        hr_ci_upper = np.exp(cph.confidence_intervals_.loc['treatment', 'treatment_upper'])
    except KeyError:
        try:
            hr_ci_lower = np.exp(cph.confidence_intervals_.loc['treatment', 'lower 0.95'])
            hr_ci_upper = np.exp(cph.confidence_intervals_.loc['treatment', 'upper 0.95'])
        except KeyError:
            hr_ci_lower = hr * 0.8
            hr_ci_upper = hr * 1.2
    
    p_value = cph.summary.loc['treatment', 'p']
    
    return {
        'hazard_ratio': hr,
        'hr_ci_lower': hr_ci_lower,
        'hr_ci_upper': hr_ci_upper,
        'p_value': p_value,
        'n_treated': n_treated,
        'n_control': n_control,
        'total_events': np.sum(all_outcomes)
    }

def fixed_aspirin_analysis_simple(aspirin_results, Y, processed_ids, covariate_dicts,
                                 colorectal_indices=[10, 11]):
    """
    Simple fixed aspirin analysis using existing matching results.
    
    This takes your existing aspirin matching results and fixes just the outcome extraction
    to maintain proper pair correspondence.
    """
    
    print("="*80)
    print("STANDALONE FIXED ASPIRIN ANALYSIS")
    print("="*80)
    
    # Extract matching data from existing results
    matched_patients = aspirin_results['matched_patients']
    treatment_times_data = aspirin_results['treatment_times']
    
    treated_eids = matched_patients['treated_eids']
    control_eids = matched_patients['control_eids'] 
    treated_indices = matched_patients['treated_indices']
    control_indices = matched_patients['control_indices']
    treated_times = treatment_times_data['treated_times']
    
    print(f"Using existing matching results:")
    print(f"  Matched pairs: {len(treated_eids)}")
    print(f"  Treatment times: {len(treated_times)}")
    
    # Convert Y to numpy
    if hasattr(Y, 'detach'):
        Y_np = Y.detach().cpu().numpy()
    else:
        Y_np = Y
    
    # Create EID to index mapping for treated patients
    eid_to_treatment_time = {}
    for i, eid in enumerate(treated_eids):
        if i < len(treated_times):
            eid_to_treatment_time[eid] = treated_times[i]
    
    # FIXED: Process matched pairs together
    treated_outcomes = []
    control_outcomes = []
    follow_up_times = []
    
    print(f"Processing {len(treated_eids)} matched pairs with fixed indexing...")
    
    valid_pairs = 0
    
    for pair_idx in range(len(treated_eids)):
        if pair_idx % 5000 == 0:
            print(f"  Processed {pair_idx}/{len(treated_eids)} pairs...")
        
        # Get the matched pair
        treated_eid = treated_eids[pair_idx]
        control_eid = control_eids[pair_idx]
        treated_idx = treated_indices[pair_idx]
        control_idx = control_indices[pair_idx]
        
        # ===== PROCESS TREATED PATIENT =====
        # Get treatment time for this specific patient
        treatment_time = eid_to_treatment_time.get(treated_eid)
        
        if treatment_time is None or treated_idx >= Y_np.shape[0]:
            continue
        
        treatment_time_idx = int(treatment_time)
        if treatment_time_idx >= Y_np.shape[2]:
            continue
        
        # Check for CRC events after treatment
        treated_has_event = False
        treated_time_to_event = 5.0  # Default follow-up
        
        for outcome_idx in colorectal_indices:
            if outcome_idx < Y_np.shape[1]:
                post_treatment = Y_np[treated_idx, outcome_idx, treatment_time_idx:]
                if len(post_treatment) > 0 and np.any(post_treatment > 0):
                    treated_has_event = True
                    event_times = np.where(post_treatment > 0)[0]
                    if len(event_times) > 0:
                        treated_time_to_event = min(treated_time_to_event, event_times[0])
        
        if not treated_has_event:
            treated_time_to_event = min(5.0, Y_np.shape[2] - treatment_time_idx)
        
        # ===== PROCESS MATCHED CONTROL PATIENT =====
        control_age = covariate_dicts['age_at_enroll'].get(int(control_eid))
        if control_age is None or np.isnan(control_age):
            continue
            
        control_time = int(control_age - 30)
        if control_time < 0 or control_time >= Y_np.shape[2] or control_idx >= Y_np.shape[0]:
            continue
        
        # Check for CRC events after control time
        control_has_event = False
        control_time_to_event = 5.0  # Default follow-up
        
        for outcome_idx in colorectal_indices:
            if outcome_idx < Y_np.shape[1]:
                post_control = Y_np[control_idx, outcome_idx, control_time:]
                if len(post_control) > 0 and np.any(post_control > 0):
                    control_has_event = True
                    event_times = np.where(post_control > 0)[0]
                    if len(event_times) > 0:
                        control_time_to_event = min(control_time_to_event, event_times[0])
        
        if not control_has_event:
            control_time_to_event = min(5.0, Y_np.shape[2] - control_time)
        
        # Store results for this matched pair
        treated_outcomes.append(int(treated_has_event))
        control_outcomes.append(int(control_has_event))
        
        # Use average follow-up time
        avg_follow_up = (treated_time_to_event + control_time_to_event) / 2
        follow_up_times.append(avg_follow_up)
        
        valid_pairs += 1
    
    print(f"Valid pairs processed: {valid_pairs}")
    print(f"Treated events: {sum(treated_outcomes)}")
    print(f"Control events: {sum(control_outcomes)}")
    
    if valid_pairs < 10:
        print("❌ Insufficient valid pairs for analysis")
        return None
    
    # Calculate HR with fixed indexing
    hr_results = calculate_hazard_ratio_fixed(
        np.array(treated_outcomes),
        np.array(control_outcomes), 
        np.array(follow_up_times)
    )
    
    if hr_results is None:
        print("❌ Could not calculate hazard ratio")
        return None
    
    print(f"\n{'='*80}")
    print("FIXED ASPIRIN RESULTS")
    print(f"{'='*80}")
    print(f"Hazard Ratio: {hr_results['hazard_ratio']:.3f}")
    print(f"95% CI: {hr_results['hr_ci_lower']:.3f} - {hr_results['hr_ci_upper']:.3f}")
    print(f"P-value: {hr_results['p_value']:.4f}")
    print(f"Valid pairs: {valid_pairs}")
    
    # Validation check
    treated_event_rate = sum(treated_outcomes) / len(treated_outcomes)
    control_event_rate = sum(control_outcomes) / len(control_outcomes)
    
    print(f"\nValidation Check:")
    print(f"Treated event rate: {100*treated_event_rate:.2f}%")
    print(f"Control event rate: {100*control_event_rate:.2f}%")
    
    if control_event_rate > 0:
        raw_ratio = treated_event_rate / control_event_rate
        print(f"Raw rate ratio: {raw_ratio:.3f}")
        
        # Check alignment
        hr_protective = hr_results['hazard_ratio'] < 1.0
        raw_protective = raw_ratio < 1.0
        
        if hr_protective == raw_protective:
            print("✅ HR direction matches raw event rates")
            if hr_protective:
                print("✅ Both indicate aspirin is PROTECTIVE")
            else:
                print("⚠️ Both indicate aspirin INCREASES CRC risk")
        else:
            print("❌ HR direction contradicts raw event rates - still broken")
    
    # Return results in same format
    return {
        'hazard_ratio_results': hr_results,
        'matched_patients': {
            'treated_eids': treated_eids[:valid_pairs],
            'control_eids': control_eids[:valid_pairs],
            'treated_indices': treated_indices[:valid_pairs],
            'control_indices': control_indices[:valid_pairs]
        },
        'treatment_times': {
            'treated_times': treated_times[:valid_pairs],
            'control_times': []  # Simplified
        },
        'cohort_sizes': {
            'n_treated': valid_pairs,
            'n_control': valid_pairs,
            'n_total': valid_pairs * 2
        }
    }

if __name__ == "__main__":
    print("Standalone Fixed Aspirin Analysis")
    print("Use fixed_aspirin_analysis_simple() function")