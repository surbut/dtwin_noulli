#!/usr/bin/env python3
"""
Debug ITE Calculation

This script adds detailed debugging to understand why the ITE calculation
is showing negative effects when HR shows positive effects.

Author: Sarah Urbut
Date: 2025-01-27
"""

import numpy as np
import pandas as pd

def debug_ite_calculation_with_sample(statin_results, Y, processed_ids, covariate_dicts):
    """
    Debug the ITE calculation by examining a sample of matched pairs in detail.
    """
    
    print("="*80)
    print("DEBUGGING ITE CALCULATION")
    print("="*80)
    
    # Extract data
    matched_patients = statin_results['matched_patients']
    treatment_times = statin_results['treatment_times']
    
    treated_eids = matched_patients['treated_eids']
    control_eids = matched_patients['control_eids'] 
    treated_times_list = treatment_times['treated_times']
    
    # Convert to numpy if needed
    if hasattr(Y, 'detach'):
        Y_np = Y.detach().cpu().numpy()
    else:
        Y_np = Y
        
    # Create EID to index mapping
    eid_to_idx = {eid: idx for idx, eid in enumerate(processed_ids)}
    
    print(f"Y tensor shape: {Y_np.shape}")
    print(f"CVD outcome index: 0 (assuming first outcome is CVD)")
    
    # Look at a sample of 10 matched pairs in detail
    sample_size = min(10, len(treated_eids))
    print(f"\nExamining {sample_size} matched pairs in detail:")
    print("="*80)
    
    outcome_idx = 0  # CVD outcome
    
    for i in range(sample_size):
        treated_eid = treated_eids[i]
        control_eid = control_eids[i]
        treatment_time = treated_times_list[i]
        
        print(f"\nPair {i+1}: Treated EID {treated_eid} vs Control EID {control_eid}")
        print(f"Treatment time: {treatment_time}")
        
        # Get indices
        if treated_eid not in eid_to_idx or control_eid not in eid_to_idx:
            print("  ❌ EID not found in processed_ids")
            continue
            
        treated_idx = eid_to_idx[treated_eid]
        control_idx = eid_to_idx[control_eid]
        
        # Get control time
        control_age = covariate_dicts['age_at_enroll'].get(int(control_eid))
        if control_age is None:
            print("  ❌ Control age not found")
            continue
        control_time = int(control_age - 30)
        
        print(f"  Treated index: {treated_idx}, Control index: {control_idx}")
        print(f"  Treatment time index: {int(treatment_time)}, Control time index: {control_time}")
        
        # Get outcome data for treated patient
        treatment_time_idx = int(treatment_time)
        if treatment_time_idx < Y_np.shape[2]:
            post_treatment_outcomes = Y_np[treated_idx, outcome_idx, treatment_time_idx:]
        else:
            print("  ❌ Treatment time out of bounds")
            continue
            
        # Get outcome data for control patient
        if control_time < Y_np.shape[2]:
            post_control_outcomes = Y_np[control_idx, outcome_idx, control_time:]
        else:
            print("  ❌ Control time out of bounds") 
            continue
        
        # Check for events
        treated_event_occurred = np.any(post_treatment_outcomes > 0)
        control_event_occurred = np.any(post_control_outcomes > 0)
        
        print(f"  Treated patient event occurred: {treated_event_occurred}")
        print(f"  Control patient event occurred: {control_event_occurred}")
        
        # Calculate time to event
        if treated_event_occurred:
            treated_event_times = np.where(post_treatment_outcomes > 0)[0]
            treated_time_to_event = treated_event_times[0] if len(treated_event_times) > 0 else 5
        else:
            treated_time_to_event = min(5, Y_np.shape[2] - treatment_time_idx)
            
        if control_event_occurred:
            control_event_times = np.where(post_control_outcomes > 0)[0]
            control_time_to_event = control_event_times[0] if len(control_event_times) > 0 else 5
        else:
            control_time_to_event = min(5, Y_np.shape[2] - control_time)
            
        print(f"  Treated time to event: {treated_time_to_event}")
        print(f"  Control time to event: {control_time_to_event}")
        
        # Calculate hazards
        treated_hazard = int(treated_event_occurred) / treated_time_to_event if treated_time_to_event > 0 else 0
        control_hazard = int(control_event_occurred) / control_time_to_event if control_time_to_event > 0 else 0
        
        print(f"  Treated hazard: {treated_hazard:.6f} (event={int(treated_event_occurred)}/time={treated_time_to_event})")
        print(f"  Control hazard: {control_hazard:.6f} (event={int(control_event_occurred)}/time={control_time_to_event})")
        
        # Calculate individual treatment effect
        ite = control_hazard - treated_hazard
        print(f"  Individual treatment effect: {ite:.6f} (control - treated)")
        
        if ite > 0:
            print(f"    → Treatment BENEFICIAL (reduces hazard)")
        elif ite < 0:
            print(f"    → Treatment HARMFUL (increases hazard)")
        else:
            print(f"    → No treatment effect")
            
    print("\n" + "="*80)
    print("SUMMARY OF DEBUG FINDINGS")
    print("="*80)
    
    # Now let's look at the overall pattern
    print("Calculating overall statistics...")
    
    # Count events
    total_treated_events = 0
    total_control_events = 0
    total_pairs = 0
    
    for i in range(min(1000, len(treated_eids))):  # Sample first 1000 pairs
        treated_eid = treated_eids[i]
        control_eid = control_eids[i]
        treatment_time = treated_times_list[i]
        
        if treated_eid not in eid_to_idx or control_eid not in eid_to_idx:
            continue
            
        treated_idx = eid_to_idx[treated_eid]
        control_idx = eid_to_idx[control_eid]
        
        control_age = covariate_dicts['age_at_enroll'].get(int(control_eid))
        if control_age is None:
            continue
        control_time = int(control_age - 30)
        
        treatment_time_idx = int(treatment_time)
        if treatment_time_idx >= Y_np.shape[2] or control_time >= Y_np.shape[2]:
            continue
            
        post_treatment_outcomes = Y_np[treated_idx, outcome_idx, treatment_time_idx:]
        post_control_outcomes = Y_np[control_idx, outcome_idx, control_time:]
        
        treated_event = np.any(post_treatment_outcomes > 0)
        control_event = np.any(post_control_outcomes > 0)
        
        total_treated_events += int(treated_event)
        total_control_events += int(control_event)
        total_pairs += 1
    
    print(f"Sample of {total_pairs} pairs:")
    print(f"  Treated patients with events: {total_treated_events} ({100*total_treated_events/total_pairs:.1f}%)")
    print(f"  Control patients with events: {total_control_events} ({100*total_control_events/total_pairs:.1f}%)")
    
    if total_control_events > total_treated_events:
        print("  ✅ Controls have MORE events than treated (expected for beneficial treatment)")
    else:
        print("  ❌ Treated have MORE events than controls (unexpected for beneficial treatment)")
        
    # Calculate expected ITE direction
    treated_event_rate = total_treated_events / total_pairs
    control_event_rate = total_control_events / total_pairs
    expected_ite_sign = control_event_rate - treated_event_rate
    
    print(f"Expected ITE sign based on event rates: {expected_ite_sign:.6f}")
    if expected_ite_sign > 0:
        print("  → Should show POSITIVE treatment effect")
    else:
        print("  → Would show NEGATIVE treatment effect")
        
    return {
        'sample_pairs': total_pairs,
        'treated_events': total_treated_events,
        'control_events': total_control_events,
        'expected_ite_sign': expected_ite_sign
    }

if __name__ == "__main__":
    print("Debug ITE Calculation")
    print("Use debug_ite_calculation_with_sample() function")