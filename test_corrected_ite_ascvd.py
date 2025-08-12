#!/usr/bin/env python3
"""
Test Corrected ITE Calculation for ASCVD Composite

This script tests the ITE calculation using the SAME ASCVD composite outcome indices
that were used in the successful statin HR calculation (indices 112-116).

Author: Sarah Urbut
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_ascvd_composite_ite(statin_results, Y, processed_ids, covariate_dicts, 
                                 ascvd_indices=[112, 113, 114, 115, 116]):
    """
    Calculate ITE using the SAME ASCVD composite outcome that produced HR=0.70.
    
    This addresses the critical issue: we were using outcome index 0 instead of 
    the ASCVD composite indices 112-116 that the successful HR calculation used.
    """
    
    print("="*80)
    print("CORRECTED ITE CALCULATION FOR ASCVD COMPOSITE")
    print(f"Using ASCVD indices: {ascvd_indices}")
    print("="*80)
    
    # Extract matching data
    matched_patients = statin_results['matched_patients']
    treatment_times = statin_results['treatment_times']
    
    treated_eids = matched_patients['treated_eids']
    control_eids = matched_patients['control_eids']
    treated_times_list = treatment_times['treated_times']
    
    # Convert Y to numpy if needed
    if hasattr(Y, 'detach'):
        Y_np = Y.detach().cpu().numpy()
    else:
        Y_np = Y
        
    print(f"Y tensor shape: {Y_np.shape}")
    print(f"ASCVD outcome indices: {ascvd_indices}")
    
    # Create EID to index mapping
    eid_to_idx = {eid: idx for idx, eid in enumerate(processed_ids)}
    
    # Calculate ITEs for ASCVD composite
    individual_effects_by_outcome = {}
    
    for outcome_idx in ascvd_indices:
        print(f"\nCalculating ITE for outcome index {outcome_idx}...")
        
        individual_effects = []
        
        # Sample first 5000 pairs for speed (you can increase this)
        sample_size = min(5000, len(treated_eids))
        
        treated_events = 0
        control_events = 0
        valid_pairs = 0
        
        for i in range(sample_size):
            if i % 1000 == 0:
                print(f"  Processed {i}/{sample_size} pairs...")
                
            treated_eid = treated_eids[i]
            control_eid = control_eids[i]
            treatment_time = treated_times_list[i]
            
            # Get indices
            if treated_eid not in eid_to_idx or control_eid not in eid_to_idx:
                continue
                
            treated_idx = eid_to_idx[treated_eid]
            control_idx = eid_to_idx[control_eid]
            
            # Get control time
            control_age = covariate_dicts['age_at_enroll'].get(int(control_eid))
            if control_age is None:
                continue
            control_time = int(control_age - 30)
            
            # Check bounds
            treatment_time_idx = int(treatment_time)
            if (treatment_time_idx >= Y_np.shape[2] or control_time >= Y_np.shape[2] or
                outcome_idx >= Y_np.shape[1]):
                continue
                
            # Get post-treatment/control outcomes for this specific outcome
            post_treatment_outcomes = Y_np[treated_idx, outcome_idx, treatment_time_idx:]
            post_control_outcomes = Y_np[control_idx, outcome_idx, control_time:]
            
            # Check for events
            treated_event_occurred = np.any(post_treatment_outcomes > 0)
            control_event_occurred = np.any(post_control_outcomes > 0)
            
            # Calculate time to event
            follow_up_years = 5
            if treated_event_occurred:
                treated_event_times = np.where(post_treatment_outcomes > 0)[0]
                treated_time_to_event = treated_event_times[0] if len(treated_event_times) > 0 else follow_up_years
            else:
                treated_time_to_event = min(follow_up_years, len(post_treatment_outcomes))
                
            if control_event_occurred:
                control_event_times = np.where(post_control_outcomes > 0)[0]
                control_time_to_event = control_event_times[0] if len(control_event_times) > 0 else follow_up_years
            else:
                control_time_to_event = min(follow_up_years, len(post_control_outcomes))
            
            # Calculate individual treatment effect
            if treated_time_to_event > 0 and control_time_to_event > 0:
                treated_hazard = int(treated_event_occurred) / treated_time_to_event
                control_hazard = int(control_event_occurred) / control_time_to_event
                ite = control_hazard - treated_hazard
                individual_effects.append(ite)
                
                # Count events for summary
                treated_events += int(treated_event_occurred)
                control_events += int(control_event_occurred)
                valid_pairs += 1
        
        print(f"  Completed outcome {outcome_idx}: {len(individual_effects)} valid pairs")
        print(f"  Treated events: {treated_events} ({100*treated_events/valid_pairs:.1f}%)")
        print(f"  Control events: {control_events} ({100*control_events/valid_pairs:.1f}%)")
        
        if len(individual_effects) > 0:
            mean_ite = np.mean(individual_effects)
            print(f"  Mean ITE: {mean_ite:.6f}")
            
            if control_events > treated_events:
                print(f"  ✅ Controls have MORE events (treatment beneficial)")
            else:
                print(f"  ❌ Treated have MORE events (unexpected)")
                
            individual_effects_by_outcome[outcome_idx] = {
                'individual_effects': individual_effects,
                'mean_ite': mean_ite,
                'treated_events': treated_events,
                'control_events': control_events,
                'valid_pairs': valid_pairs
            }
    
    # Calculate composite ASCVD ITE
    print(f"\n{'='*60}")
    print("CALCULATING ASCVD COMPOSITE ITE")
    print(f"{'='*60}")
    
    # Combine all ASCVD outcomes - a patient has ASCVD event if ANY of the component events occur
    composite_individual_effects = []
    composite_treated_events = 0
    composite_control_events = 0
    composite_valid_pairs = 0
    
    sample_size = min(5000, len(treated_eids))
    
    for i in range(sample_size):
        if i % 1000 == 0:
            print(f"Processing composite pair {i}/{sample_size}...")
            
        treated_eid = treated_eids[i]
        control_eid = control_eids[i]
        treatment_time = treated_times_list[i]
        
        # Get indices
        if treated_eid not in eid_to_idx or control_eid not in eid_to_idx:
            continue
            
        treated_idx = eid_to_idx[treated_eid]
        control_idx = eid_to_idx[control_eid]
        
        # Get control time
        control_age = covariate_dicts['age_at_enroll'].get(int(control_eid))
        if control_age is None:
            continue
        control_time = int(control_age - 30)
        
        # Check bounds
        treatment_time_idx = int(treatment_time)
        if treatment_time_idx >= Y_np.shape[2] or control_time >= Y_np.shape[2]:
            continue
        
        # Check for ANY ASCVD event (composite)
        treated_ascvd_event = False
        control_ascvd_event = False
        treated_earliest_event_time = float('inf')
        control_earliest_event_time = float('inf')
        
        for outcome_idx in ascvd_indices:
            if outcome_idx >= Y_np.shape[1]:
                continue
                
            # Check treated patient
            post_treatment_outcomes = Y_np[treated_idx, outcome_idx, treatment_time_idx:]
            if np.any(post_treatment_outcomes > 0):
                treated_ascvd_event = True
                event_times = np.where(post_treatment_outcomes > 0)[0]
                if len(event_times) > 0:
                    treated_earliest_event_time = min(treated_earliest_event_time, event_times[0])
                    
            # Check control patient
            post_control_outcomes = Y_np[control_idx, outcome_idx, control_time:]
            if np.any(post_control_outcomes > 0):
                control_ascvd_event = True
                event_times = np.where(post_control_outcomes > 0)[0]
                if len(event_times) > 0:
                    control_earliest_event_time = min(control_earliest_event_time, event_times[0])
        
        # Calculate composite time to event
        follow_up_years = 5
        treated_time_to_event = treated_earliest_event_time if treated_earliest_event_time != float('inf') else follow_up_years
        control_time_to_event = control_earliest_event_time if control_earliest_event_time != float('inf') else follow_up_years
        
        # Cap at follow-up period
        treated_time_to_event = min(treated_time_to_event, follow_up_years)
        control_time_to_event = min(control_time_to_event, follow_up_years)
        
        # Calculate composite ITE
        if treated_time_to_event > 0 and control_time_to_event > 0:
            treated_hazard = int(treated_ascvd_event) / treated_time_to_event
            control_hazard = int(control_ascvd_event) / control_time_to_event
            composite_ite = control_hazard - treated_hazard
            composite_individual_effects.append(composite_ite)
            
            composite_treated_events += int(treated_ascvd_event)
            composite_control_events += int(control_ascvd_event)
            composite_valid_pairs += 1
    
    # Results
    composite_mean_ite = np.mean(composite_individual_effects) if composite_individual_effects else 0
    
    print(f"\n{'='*80}")
    print("ASCVD COMPOSITE ITE RESULTS")
    print(f"{'='*80}")
    print(f"Valid pairs: {composite_valid_pairs}")
    print(f"Treated ASCVD events: {composite_treated_events} ({100*composite_treated_events/composite_valid_pairs:.1f}%)")
    print(f"Control ASCVD events: {composite_control_events} ({100*composite_control_events/composite_valid_pairs:.1f}%)")
    print(f"Composite mean ITE: {composite_mean_ite:.6f}")
    
    # Validation against HR=0.70
    if composite_control_events > composite_treated_events:
        print("✅ Controls have MORE ASCVD events than treated patients")
        print("   This matches expectation for beneficial treatment (HR=0.70)")
        if composite_mean_ite > 0:
            print("✅ Positive ITE aligns with beneficial treatment")
            validation_status = 'PASSED'
        else:
            print("❌ ITE is negative despite more control events - calculation issue")
            validation_status = 'CALCULATION_ERROR'
    else:
        print("❌ Treated have MORE ASCVD events than controls")
        print("   This contradicts HR=0.70 showing benefit")
        validation_status = 'DATA_MISMATCH'
    
    return {
        'validation_status': validation_status,
        'composite_mean_ite': composite_mean_ite,
        'individual_outcomes': individual_effects_by_outcome,
        'composite_data': {
            'individual_effects': composite_individual_effects,
            'treated_events': composite_treated_events,
            'control_events': composite_control_events,
            'valid_pairs': composite_valid_pairs
        }
    }

if __name__ == "__main__":
    print("Corrected ITE Calculation for ASCVD Composite")
    print("Use calculate_ascvd_composite_ite() function")