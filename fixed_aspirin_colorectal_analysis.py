#!/usr/bin/env python3
"""
Fixed Aspirin-Colorectal Cancer Analysis

This script fixes the indexing bug in the original aspirin analysis by maintaining
proper pair correspondence throughout the outcome extraction process.

Key Fix: Process matched pairs together instead of processing all treated then all controls.

Author: Sarah Urbut  
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from lifelines import CoxPHFitter
except ImportError:
    print("Warning: lifelines not available for HR calculation")

def fixed_aspirin_colorectal_analysis(gp_scripts, true_aspirins, processed_ids, thetas, 
                                     sig_indices, covariate_dicts, Y, 
                                     colorectal_cancer_indices=[10, 11], cov=None):
    """
    Fixed aspirin analysis that maintains proper pair correspondence.
    
    Uses the same logic as the working simple_treatment_analysis.py but applied to aspirin/CRC.
    """
    
    print("="*80)
    print("FIXED ASPIRIN-COLORECTAL CANCER ANALYSIS")  
    print("Expected: Aspirin should REDUCE colorectal cancer risk (HR < 1.0)")
    print("="*80)
    
    # Import the working components from simple_treatment_analysis
    import sys
    sys.path.append('/Users/sarahurbut/dtwin_noulli/scripts')
    
    try:
        from simple_treatment_analysis import (
            encode_smoking, build_features, perform_nearest_neighbor_matching,
            assess_matching_balance, verify_patient_cohorts_simple as verify_patient_cohorts,
            verify_matching_results
        )
    except ImportError:
        print("Error: Cannot import from simple_treatment_analysis")
        return None
    
    # Step 1: Verify patient cohorts  
    print("1. Verifying patient cohort definitions:")
    verification_results = verify_patient_cohorts(gp_scripts, true_aspirins, processed_ids)
    if not verification_results:
        print("❌ Patient cohort verification failed")
        return None
    
    # Step 2: Extract treated patients using ObservationalTreatmentPatternLearner
    print("\n2. Extracting treated patients using ObservationalTreatmentPatternLearner:")
    
    try:
        from observational_treatment_patterns import ObservationalTreatmentPatternLearner
    except ImportError:
        print("Error: Cannot import ObservationalTreatmentPatternLearner")
        return None
        
    learner = ObservationalTreatmentPatternLearner(
        processed_ids=processed_ids,
        thetas=thetas,
        treatment_data=true_aspirins,
        covariate_data=covariate_dicts,
        outcomes=Y
    )
    
    treated_eids, treated_times, treated_t0s = learner.get_treated_patients()
    never_treated_eids, never_treated_t0s = learner.get_never_treated_patients()
    
    print(f"Found {len(treated_eids)} treated patients")  
    print(f"Found {len(never_treated_eids)} never-treated patients")
    
    # Verification
    print("\n=== TREATED PATIENT VERIFICATION (ASPIRIN) ===")
    all_aspirin_eids = set(true_aspirins['eid'].unique())
    treated_with_aspirin = sum(1 for eid in treated_eids if eid in all_aspirin_eids)
    treated_without_aspirin = len(treated_eids) - treated_with_aspirin
    print(f"Claimed treated patients: {len(treated_eids):,}")
    print(f"  - With aspirin: {treated_with_aspirin:,}")
    print(f"  - Without aspirin: {treated_without_aspirin:,}")
    if treated_without_aspirin == 0:
        print("✅ All treated patients have aspirin")
    else:
        print("❌ Some treated patients don't have aspirin!")
    
    # Step 3: Define clean controls
    print("\n3. Defining clean controls:")
    
    # Clean controls: never-treated patients with 10+ year follow-up
    min_follow_up = 10
    control_eids_with_followup = []
    control_t0s = []
    
    for eid, t0 in zip(never_treated_eids, never_treated_t0s):
        max_time = thetas.shape[2] - 1
        follow_up_years = max_time - t0
        if follow_up_years >= min_follow_up:
            control_eids_with_followup.append(eid)
            control_t0s.append(t0)
    
    print(f"   Found {len(never_treated_eids)} never-treated patients with signature data")
    
    # Verification
    print("\n=== CONTROL PATIENT VERIFICATION (ASPIRIN) ===")
    control_with_aspirin = sum(1 for eid in control_eids_with_followup if eid in all_aspirin_eids)
    control_without_aspirin = len(control_eids_with_followup) - control_with_aspirin
    print(f"Claimed control patients: {len(control_eids_with_followup):,}")
    print(f"  - With aspirin: {control_with_aspirin:,}")
    print(f"  - Without aspirin: {control_without_aspirin:,}")
    if control_with_aspirin == 0:
        print("✅ All controls are clean (no aspirin)")
    else:
        print("❌ Some controls have aspirin!")
    
    print(f"   Control patients with {min_follow_up}-year follow-up: {len(control_eids_with_followup):,}")
    
    # Step 4: Build features for treated patients
    print("\n4. Building features for treated patients:")
    treated_features, treated_indices, treated_eids_matched = build_features(
        treated_eids, treated_t0s, processed_ids, thetas, covariate_dicts,
        sig_indices=sig_indices, is_treated=True, treatment_dates=treated_times
    )
    
    print(f"   Treated patients after exclusions: {len(treated_features):,}")
    
    # Step 5: Build features for control patients  
    print("\n5. Building features for control patients:")
    control_features, control_indices, control_eids_matched = build_features(
        control_eids_with_followup, control_t0s, processed_ids, thetas, covariate_dicts,
        sig_indices=sig_indices, is_treated=False
    )
    
    print(f"   Control patients after exclusions: {len(control_features):,}")
    
    # Step 6: Perform nearest neighbor matching
    print("\n6. Performing nearest neighbor matching:")
    (matched_treated_indices, matched_control_indices, 
     matched_treated_eids, matched_control_eids) = perform_nearest_neighbor_matching(
        treated_features, control_features, treated_indices, control_indices,
        treated_eids_matched, control_eids_matched
    )
    
    print(f"   Successfully matched pairs: {len(matched_treated_indices):,}")
    
    # Step 7: Calculate outcomes with FIXED pair correspondence
    print("\n7. Calculating outcomes and hazard ratio:")
    
    if Y is not None:
        # Convert PyTorch tensor to numpy if needed
        if hasattr(Y, 'detach'):
            Y_np = Y.detach().cpu().numpy()
        else:
            Y_np = Y
        
        # FIXED: Process matched PAIRS together to maintain correspondence
        treated_outcomes = []
        control_outcomes = []  
        follow_up_times = []
        
        print(f"Processing {len(matched_treated_indices)} matched pairs...")
        
        # THE FIX: Process pairs together instead of separately
        for pair_idx, (treated_idx, control_idx) in enumerate(zip(matched_treated_indices, matched_control_indices)):
            
            if pair_idx % 5000 == 0:
                print(f"  Processed {pair_idx}/{len(matched_treated_indices)} pairs...")
            
            # ===== PROCESS TREATED PATIENT =====
            treated_eid = processed_ids[treated_idx]
            treatment_time = None
            
            # Find treatment time
            for i, eid in enumerate(treated_eids):
                if eid == treated_eid:
                    treatment_time = treated_times[i]
                    break
            
            if treatment_time is not None and treated_idx < Y_np.shape[0]:
                # Look for CRC events after treatment time
                if colorectal_cancer_indices is not None:
                    post_treatment_outcomes = Y_np[treated_idx, colorectal_cancer_indices, int(treatment_time):]
                    post_treatment_outcomes = np.any(post_treatment_outcomes > 0, axis=0)
                else:
                    post_treatment_outcomes = Y_np[treated_idx, :, int(treatment_time):]
                    post_treatment_outcomes = np.any(post_treatment_outcomes > 0, axis=0)
                
                treated_event_occurred = np.any(post_treatment_outcomes > 0)
                
                if treated_event_occurred:
                    event_times = np.where(post_treatment_outcomes > 0)[0]
                    treated_time_to_event = event_times[0] if len(event_times) > 0 else 5.0
                else:
                    treated_time_to_event = min(5.0, Y_np.shape[2] - int(treatment_time))
                
                treated_outcomes.append(int(treated_event_occurred))
                
            else:
                # Skip this pair if treated patient data is invalid
                continue
            
            # ===== PROCESS MATCHED CONTROL PATIENT =====
            control_eid = processed_ids[control_idx]
            
            # For controls, use age-based time point
            age_at_enroll = covariate_dicts['age_at_enroll'].get(int(control_eid))
            if age_at_enroll is not None and not np.isnan(age_at_enroll):
                control_time = int(age_at_enroll - 30)
                
                if control_idx < Y_np.shape[0] and control_time < Y_np.shape[2]:
                    # Look for CRC events after control time
                    if colorectal_cancer_indices is not None:
                        post_control_outcomes = Y_np[control_idx, colorectal_cancer_indices, control_time:]
                        post_control_outcomes = np.any(post_control_outcomes > 0, axis=0)
                    else:
                        post_control_outcomes = Y_np[control_idx, :, control_time:]
                        post_control_outcomes = np.any(post_control_outcomes > 0, axis=0)
                    
                    control_event_occurred = np.any(post_control_outcomes > 0)
                    
                    if control_event_occurred:
                        event_times = np.where(post_control_outcomes > 0)[0]
                        control_time_to_event = event_times[0] if len(event_times) > 0 else 5.0
                    else:
                        control_time_to_event = min(5.0, Y_np.shape[2] - control_time)
                    
                    control_outcomes.append(int(control_event_occurred))
                    
                    # Use average of treated and control follow-up times
                    avg_follow_up = (treated_time_to_event + control_time_to_event) / 2
                    follow_up_times.append(avg_follow_up)
                    
                else:
                    # Remove the treated outcome we just added if control is invalid
                    treated_outcomes.pop()
                    continue
            else:
                # Remove the treated outcome we just added if control is invalid
                treated_outcomes.pop()
                continue
        
        print(f"Final valid pairs: {len(treated_outcomes)}")
        
        if len(treated_outcomes) > 10 and len(control_outcomes) > 10:
            # Import HR calculation function
            from simple_treatment_analysis import calculate_hazard_ratio
            
            hr_results = calculate_hazard_ratio(
                np.array(treated_outcomes),
                np.array(control_outcomes),
                np.array(follow_up_times)
            )
            
            print(f"   Treated colorectal cancer events: {np.sum(treated_outcomes):,}")
            print(f"   Control colorectal cancer events: {np.sum(control_outcomes):,}")  
            print(f"   Total colorectal cancer events: {hr_results['total_events']:,}")
            print()
            
            print("=== FIXED ASPIRIN RESULTS (COLORECTAL CANCER PREVENTION) ===")
            print(f"Hazard Ratio: {hr_results['hazard_ratio']:.3f}")
            print(f"95% CI: {hr_results['hr_ci_lower']:.3f} - {hr_results['hr_ci_upper']:.3f}")
            print(f"P-value: {hr_results['p_value']:.4f}")
            print(f"Matched pairs: {len(treated_outcomes):,}")
            
            # Interpret results
            if hr_results['hazard_ratio'] < 1.0:
                risk_reduction = (1 - hr_results['hazard_ratio']) * 100
                print(f"Risk reduction: {risk_reduction:.1f}%")
                print("✅ Aspirin shows protective effect against CRC")
            else:
                risk_increase = (hr_results['hazard_ratio'] - 1) * 100
                print(f"Risk increase: {risk_increase:.1f}%")
                print("❌ Aspirin shows increased CRC risk (unexpected)")
            
            # Validation check
            treated_event_rate = np.sum(treated_outcomes) / len(treated_outcomes)
            control_event_rate = np.sum(control_outcomes) / len(control_outcomes)
            raw_ratio = treated_event_rate / control_event_rate if control_event_rate > 0 else float('inf')
            
            print(f"\nValidation Check:")
            print(f"Treated event rate: {100*treated_event_rate:.2f}%")
            print(f"Control event rate: {100*control_event_rate:.2f}%")
            print(f"Raw event rate ratio: {raw_ratio:.3f}")
            print(f"HR direction matches raw ratio: {(hr_results['hazard_ratio'] > 1) == (raw_ratio > 1)}")
            
            # Expected HR
            expected_hr = 0.75
            hr_difference = hr_results['hazard_ratio'] - expected_hr
            ci_overlaps_expected = (hr_results['hr_ci_lower'] <= expected_hr <= hr_results['hr_ci_upper'])
            print(f"Expected HR from trials: {expected_hr:.3f}")
            print(f"Difference from expected: {hr_difference:.3f}")
            print(f"CI overlaps expected: {ci_overlaps_expected}")
            
            # Return comprehensive results (same format as working statin analysis)
            comprehensive_results = {
                'hazard_ratio_results': hr_results,
                'matched_patients': {
                    'treated_eids': matched_treated_eids[:len(treated_outcomes)],  # Trim to valid pairs
                    'control_eids': matched_control_eids[:len(control_outcomes)],
                    'treated_indices': matched_treated_indices[:len(treated_outcomes)],
                    'control_indices': matched_control_indices[:len(control_outcomes)]
                },
                'treatment_times': {
                    'treated_times': [treated_times[i] for i, eid in enumerate(treated_eids) 
                                    if eid in matched_treated_eids[:len(treated_outcomes)]],
                    'control_times': control_t0s[:len(control_outcomes)]  # Simplified
                },
                'cohort_sizes': {
                    'n_treated': len(treated_outcomes),
                    'n_control': len(control_outcomes),
                    'n_total': len(treated_outcomes) + len(control_outcomes)
                },
                'matching_features': {
                    'treated_features': treated_features,
                    'control_features': control_features,
                    'feature_names': ['signatures'] + ['age', 'sex', 'dm2', 'antihtn', 'dm1', 'ldl_prs', 'cad_prs', 'tchol', 'hdl', 'sbp', 'pce_goff', 'smoke_never', 'smoke_previous', 'smoke_current']
                }
            }
            
            return comprehensive_results
        else:
            print("   Insufficient outcome data for HR calculation")
            return None
    else:
        print("   No outcomes (Y) found - cannot calculate HR")
        return None

if __name__ == "__main__":
    print("Fixed Aspirin-Colorectal Cancer Analysis")
    print("Use fixed_aspirin_colorectal_analysis() function")