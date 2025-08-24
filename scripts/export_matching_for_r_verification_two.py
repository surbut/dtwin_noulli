"""
Export Matching Data for R Verification and ITE Analysis

This script exports the results from simple_treatment_analysis to R-compatible formats
for verification and individual treatment effect (ITE) analysis.

Author: Sarah Urbut
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def export_matching_for_r_verification(statin_results, thetas, processed_ids, covariate_dicts, 
                                     output_dir="r_verification_data"):
    """
    Export matched data to R-compatible formats for verification and ITE analysis
    
    Parameters:
    - statin_results: Results from simple_treatment_analysis
    - thetas: Signature loadings (N x K x T) - not exported, just for reference
    - processed_ids: Array of all processed patient IDs
    - covariate_dicts: Dictionary with covariate data
    - output_dir: Directory to save exported files
    
    Returns:
    - Dictionary with file paths and summary statistics
    """
    
    print("=== EXPORTING MATCHING DATA FOR R VERIFICATION ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key information from results
    matched_treated_eids = statin_results['matched_patients']['treated_eids']
    matched_control_eids = statin_results['matched_patients']['control_eids']
    matched_treated_indices = statin_results['matched_patients']['treated_indices']
    matched_control_indices = statin_results['matched_patients']['control_indices']
    
    treated_times = statin_results['treatment_times']['treated_times']
    control_times = statin_results['treatment_times']['control_times']
    
    print(f"Exporting {len(matched_treated_eids):,} matched pairs")
    print(f"  - Treated patients: {len(matched_treated_eids):,}")
    print(f"  - Control patients: {len(matched_control_eids):,}")
    
    # 1. Export matched patient IDs and treatment times
    print("\n1. Exporting matched patient pairs...")
    
    # Create matched pairs DataFrame
    matched_pairs = pd.DataFrame({
        'pair_id': range(1, len(matched_treated_eids) + 1),
        'treated_eid': matched_treated_eids,
        'control_eid': matched_control_eids,
        'treated_time': treated_times,
        'control_time': control_times,
        'treated_global_index': matched_treated_indices,
        'control_global_index': matched_control_indices
    })
    
    matched_pairs_path = os.path.join(output_dir, "matched_pairs.csv")
    matched_pairs.to_csv(matched_pairs_path, index=False)
    print(f"   ✅ Saved matched pairs to: {matched_pairs_path}")
    
    # 2. Export covariates for matched patients
    print("\n2. Exporting covariates for matched patients...")
    
    # Get all matched EIDs and treatment status
    all_matched_eids = matched_treated_eids + matched_control_eids
    all_treatment_status = [1] * len(matched_treated_eids) + [0] * len(matched_control_eids)
    
    covariates_data = []
    
    for i, (eid, treatment_status) in enumerate(zip(all_matched_eids, all_treatment_status)):
        row = {
            'patient_id': i + 1,
            'eid': eid,
            'treatment_status': treatment_status
        }
        
        # Add all available covariates
        for cov_name, cov_dict in covariate_dicts.items():
            if isinstance(cov_dict, dict):
                value = cov_dict.get(int(eid))
                if value is not None:
                    # Check if value is numeric before calling np.isnan
                    try:
                        if np.isnan(value):
                            row[cov_name] = np.nan
                        else:
                            row[cov_name] = value
                    except (TypeError, ValueError):
                        # Value is not numeric (e.g., string), store as is
                        row[cov_name] = value
                else:
                    row[cov_name] = np.nan
            else:
                row[cov_name] = np.nan
        
        covariates_data.append(row)
    
    covariates_df = pd.DataFrame(covariates_data)
    covariates_path = os.path.join(output_dir, "covariates.csv")
    covariates_df.to_csv(covariates_path, index=False)
    print(f"   ✅ Saved covariates to: {covariates_path}")
    print(f"   📊 Covariates data shape: {covariates_df.shape}")
    
    # 3. Export treatment timing information
    print("\n3. Exporting treatment timing information...")
    
    treatment_timing = pd.DataFrame({
        'patient_id': range(1, len(all_matched_eids) + 1),
        'eid': all_matched_eids,
        'treatment_status': all_treatment_status,
        'treatment_time': treated_times + control_times,
        'age_at_treatment': [30 + treatment_time 
                            for treatment_time in treated_times + control_times]
    })
    
    treatment_timing_path = os.path.join(output_dir, "treatment_timing.csv")
    treatment_timing.to_csv(treatment_timing_path, index=False)
    print(f"   ✅ Saved treatment timing to: {treatment_timing_path}")
    
    # 4. Export outcomes data (if available)
    print("\n4. Exporting outcomes data...")
    
    if 'hazard_ratio_results' in statin_results:
        hr_results = statin_results['hazard_ratio_results']
        
        # Create outcomes summary
        outcomes_summary = pd.DataFrame({
            'group': ['Treated', 'Control'],
            'n_patients': [hr_results['n_treated'], hr_results['n_control']],
            'n_events': [hr_results['total_events'] - hr_results['n_control'] + hr_results['n_treated'], 
                        hr_results['n_control']],
            'hazard_ratio': [hr_results['hazard_ratio'], 1.0],
            'hr_ci_lower': [hr_results['hr_ci_lower'], 1.0],
            'hr_ci_upper': [hr_results['hr_ci_upper'], 1.0],
            'p_value': [hr_results['p_value'], 1.0]
        })
        
        outcomes_path = os.path.join(output_dir, "outcomes.csv")
        outcomes_summary.to_csv(outcomes_path, index=False)
        print(f"   ✅ Saved outcomes summary to: {outcomes_path}")
        
        # Export individual patient outcomes for full HR reproduction
        print("   Exporting individual patient outcomes...")
        
        # Extract the actual follow-up times and event data from the analysis
        individual_outcomes = []
        
        # Get the follow-up times and outcomes directly from statin_results
        if 'follow_up_times' in statin_results and 'treated_outcomes' in statin_results:
            treated_follow_ups = statin_results['follow_up_times']
            treated_events = statin_results['treated_outcomes']
                
                for i, (eid, treatment_time, follow_up, event) in enumerate(zip(
                    matched_treated_eids, treated_times, treated_follow_ups, treated_events)):
                    
                    individual_outcomes.append({
                        'patient_id': i + 1,
                        'eid': eid,
                        'treatment_status': 1,
                        'treatment_time': treatment_time,
                    'age_at_treatment': 30 + treatment_time,
                        'follow_up_time': follow_up,
                        'event_occurred': event,
                        'group': 'treated'
                    })
                
                # Extract follow-up times and outcomes for control patients
            if 'control_outcomes' in statin_results:
                control_events = statin_results['control_outcomes']
                    
                    for i, (eid, control_time, follow_up, event) in enumerate(zip(
                        matched_control_eids, control_times, treated_follow_ups[len(matched_treated_eids):], control_events)):
                        
                        individual_outcomes.append({
                            'patient_id': len(matched_treated_eids) + i + 1,
                            'eid': eid,
                            'treatment_status': 0,
                            'treatment_time': control_time,
                        'age_at_treatment': 30 + control_time,
                            'follow_up_time': follow_up,
                            'event_occurred': event,
                            'group': 'control'
                        })
                
            print(f"   ✅ Extracted follow-up times and events from statin_results")
            else:
                print("   ⚠️  Follow-up times not found in statin_results")
                # Fallback to basic structure without follow-up times
                for i, (eid, treatment_time) in enumerate(zip(matched_treated_eids, treated_times)):
                    individual_outcomes.append({
                        'patient_id': i + 1,
                        'eid': eid,
                        'treatment_status': 1,
                        'treatment_time': treatment_time,
                        'age_at_treatment': 30 + treatment_time,
                        'group': 'treated'
                    })
                
                for i, (eid, control_time) in enumerate(zip(matched_control_eids, control_times)):
                    individual_outcomes.append({
                        'patient_id': len(matched_treated_eids) + i + 1,
                        'eid': eid,
                        'treatment_status': 0,
                        'treatment_time': control_time,
                        'age_at_treatment': 30 + control_time,
                        'group': 'control'
                    })

        
        # Create individual outcomes DataFrame
        individual_outcomes_df = pd.DataFrame(individual_outcomes)
        individual_outcomes_path = os.path.join(output_dir, "individual_outcomes.csv")
        individual_outcomes_df.to_csv(individual_outcomes_path, index=False)
        print(f"   ✅ Saved individual outcomes to: {individual_outcomes_path}")
        print(f"   📊 Individual outcomes data shape: {individual_outcomes_df.shape}")
        
        # Check if we have follow-up times
        has_follow_ups = 'follow_up_time' in individual_outcomes_df.columns
        if has_follow_ups:
            print(f"   ✅ Follow-up times included - R can fully reproduce HR calculation")
        else:
            print(f"   ⚠️  Follow-up times not available - R will need summary statistics for HR verification")
    
    else:
        print("   No hazard ratio results found - skipping outcomes export")
    
    # 5. Create R verification script
    print("\n5. Creating R verification script...")
    
    r_script = create_r_verification_script(output_dir)
    r_script_path = os.path.join(output_dir, "verify_matching_and_hr.R")
    
    with open(r_script_path, 'w') as f:
        f.write(r_script)
    
    print(f"   ✅ Saved R verification script to: {r_script_path}")
    
    # 6. Create summary report
    print("\n6. Creating summary report...")
    
    summary_report = create_summary_report(statin_results, output_dir)
    summary_path = os.path.join(output_dir, "export_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    print(f"   ✅ Saved summary report to: {summary_path}")
    
    # Return summary of exported files
    exported_files = {
        'matched_pairs': matched_pairs_path,
        'covariates': covariates_path,
        'treatment_timing': treatment_timing_path,
        'r_script': r_script_path,
        'summary': summary_path
    }
    
    if 'hazard_ratio_results' in statin_results:
        exported_files['outcomes'] = os.path.join(output_dir, "outcomes.csv")
        exported_files['individual_outcomes'] = os.path.join(output_dir, "individual_outcomes.csv")
    
    print(f"\n=== EXPORT COMPLETE ===")
    print(f"All files saved to: {output_dir}")
    print(f"Ready for R verification!")
    print(f"\nNote: Signatures not exported - R can access them using the EIDs and global indices")
    print(f"Individual outcomes exported for HR reproduction in R")
    
    return exported_files

def create_r_verification_script(output_dir):
    """Create R script for comprehensive verification and HR reproduction"""
    
    # Build R script without f-string to avoid backslash issues
    r_script = f"""# R Script for Comprehensive Verification and HR Reproduction
# Generated from Python export script
# Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Load required libraries
library(dplyr)
library(survival)
library(ggplot2)

# Set working directory (adjust path as needed)
# setwd("{output_dir}")

# Load data
cat("Loading exported data...\\n")
matched_pairs <- read.csv("matched_pairs.csv")
covariates <- read.csv("covariates.csv")
treatment_timing <- read.csv("treatment_timing.csv")

# Load outcomes data if available
if (file.exists("individual_outcomes.csv")) {{
    individual_outcomes <- read.csv("individual_outcomes.csv")
    cat("Individual outcomes data loaded\\n")
    
    # Check if we have follow-up times and events
    has_follow_ups <- "follow_up_time" %in% colnames(individual_outcomes)
    has_events <- "event_occurred" %in% colnames(individual_outcomes)
    
    if (has_follow_ups && has_events) {{
        cat("Full outcome data available - can reproduce HR calculation\\n")
    }} else if (has_events) {{
        cat("Events available but no follow-up times - limited HR reproduction\\n")
    }} else {{
        cat("Basic structure only - will use summary statistics\\n")
    }}
}} else {{
    individual_outcomes <- NULL
    cat("Individual outcomes data not found\\n")
}}

if (file.exists("outcomes.csv")) {{
    outcomes_summary <- read.csv("outcomes.csv")
    cat("Outcomes summary loaded\\n")
}} else {{
    outcomes_summary <- NULL
    cat("Outcomes summary not found\\n")
}}

# Verify data integrity
cat("\\n=== DATA VERIFICATION ===\\n")
cat("Matched pairs:", nrow(matched_pairs), "\\n")
cat("Treated patients:", sum(matched_pairs$treated_eid > 0), "\\n")
cat("Control patients:", sum(matched_pairs$control_eid > 0), "\\n")
cat("Covariates data:", nrow(covariates), "patients\\n")
cat("Treatment timing data:", nrow(treatment_timing), "patients\\n")

if (!is.null(individual_outcomes)) {{
    cat("Individual outcomes data:", nrow(individual_outcomes), "patients\\n")
    cat("  - Treated:", sum(individual_outcomes$treatment_status == 1), "\\n")
    cat("  - Controls:", sum(individual_outcomes$treatment_status == 0), "\\n")
}}

if (!is.null(outcomes_summary)) {{
    cat("Outcomes summary available with HR results\\n")
    cat("  - Overall HR:", outcomes_summary$hazard_ratio[1], "\\n")
    cat("  - 95% CI:", outcomes_summary$hr_ci_lower[1], "-", outcomes_summary$hr_ci_upper[1], "\\n")
    cat("  - P-value:", outcomes_summary$p_value[1], "\\n")
}}

# COMPREHENSIVE VERIFICATION SECTION
cat("\\n=== COMPREHENSIVE VERIFICATION ===\\n")

# 1. Verify covariates and times alignment
cat("\\n1. VERIFYING COVARIATES AND TIMES ALIGNMENT:\\n")

# Check if all patients in matched_pairs have covariates
matched_eids <- c(matched_pairs$treated_eid, matched_pairs$control_eid)
cov_eids <- covariates$eid
missing_covs <- setdiff(matched_eids, cov_eids)

if (length(missing_covs) == 0) {{
    cat("✅ All matched patients have covariates\\n")
}} else {{
    cat("❌ Missing covariates for", length(missing_covs), "patients\\n")
    cat("Sample missing EIDs:", head(missing_covs, 5), "\\n")
}}

# Check if all patients in matched_pairs have treatment timing
timing_eids <- treatment_timing$eid
missing_timing <- setdiff(matched_eids, timing_eids)

if (length(missing_timing) == 0) {{
    cat("✅ All matched patients have treatment timing\\n")
}} else {{
    cat("❌ Missing treatment timing for", length(missing_timing), "patients\\n")
    cat("Sample missing EIDs:", head(missing_timing, 5), "\\n")
}}

# 2. Verify treatment time matching with first script dates
cat("\\n2. VERIFYING TREATMENT TIME MATCHING:\\n")

# Check treatment time ranges
treated_timing <- treatment_timing[treatment_timing$treatment_status == 1, ]
control_timing <- treatment_timing[treatment_timing$treatment_status == 0, ]

cat("Treatment time ranges:\\n")
cat("  - Treated times (index):", range(treated_timing$treatment_time), "\\n")
cat("  - Control times (index):", range(control_timing$treatment_time), "\\n")
cat("  - Age at treatment ranges:\\n")
cat("    Treated:", range(treated_timing$age_at_treatment), "\\n")
cat("    Control:", range(control_timing$age_at_treatment), "\\n")

# Check for reasonable treatment times
cat("\\nTreatment time validation:\\n")
cat("  - Treated patients with treatment_time < 10:", sum(treated_timing$treatment_time < 10), "\\n")
cat("  - Treated patients with treatment_time > 40:", sum(treated_timing$treatment_time > 40), "\\n")
cat("  - Control patients with treatment_time < 10:", sum(control_timing$treatment_time < 10), "\\n")
cat("  - Control patients with treatment_time > 40:", sum(control_timing$treatment_time > 40), "\\n")

# 3. Verify follow-up time and event indicator correctness
cat("\\n3. VERIFYING FOLLOW-UP TIMES AND EVENTS:\\n")

if (!is.null(individual_outcomes) && has_follow_ups && has_events) {{
    # Check follow-up time ranges
    treated_outcomes <- individual_outcomes[individual_outcomes$treatment_status == 1, ]
    control_outcomes <- individual_outcomes[individual_outcomes$treatment_status == 0, ]
    
    cat("Follow-up time validation:\\n")
    cat("  - Treated follow-up range:", range(treated_outcomes$follow_up_time), "\\n")
    cat("  - Control follow-up range:", range(control_outcomes$follow_up_time), "\\n")
    
    # Check for unreasonable follow-up times
    cat("  - Treated follow-up > 50 years:", sum(treated_outcomes$follow_up_time > 50), "\\n")
    cat("  - Control follow-up > 50 years:", sum(control_outcomes$follow_up_time > 50), "\\n")
    cat("  - Treated follow-up < 0 years:", sum(treated_outcomes$follow_up_time < 0), "\\n")
    cat("  - Control follow-up < 0 years:", sum(control_outcomes$follow_up_time < 0), "\\n")
    
    # Check event rates
    cat("\\nEvent rate validation:\\n")
    treated_event_rate <- mean(treated_outcomes$event_occurred) * 100
    control_event_rate <- mean(control_outcomes$event_occurred) * 100
    
    cat("  - Treated event rate:", round(treated_event_rate, 1), "%\\n")
    cat("  - Control event rate:", round(control_event_rate, 1), "%\\n")
    cat("  - Rate difference:", round(treated_event_rate - control_event_rate, 1), "%\\n")
    
    # Check follow-up time by event status
    cat("\\nFollow-up time by event status:\\n")
    treated_events <- treated_outcomes[treated_outcomes$event_occurred == 1, ]
    treated_censored <- treated_outcomes[treated_outcomes$event_occurred == 0, ]
    
    if (nrow(treated_events) > 0) {{
        cat("  - Treated with events - follow-up range:", range(treated_events$follow_up_time), "\\n")
        cat("  - Treated with events - mean follow-up:", round(mean(treated_events$follow_up_time), 1), "\\n")
    }}
    
    if (nrow(treated_censored) > 0) {{
        cat("  - Treated censored - follow-up range:", range(treated_censored$follow_up_time), "\\n")
        cat("  - Treated censored - mean follow-up:", round(mean(treated_censored$follow_up_time), 1), "\\n")
    }}
    
    control_events <- control_outcomes[control_outcomes$event_occurred == 1, ]
    control_censored <- control_outcomes[control_outcomes$event_occurred == 0, ]
    
    if (nrow(control_events) > 0) {{
        cat("  - Control with events - follow-up range:", range(control_events$follow_up_time), "\\n")
        cat("  - Control with events - mean follow-up:", round(mean(control_events$follow_up_time), 1), "\\n")
    }}
    
    if (nrow(control_censored) > 0) {{
        cat("  - Control censored - follow-up range:", range(control_censored$follow_up_time), "\\n")
        cat("  - Control censored - mean follow-up:", round(mean(control_censored$follow_up_time), 1), "\\n")
    }}
    
}} else {{
    cat("⚠️ Cannot verify follow-up times - data not available\\n")
}}

# 4. Verify pre-event exclusion (check for events at time 0)
cat("\\n4. VERIFYING PRE-EVENT EXCLUSION:\\n")

if (!is.null(individual_outcomes) && has_follow_ups && has_events) {{
    # Check for patients with events at time 0 (should be excluded)
    zero_time_events <- individual_outcomes[individual_outcomes$follow_up_time == 0 & individual_outcomes$event_occurred == 1, ]
    
    if (nrow(zero_time_events) == 0) {{
        cat("✅ No patients with events at time 0 (pre-event exclusion working)\\n")
    }} else {{
        cat("❌ Found", nrow(zero_time_events), "patients with events at time 0\\n")
        cat("Sample patients with time 0 events:\\n")
        print(head(zero_time_events[, c("patient_id", "eid", "treatment_status", "follow_up_time", "event_occurred")], 5))
    }}
    
    # Check for very early events (within first year)
    early_events <- individual_outcomes[individual_outcomes$follow_up_time <= 1 & individual_outcomes$event_occurred == 1, ]
    cat("  - Patients with events ≤ 1 year:", nrow(early_events), "\\n")
    
    # Check for very late events (suspicious)
    late_events <- individual_outcomes[individual_outcomes$follow_up_time > 50 & individual_outcomes$event_occurred == 1, ]
    cat("  - Patients with events > 50 years:", nrow(late_events), "\\n")
    
}} else {{
    cat("⚠️ Cannot verify pre-event exclusion - data not available\\n")
}}

# 5. Perform Cox regression to reproduce HR
cat("\\n5. COX REGRESSION TO REPRODUCE HR:\\n")

if (!is.null(individual_outcomes) && has_follow_ups && has_events) {{
    # Prepare data for Cox model
    cox_data <- data.frame(
        time = individual_outcomes$follow_up_time,
        event = individual_outcomes$event_occurred,
        treatment = individual_outcomes$treatment_status
    )
    
    # Remove any invalid data
    cox_data <- cox_data[cox_data$time > 0 & !is.na(cox_data$time) & !is.na(cox_data$event), ]
    
    cat("Cox regression data summary:\\n")
    cat("  - Total patients:", nrow(cox_data), "\\n")
    cat("  - Treated patients:", sum(cox_data$treatment == 1), "\\n")
    cat("  - Control patients:", sum(cox_data$treatment == 0), "\\n")
    cat("  - Total events:", sum(cox_data$event), "\\n")
    cat("  - Follow-up time range:", range(cox_data$time), "\\n")
    
    # Fit Cox model
    cox_model <- coxph(Surv(time, event) ~ treatment, data = cox_data)
    cox_summary <- summary(cox_model)
    
    # Extract results
    hr <- exp(cox_summary$coefficients[1, 1])
    hr_ci_lower <- exp(cox_summary$coefficients[1, 1] - 1.96 * cox_summary$coefficients[1, 3])
    hr_ci_upper <- exp(cox_summary$coefficients[1, 1] + 1.96 * cox_summary$coefficients[1, 3])
    p_value <- cox_summary$coefficients[1, 5]
    concordance <- cox_summary$concordance[1]
    
    cat("\\nCox regression results:\\n")
    cat("  - Hazard Ratio:", round(hr, 3), "\\n")
    cat("  - 95% CI:", round(hr_ci_lower, 3), "-", round(hr_ci_upper, 3), "\\n")
    cat("  - P-value:", format(p_value, scientific = TRUE, digits = 3), "\\n")
    cat("  - Concordance index:", round(concordance, 3), "\\n")
    
    # Compare with Python results
    if (!is.null(outcomes_summary)) {{
        python_hr <- outcomes_summary$hazard_ratio[1]
        hr_diff <- hr - python_hr
        
        cat("\\nComparison with Python results:\\n")
        cat("  - Python HR:", round(python_hr, 3), "\\n")
        cat("  - R HR:", round(hr, 3), "\\n")
        cat("  - Difference:", round(hr_diff, 3), "\\n")
        
        if (abs(hr_diff) < 0.01) {{
            cat("  ✅ HR values match closely\\n")
        }} else {{
            cat("  ⚠️ HR values differ - investigate further\\n")
        }}
    }}
    
    # Test proportional hazards assumption
    cat("\\nTesting proportional hazards assumption:\\n")
    ph_test <- cox.zph(cox_model)
    
    if (ph_test$table[1, "p"] < 0.05) {{
        cat("  ❌ Proportional hazards assumption violated (p < 0.05)\\n")
        cat("  This could explain HR discrepancy!\\n")
    }} else {{
        cat("  ✅ Proportional hazards assumption holds\\n")
    }}
    
}} else {{
    cat("⚠️ Cannot perform Cox regression - insufficient data\\n")
}}

# 6. Verify matching balance
cat("\\n6. VERIFYING MATCHING BALANCE:\\n")

# Check key covariates
balance_vars <- c("age_at_enroll", "sex", "dm2_prev", "antihtnbase", "ldl_prs")
for (var in balance_vars) {{
    if (var %in% names(covariates)) {{
        treated_covs <- covariates[covariates$treatment_status == 1, ]
        control_covs <- covariates[covariates$treatment_status == 0, ]
        
        treated_mean <- mean(treated_covs[[var]], na.rm = TRUE)
        control_mean <- mean(control_covs[[var]], na.rm = TRUE)
        treated_sd <- sd(treated_covs[[var]], na.rm = TRUE)
        control_sd <- sd(control_covs[[var]], na.rm = TRUE)
        
        # Calculate standardized difference
        pooled_sd <- sqrt((treated_sd^2 + control_sd^2) / 2)
        std_diff <- (treated_mean - control_mean) / pooled_sd
        
        cat(sprintf("%s: Treated=%.2f±%.2f, Control=%.2f±%.2f, StdDiff=%.3f\\n", 
                   var, treated_mean, treated_sd, control_mean, control_sd, std_diff))
        
        # Flag large imbalances
        if (abs(std_diff) > 0.1) {{
            cat("  ⚠️ Large imbalance detected\\n")
        }}
    }}
}}

# 7. Summary and recommendations
cat("\\n=== SUMMARY AND RECOMMENDATIONS ===\\n")

if (!is.null(individual_outcomes) && has_follow_ups && has_events) {{
    # Check for potential issues
    issues_found <- FALSE
    
    # Issue 1: Very early events
    if (nrow(early_events) > 0) {{
        cat("⚠️ Issue 1: Found", nrow(early_events), "patients with events ≤ 1 year\\n")
        cat("   This could indicate pre-treatment events not properly excluded\\n")
        issues_found <- TRUE
    }}
    
    # Issue 2: Very late events
    if (nrow(late_events) > 0) {{
        cat("⚠️ Issue 2: Found", nrow(late_events), "patients with events > 50 years\\n")
        cat("   This could indicate follow-up time calculation errors\\n")
        issues_found <- TRUE
    }}
    
    # Issue 3: Follow-up time mismatches
    treated_max_fu <- max(treated_outcomes$follow_up_time, na.rm = TRUE)
    control_max_fu <- max(control_outcomes$follow_up_time, na.rm = TRUE)
    
    if (abs(treated_max_fu - control_max_fu) > 10) {{
        cat("⚠️ Issue 3: Large difference in max follow-up times\\n")
        cat("   Treated max:", treated_max_fu, "vs Control max:", control_max_fu, "\\n")
        issues_found <- TRUE
    }}
    
    if (!issues_found) {{
        cat("✅ No obvious data quality issues detected\\n")
        cat("   The HR discrepancy may be due to:\\n")
        cat("   1. Time-varying treatment effects\\n")
        cat("   2. Selection bias in the observational data\\n")
        cat("   3. Unmeasured confounding\\n")
    }}
    
}} else {{
    cat("⚠️ Cannot provide comprehensive recommendations - data incomplete\\n")
}}

cat("\\n=== VERIFICATION COMPLETE ===\\n")
cat("Check the output above for any issues or warnings\\n")
cat("If HR values don't match, investigate the flagged issues\\n")
"""
    
    return r_script

def create_summary_report(statin_results, output_dir):
    """Create a summary report of the exported data"""
    
    report = f"""EXPORT SUMMARY REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Output Directory: {output_dir}

=== ANALYSIS SUMMARY ===
Matched pairs: {len(statin_results['matched_patients']['treated_eids']):,}
Treated patients: {len(statin_results['matched_patients']['treated_eids']):,}
Control patients: {len(statin_results['matched_patients']['control_eids']):,}

=== EXPORTED FILES ===
1. matched_pairs.csv - Matched patient pairs with treatment times
2. covariates.csv - Clinical covariates for all matched patients
3. treatment_timing.csv - Treatment timing information
4. verify_matching_and_hr.R - R script for verification and ITE analysis
5. export_summary.txt - This summary report

=== DATA STRUCTURE ===
- Each patient has a unique patient_id (1 to N)
- treatment_status: 1 = treated, 0 = control
- time_from_treatment: Years relative to treatment/enrollment
- Covariates: All available clinical variables
- Treatment timing: Treatment time and age at treatment

=== R ANALYSIS CAPABILITIES ===
1. Verify matching balance
2. Calculate Individual Treatment Effects (ITE)
3. Stratify by clinical characteristics
4. Stratify by signature patterns
5. Perform subgroup HR analysis
6. Visualize signature trajectories

=== NEXT STEPS ===
1. Run the R verification script
2. Check matching balance across strata
3. Calculate ITE for individual patients
4. Identify heterogeneous treatment effects
5. Validate results against Python analysis

For questions or issues, check the R script comments and data structure.
"""
    
    return report
def export_matched_patient_y_data(results, Y, processed_ids, output_dir="./r_verification_data"):
    """
    Export Y tensor data only for the matched patients
    """
    import torch
    
    # Get matched patient indices
    matched_treated_indices = results['matched_patients']['treated_indices']
    matched_control_indices = results['matched_patients']['control_indices']
    
    # Combine all matched indices
    all_matched_indices = np.concatenate([matched_treated_indices, matched_control_indices])
    
    # Extract Y data for matched patients only
    matched_y_data = Y[all_matched_indices, :, :]  # [n_matched, events, time]
    
    # Create mapping from matched index to patient info
    matched_patient_info = []
    for i, idx in enumerate(all_matched_indices):
        eid = processed_ids[idx]
        is_treated = i < len(matched_treated_indices)
        matched_patient_info.append({
            'local_index': i,
            'global_index': int(idx),
            'eid': int(eid),
            'treatment_status': 1 if is_treated else 0
        })
    
    # Save as PyTorch tensor (.pt)
    torch.save(matched_y_data, f"{output_dir}/matched_patient_y_data.pt")
    
    # Save patient mapping
    import pandas as pd
    patient_df = pd.DataFrame(matched_patient_info)
    patient_df.to_csv(f"{output_dir}/matched_patient_indices.csv", index=False)
    
    print(f"✅ Exported Y data for {len(all_matched_indices)} matched patients")
    print(f"   Y shape: {matched_y_data.shape}")
    print(f"   Saved to: {output_dir}/matched_patient_y_data.pt")
    print(f"   Patient mapping: {output_dir}/matched_patient_indices.csv")
    
    return matched_y_data, matched_patient_info    

# Example usage
if __name__ == "__main__":
    print("This script exports matching data for R verification and ITE analysis.")
    print("Use it by calling export_matching_for_r_verification() with your results.")
    
    # Example:
    # exported_files = export_matching_for_r_verification(
    #     statin_results, thetas, processed_ids, covariate_dicts
    # )
    
  
