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
        'age_at_treatment': [covariate_dicts['age_at_enroll'].get(int(eid), np.nan) + treatment_time 
                            for eid, treatment_time in zip(all_matched_eids, treated_times + control_times)]
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
        
        # Get the follow-up times and outcomes from the comprehensive results
        if 'comprehensive_results' in statin_results:
            comp_results = statin_results['comprehensive_results']
            
            # Extract follow-up times and outcomes for treated patients
            if 'follow_up_times' in comp_results and 'treated_outcomes' in comp_results:
                treated_follow_ups = comp_results['follow_up_times']
                treated_events = comp_results['treated_outcomes']
                
                for i, (eid, treatment_time, follow_up, event) in enumerate(zip(
                    matched_treated_eids, treated_times, treated_follow_ups, treated_events)):
                    
                    individual_outcomes.append({
                        'patient_id': i + 1,
                        'eid': eid,
                        'treatment_status': 1,
                        'treatment_time': treatment_time,
                        'age_at_treatment': covariate_dicts['age_at_enroll'].get(int(eid), np.nan) + treatment_time,
                        'follow_up_time': follow_up,
                        'event_occurred': event,
                        'group': 'treated'
                    })
                
                # Extract follow-up times and outcomes for control patients
                if 'control_outcomes' in comp_results:
                    control_events = comp_results['control_outcomes']
                    
                    for i, (eid, control_time, follow_up, event) in enumerate(zip(
                        matched_control_eids, control_times, treated_follow_ups[len(matched_treated_eids):], control_events)):
                        
                        individual_outcomes.append({
                            'patient_id': len(matched_treated_eids) + i + 1,
                            'eid': eid,
                            'treatment_status': 0,
                            'treatment_time': control_time,
                            'age_at_treatment': covariate_dicts['age_at_enroll'].get(int(eid), np.nan) + control_time,
                            'follow_up_time': follow_up,
                            'event_occurred': event,
                            'group': 'control'
                        })
                
                print(f"   ✅ Extracted follow-up times and events from comprehensive results")
            else:
                print("   ⚠️  Follow-up times not found in comprehensive results")
                # Fallback to basic structure without follow-up times
                for i, (eid, treatment_time) in enumerate(zip(matched_treated_eids, treated_times)):
                    individual_outcomes.append({
                        'patient_id': i + 1,
                        'eid': eid,
                        'treatment_status': 1,
                        'treatment_time': treatment_time,
                        'age_at_treatment': covariate_dicts['age_at_enroll'].get(int(eid), np.nan) + treatment_time,
                        'group': 'treated'
                    })
                
                for i, (eid, control_time) in enumerate(zip(matched_control_eids, control_times)):
                    individual_outcomes.append({
                        'patient_id': len(matched_treated_eids) + i + 1,
                        'eid': eid,
                        'treatment_status': 0,
                        'treatment_time': control_time,
                        'age_at_treatment': covariate_dicts['age_at_enroll'].get(int(eid), np.nan) + control_time,
                        'group': 'control'
                    })
        else:
            print("   ⚠️  Comprehensive results not found - using basic structure")
            # Basic structure without follow-up times
            for i, (eid, treatment_time) in enumerate(zip(matched_treated_eids, treated_times)):
                individual_outcomes.append({
                    'patient_id': i + 1,
                    'eid': eid,
                    'treatment_status': 1,
                    'treatment_time': treatment_time,
                    'age_at_treatment': covariate_dicts['age_at_enroll'].get(int(eid), np.nan) + treatment_time,
                    'group': 'treated'
                })
            
            for i, (eid, control_time) in enumerate(zip(matched_control_eids, control_times)):
                individual_outcomes.append({
                    'patient_id': len(matched_treated_eids) + i + 1,
                    'eid': eid,
                    'treatment_status': 0,
                    'treatment_time': control_time,
                    'age_at_treatment': covariate_dicts['age_at_enroll'].get(int(eid), np.nan) + control_time,
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
    """Create R script for verification and ITE analysis"""
    
    r_script = f'''# R Script for Verifying Matching and Calculating ITE
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
if (file.exists("individual_outcomes.csv")) {
    individual_outcomes <- read.csv("individual_outcomes.csv")
    cat("Individual outcomes data loaded\\n")
    
    # Check if we have follow-up times and events
    has_follow_ups <- "follow_up_time" %in% colnames(individual_outcomes)
    has_events <- "event_occurred" %in% colnames(individual_outcomes)
    
    if (has_follow_ups && has_events) {
        cat("✅ Full outcome data available - can reproduce HR calculation\\n")
    } else if (has_events) {
        cat("⚠️  Events available but no follow-up times - limited HR reproduction\\n")
    } else {
        cat("⚠️  Basic structure only - will use summary statistics\\n")
    }
} else {
    individual_outcomes <- NULL
    cat("Individual outcomes data not found\\n")
}

if (file.exists("outcomes.csv")) {
    outcomes_summary <- read.csv("outcomes.csv")
    cat("Outcomes summary loaded\\n")
} else {
    outcomes_summary <- NULL
    cat("Outcomes summary not found\\n")
}

# Verify data integrity
cat("\\n=== DATA VERIFICATION ===\\n")
cat("Matched pairs:", nrow(matched_pairs), "\\n")
cat("Treated patients:", sum(matched_pairs$treated_eid > 0), "\\n")
cat("Control patients:", sum(matched_pairs$control_eid > 0), "\\n")
cat("Covariates data:", nrow(covariates), "patients\\n")
cat("Treatment timing data:", nrow(treatment_timing), "patients\\n")

if (!is.null(individual_outcomes)) {
    cat("Individual outcomes data:", nrow(individual_outcomes), "patients\\n")
    cat("  - Treated:", sum(individual_outcomes$treatment_status == 1), "\\n")
    cat("  - Controls:", sum(individual_outcomes$treatment_status == 0), "\\n")
}

if (!is.null(outcomes_summary)) {
    cat("Outcomes summary available with HR results\\n")
    cat("  - Overall HR:", outcomes_summary$hazard_ratio[1], "\\n")
    cat("  - 95% CI:", outcomes_summary$hr_ci_lower[1], "-", outcomes_summary$hr_ci_upper[1], "\\n")
    cat("  - P-value:", outcomes_summary$p_value[1], "\\n")
}

# Verify matching balance
cat("\\n=== MATCHING BALANCE VERIFICATION ===\\n")
treated_covs <- covariates[covariates$treatment_status == 1, ]
control_covs <- covariates[covariates$treatment_status == 0, ]

# Check key covariates
balance_vars <- c("age_at_enroll", "sex", "dm2_prev", "antihtnbase", "ldl_prs")
for (var in balance_vars) {{
    if (var %in% names(treated_covs)) {{
        treated_mean <- mean(treated_covs[[var]], na.rm = TRUE)
        control_mean <- mean(control_covs[[var]], na.rm = TRUE)
        cat(sprintf("%s: Treated=%.2f, Control=%.2f, Diff=%.2f\\n", 
                   var, treated_mean, control_mean, treated_mean - control_mean))
    }}
}}

# Calculate Individual Treatment Effects (ITE) using signatures
cat("\\n=== INDIVIDUAL TREATMENT EFFECTS (ITE) ===\\n")

# Function to calculate ITE for a patient
calculate_ite <- function(patient_id, signatures_data) {{
    patient_sigs <- signatures_data[signatures_data$patient_id == patient_id, ]
    
    # Get pre-treatment signatures (time_from_treatment < 0)
    pre_treatment <- patient_sigs[patient_sigs$time_from_treatment < 0, ]
    
    # Get post-treatment signatures (time_from_treatment > 0)
    post_treatment <- patient_sigs[patient_sigs$time_from_treatment > 0, ]
    
    if (nrow(pre_treatment) > 0 && nrow(post_treatment) > 0) {{
        # Calculate change in each signature
        ite_effects <- c()
        for (i in 0:20) {{  # Assuming 21 signatures
            sig_col <- paste0("signature_", i)
            if (sig_col %in% names(pre_treatment)) {{
                pre_mean <- mean(pre_treatment[[sig_col]], na.rm = TRUE)
                post_mean <- mean(post_treatment[[sig_col]], na.rm = TRUE)
                ite_effects <- c(ite_effects, post_mean - pre_mean)
            }}
        }}
        return(ite_effects)
    }}
    return(rep(NA, 21))
}}

# Calculate ITE for all treated patients
treated_patients <- unique(covariates[covariates$treatment_status == 1, "patient_id"])
ite_results <- list()

for (patient_id in treated_patients) {{
    ite_effects <- calculate_ite(patient_id, covariates) # Use covariates for signatures
    ite_results[[as.character(patient_id)]] <- ite_effects
}}

# Convert to matrix
ite_matrix <- do.call(rbind, ite_results)
colnames(ite_matrix) <- paste0("signature_", 0:20)

# Summary statistics for ITE
cat("\\nITE Summary Statistics:\\n")
cat("Mean ITE across signatures:", mean(ite_matrix, na.rm = TRUE), "\\n")
cat("SD of ITE across signatures:", sd(ite_matrix, na.rm = TRUE), "\\n")

# Stratification analysis
cat("\\n=== STRATIFICATION ANALYSIS ===\\n")

# Stratify by age
age_quartiles <- quantile(covariates$age_at_enroll, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
covariates$age_group <- cut(covariates$age_at_enroll, 
                           breaks = c(-Inf, age_quartiles, Inf), 
                           labels = c("Q1", "Q2", "Q3", "Q4"))

# Stratify by signature patterns (using first signature as example)
if ("signature_0" %in% names(covariates)) {{
    sig_0_means <- aggregate(signature_0 ~ patient_id, data = covariates, FUN = mean)
    sig_0_quartiles <- quantile(sig_0_means$signature_0, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
    sig_0_means$sig_group <- cut(sig_0_means$signature_0, 
                                 breaks = c(-Inf, sig_0_quartiles, Inf), 
                                 labels = c("Low", "Medium", "High", "Very High"))
    
    # Merge back with covariates
    covariates <- merge(covariates, sig_0_means[, c("patient_id", "sig_group")], 
                       by = "patient_id", all.x = TRUE)
}}

# Calculate HR by strata
cat("\\nHazard Ratio by Age Group:\\n")
for (age_group in levels(covariates$age_group)) {{
    group_patients <- covariates[covariates$age_group == age_group, "patient_id"]
    if (length(group_patients) > 10) {{
        cat("Age group", age_group, ":", length(group_patients), "patients\\n")
    }}
}}

cat("\\nHazard Ratio by Signature Group:\\n")
for (sig_group in levels(covariates$sig_group)) {{
    group_patients <- covariates[covariates$sig_group == sig_group, "patient_id"]
    if (length(group_patients) > 10) {{
        cat("Signature group", sig_group, ":", length(group_patients), "patients\\n")
    }}
}}

cat("\\n=== EXPORT COMPLETE ===")
cat("\\nData ready for further analysis in R!")
cat("\\n\\nNext steps:")
cat("\\n1. Verify matching balance")
cat("\\n2. Calculate ITE for individual patients")
cat("\\n3. Stratify by clinical and signature characteristics")
cat("\\n4. Perform subgroup HR analysis")
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

# Example usage
if __name__ == "__main__":
    print("This script exports matching data for R verification and ITE analysis.")
    print("Use it by calling export_matching_for_r_verification() with your results.")
    
    # Example:
    # exported_files = export_matching_for_r_verification(
    #     statin_results, thetas, processed_ids, covariate_dicts
    # )
