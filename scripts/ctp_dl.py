"""
Comprehensive Treatment Pattern Analysis Pipeline

This script combines:
1. Age-matched feature building (from dt.py)
2. Observational pattern learning (from observational_treatment_patterns.py) 
3. Bayesian causal inference for propensity-response modeling

Author: Sarah Urbut
Date: 2025-07-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind, mannwhitneyu
# Use scipy optimization for MAP estimation (like your existing work)
from scipy.optimize import minimize

try:
    from observational_treatment_patterns import ObservationalTreatmentPatternLearner
except ImportError:
    # Try importing from scripts directory
    from scripts.observational_treatment_patterns import ObservationalTreatmentPatternLearner

def encode_smoking(status):
    # One-hot encoding: [Never, Previous, Current]
    if status == "Never":
        return [1, 0, 0]
    elif status == "Previous":
        return [0, 1, 0]
    elif status == "Current":
        return [0, 0, 1]
    else:
        return [0, 0, 0]  # For missing/unknown

def build_features(eids, t0s, processed_ids, thetas, covariate_dicts, sig_indices=None, is_treated=False, treatment_dates=None):
    """
    Build feature vectors for matching using 10-year signature trajectories and comprehensive covariates
    (Proper implementation from dt.py with NaN handling)
    
    Parameters:
    - eids: List of patient IDs
    - t0s: List of time indices for each patient
    - processed_ids: Array of all processed patient IDs
    - thetas: Signature loadings (N x K x T)
    - covariate_dicts: Dictionary with covariate data
    - sig_indices: Optional list of signature indices to use
    
    Returns:
    - features: Feature matrix for matching
    - indices: Patient indices in thetas
    - kept_eids: Successfully matched patient IDs
    """
    features = []
    indices = []
    kept_eids = []
    window = 10
    n_signatures = thetas.shape[1]
    if sig_indices is None:
        sig_indices = list(range(n_signatures))
    expected_length = len(sig_indices) * window
    
    # Calculate means for imputation (do this once at the beginning)
    ldl_values = [v for v in covariate_dicts.get('ldl_prs', {}).values() if v is not None and not np.isnan(v)]
    hdl_values = [v for v in covariate_dicts.get('hdl', {}).values() if v is not None and not np.isnan(v)]
    tchol_values = [v for v in covariate_dicts.get('tchol', {}).values() if v is not None and not np.isnan(v)]
    sbp_values = [v for v in covariate_dicts.get('SBP', {}).values() if v is not None and not np.isnan(v)]
    
    ldl_mean = np.mean(ldl_values) if ldl_values else 0
    hdl_mean = np.mean(hdl_values) if hdl_values else 50
    tchol_mean = np.mean(tchol_values) if tchol_values else 200
    sbp_mean = np.mean(sbp_values) if sbp_values else 140
    
    print(f"Processing {len(eids)} patients with sig_indices={sig_indices}")
    
    for i, (eid, t0) in enumerate(zip(eids, t0s)):
        if i % 1000 == 0:
            print(f"Processed {i} patients, kept {len(features)} so far")
            
        try:
            idx = np.where(processed_ids == int(eid))[0][0]
        except Exception:
            continue
        if t0 < window:
            continue  # Not enough history
        t0_int = int(t0)
        sig_traj = thetas[idx, sig_indices, t0_int-window:t0_int].flatten()
        #print(f"EID {eid}: sig_traj shape {sig_traj.shape}, expected {expected_length}")
        if sig_traj.shape[0] != expected_length:
            print(f"EID {eid}: sig_traj shape {sig_traj.shape}, expected {expected_length} and wrong length")
            continue  # Skip if not the right length
        
        # Check for NaN in signature trajectory
        if np.any(np.isnan(sig_traj)):
            continue  # Skip if signature has NaN values
        
        # Extract covariates with proper handling
        age_at_enroll = covariate_dicts['age_at_enroll'].get(int(eid), 57)
        if np.isnan(age_at_enroll) or age_at_enroll is None:
            age_at_enroll = 57  # Fallback age
            
        # Determine the correct age for matching based on index date
        if is_treated and treatment_dates is not None:
            # For treated patients: use treatment age as index date
            treatment_idx = eids.index(eid) if eid in eids else None
            if treatment_idx is not None and treatment_idx < len(treatment_dates):
                treatment_age = age_at_enroll + (treatment_dates[treatment_idx] / 12)  # Convert months to years
                age = treatment_age  # Use treatment age for matching
            else:
                # Fallback: no treatment date found, use enrollment age
                age = age_at_enroll
        else:
            # For control patients: use enrollment age as index date
            age = age_at_enroll
            
        sex = covariate_dicts['sex'].get(int(eid), 0)
        if np.isnan(sex) or sex is None:
            sex = 0
        sex = int(sex)
        
        # For "prev" variables: EXCLUDE if missing
        dm2 = covariate_dicts['dm2_prev'].get(int(eid))
        if dm2 is None or np.isnan(dm2):
            print(f"Excluding EID {eid}: missing dm2_prev")
            continue  # Skip this patient
            
        antihtn = covariate_dicts['antihtnbase'].get(int(eid))
        if antihtn is None or np.isnan(antihtn):
            print(f"Excluding EID {eid}: missing antihtnbase")
            continue  # Skip this patient
            
        dm1 = covariate_dicts['dm1_prev'].get(int(eid))
        if dm1 is None or np.isnan(dm1):
            print(f"Excluding EID {eid}: missing dm1_prev")
            continue  # Skip this patient
        
        # EXCLUDE patients with CAD before treatment/enrollment (incident user logic)
        age_at_enroll = covariate_dicts['age_at_enroll'].get(int(eid), 57)
        
        # Determine the reference age for CAD exclusion
        if is_treated and treatment_dates is not None:
            # For treated patients: use first statin prescription date
            treatment_idx = eids.index(eid) if eid in eids else None
            if treatment_idx is not None and treatment_idx < len(treatment_dates):
                treatment_age = age_at_enroll + (treatment_dates[treatment_idx] / 12)  # Convert months to years
                reference_age = treatment_age
                reference_type = "treatment"
            else:
                reference_age = age_at_enroll
                reference_type = "enrollment"
        else:
            # For control patients: use enrollment date
            reference_age = age_at_enroll
            reference_type = "enrollment"
        
        # Check CAD exclusion (only exclude CAD before index date)
        cad_any = covariate_dicts.get('Cad_Any', {}).get(int(eid), 0)
        cad_censor_age = covariate_dicts.get('Cad_censor_age', {}).get(int(eid))
        if cad_any == 2 and cad_censor_age is not None and not np.isnan(cad_censor_age):
            if cad_censor_age < reference_age:
                #print(f"Excluding EID {eid}: CAD occurred before {reference_type} (age {cad_censor_age:.1f} < {reference_age:.1f})")
                continue  # Skip this patient
        
        # Check for missing DM/HTN/HyperLip status (exclude if unknown)
        dm_any = covariate_dicts.get('Dm_Any', {}).get(int(eid))
        if dm_any is None or np.isnan(dm_any):
            print(f"Excluding EID {eid}: missing Dm_Any status")
            continue  # Skip this patient
            
        ht_any = covariate_dicts.get('Ht_Any', {}).get(int(eid))
        if ht_any is None or np.isnan(ht_any):
            print(f"Excluding EID {eid}: missing Ht_Any status")
            continue  # Skip this patient
            
        hyperlip_any = covariate_dicts.get('HyperLip_Any', {}).get(int(eid))
        if hyperlip_any is None or np.isnan(hyperlip_any):
            print(f"Excluding EID {eid}: missing HyperLip_Any status")
            continue  # Skip this patient
        
        # For clinical variables: IMPUTE with mean
        ldl_prs = covariate_dicts.get('ldl_prs', {}).get(int(eid))
        if np.isnan(ldl_prs) or ldl_prs is None:
            ldl_prs = ldl_mean
            
        hdl = covariate_dicts.get('hdl', {}).get(int(eid))
        if np.isnan(hdl) or hdl is None:
            hdl = hdl_mean
            
        tchol = covariate_dicts.get('tchol', {}).get(int(eid))
        if np.isnan(tchol) or tchol is None:
            tchol = tchol_mean
            
        sbp = covariate_dicts.get('SBP', {}).get(int(eid))
        if np.isnan(sbp) or sbp is None:
            sbp = sbp_mean
        
        pce_goff = covariate_dicts.get('pce_goff', {}).get(int(eid), 0.09)
        if np.isnan(pce_goff) or pce_goff is None:
            pce_goff = 0.09
        
        # Add this line back (around line 120, after ldl_prs):
        cad_prs = covariate_dicts.get('cad_prs', {}).get(int(eid), 0)
        if np.isnan(cad_prs) or cad_prs is None:
            cad_prs = 0
        
        # Extract smoking status and encode
        smoke = encode_smoking(covariate_dicts.get('smoke', {}).get(int(eid), None))
        if np.any(np.isnan(smoke)):
            smoke = [0, 0, 0]  # Default to "Never" smoking
        
        # Create feature vector and check for NaN
        feature_vector = np.concatenate([
            sig_traj, [age, sex, dm2, antihtn, dm1, ldl_prs, cad_prs, tchol, hdl, sbp, pce_goff] + smoke
        ])
        
        # Debug: Check where NaN might be coming from
        if np.any(np.isnan(feature_vector)):
            print(f"Warning: NaN found in feature vector for EID {eid}")
            print(f"  sig_traj NaN: {np.any(np.isnan(sig_traj))}")
            print(f"  age: {age}, sex: {sex}, dm2: {dm2}")
            print(f"  smoke: {smoke}")
            print(f"  ldl_prs: {ldl_prs}, cad_prs: {cad_prs}")
            continue
        
        features.append(feature_vector)
        indices.append(idx)
        kept_eids.append(eid)
    
    print(f"Final result: {len(features)} patients kept out of {len(eids)}")
    
    if len(features) == 0:
        print("Warning: No valid features found after NaN filtering")
        return np.array([]), [], []
    
    return np.array(features), indices, kept_eids

def perform_nearest_neighbor_matching(treated_features, control_features, 
                                    treated_indices, control_indices,
                                    treated_eids, control_eids):
    """
    Perform 1:1 nearest neighbor matching between treated and control groups
    
    Returns:
    - matched_treated_indices: Indices of matched treated patients
    - matched_control_indices: Indices of matched control patients  
    - matched_treated_eids: EIDs of matched treated patients
    - matched_control_eids: EIDs of matched control patients
    """
    
    # Standardize features
    scaler = StandardScaler()
    treated_features_std = scaler.fit_transform(treated_features)
    control_features_std = scaler.transform(control_features)
    
    # Perform nearest neighbor matching
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(control_features_std)
    distances, indices = nn.kneighbors(treated_features_std)
    
    # Extract matched pairs
    matched_control_indices = [control_indices[i] for i in indices.flatten()]
    matched_treated_indices = treated_indices
    matched_control_eids = [control_eids[i] for i in indices.flatten()]
    matched_treated_eids = treated_eids
    
    return (matched_treated_indices, matched_control_indices, 
            matched_treated_eids, matched_control_eids)

def assess_matching_balance(treated_features, control_features, 
                          matched_treated_indices, matched_control_indices,
                          covariate_names=None):
    """
    Assess matching balance by comparing covariate distributions before and after matching
    
    Parameters:
    - treated_features: Original treated patient features
    - control_features: Original control patient features  
    - matched_treated_indices: Indices of matched treated patients (local indices)
    - matched_control_indices: Indices of matched control patients (local indices)
    - covariate_names: Optional list of covariate names for labeling
    
    Returns:
    - Dictionary with balance statistics
    """
    
    # Use local indices for feature arrays
    matched_treated_features = treated_features[matched_treated_indices]
    matched_control_features = control_features[matched_control_indices]
    
    # Calculate statistics for each covariate
    n_covariates = treated_features.shape[1]
    balance_stats = {}
    
    for i in range(n_covariates):
        # Before matching
        treated_before = treated_features[:, i]
        control_before = control_features[:, i]
        
        # After matching
        treated_after = matched_treated_features[:, i]
        control_after = matched_control_features[:, i]
        
        # Calculate standardized differences
        def standardized_difference(group1, group2):
            mean_diff = np.mean(group1) - np.mean(group2)
            pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
            return mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Calculate SEM (standard error of the mean)
        def sem(group):
            return np.std(group) / np.sqrt(len(group))
        
        # Before matching
        std_diff_before = standardized_difference(treated_before, control_before)
        treated_sem_before = sem(treated_before)
        control_sem_before = sem(control_before)
        
        # After matching
        std_diff_after = standardized_difference(treated_after, control_after)
        treated_sem_after = sem(treated_after)
        control_sem_after = sem(control_after)
        
        # Store results
        cov_name = f"Covariate_{i}" if covariate_names is None else covariate_names[i]
        balance_stats[cov_name] = {
            'before_matching': {
                'treated_mean': np.mean(treated_before),
                'control_mean': np.mean(control_before),
                'treated_sem': treated_sem_before,
                'control_sem': control_sem_before,
                'std_difference': std_diff_before
            },
            'after_matching': {
                'treated_mean': np.mean(treated_after),
                'control_mean': np.mean(control_after),
                'treated_sem': treated_sem_after,
                'control_sem': control_sem_after,
                'std_difference': std_diff_after
            },
            'improvement': abs(std_diff_before) - abs(std_diff_after)
        }
    
    return balance_stats

def print_balance_summary(balance_stats, top_n=10):
    """
    Print a summary of the balance assessment
    """
    print("\n=== MATCHING BALANCE ASSESSMENT ===")
    
    # Sort by improvement (largest improvements first)
    sorted_stats = sorted(balance_stats.items(), 
                         key=lambda x: x[1]['improvement'], reverse=True)
    
    print(f"\nTop {top_n} covariates with largest balance improvements:")
    print(f"{'Covariate':<20} {'Before':<10} {'After':<10} {'Improvement':<12}")
    print("-" * 55)
    
    for i, (cov_name, stats) in enumerate(sorted_stats[:top_n]):
        before_std = abs(stats['before_matching']['std_difference'])
        after_std = abs(stats['after_matching']['std_difference'])
        improvement = stats['improvement']
        
        print(f"{cov_name:<20} {before_std:<10.3f} {after_std:<10.3f} {improvement:<12.3f}")
    
    # Overall summary
    all_improvements = [stats['improvement'] for stats in balance_stats.values()]
    mean_improvement = np.mean(all_improvements)
    max_improvement = max(all_improvements)
    
    print(f"\nOverall balance improvement:")
    print(f"  Mean improvement: {mean_improvement:.3f}")
    print(f"  Max improvement: {max_improvement:.3f}")
    print(f"  Covariates with improved balance: {sum(1 for imp in all_improvements if imp > 0)}/{len(all_improvements)}")

class EnhancedObservationalLearner(ObservationalTreatmentPatternLearner):
    """
    Enhanced version that uses matched controls instead of random sampling
    """
    
    def __init__(self, signature_loadings, processed_ids, statin_prescriptions, 
                 covariates, matched_control_indices=None, time_grid_start_age=30, gp_scripts=None):
        
        super().__init__(signature_loadings, processed_ids, statin_prescriptions, 
                        covariates, time_grid_start_age, gp_scripts=gp_scripts)
        
        self.matched_control_indices = matched_control_indices
        
    def compare_treated_vs_matched_controls(self, window=12):
        """
        Compare signature patterns between treated patients and matched controls
        """
        if self.matched_control_indices is None:
            print("No matched controls provided, using original method")
            return self.compare_treated_vs_untreated_patterns(window)
        
        treated_patterns = self.treatment_patterns['pre_treatment_signatures']
        
        # Extract matched control patterns using proper age-based time points
        control_patterns = []
        for ctrl_idx in self.matched_control_indices:
            try:
                # Get the actual time point for this matched control (not random!)
                ctrl_eid = self.processed_ids[ctrl_idx]
                
                # Try to get age from covariate_dicts if available
                if hasattr(self, 'covariate_dicts') and self.covariate_dicts is not None:
                    age_at_enroll = self.covariate_dicts.get('age_at_enroll', {}).get(int(ctrl_eid))
                else:
                    # Fallback to covariates DataFrame
                    try:
                        age_at_enroll = self.covariates[self.covariates['eid'] == ctrl_eid]['age_at_enroll'].iloc[0]
                    except:
                        age_at_enroll = None
                
                if age_at_enroll is not None and not np.isnan(age_at_enroll):
                    t0 = int(age_at_enroll - 30)  # Convert to time index
                    if t0 >= window and t0 < self.signatures.shape[2] - window:
                        pattern = self.signatures[ctrl_idx, :, t0-window:t0]
                        if pattern.shape == (self.signatures.shape[1], window):
                            control_patterns.append(pattern)
            except Exception as e:
                continue
        
        if len(control_patterns) == 0:
            return None
            
        control_patterns = np.array(control_patterns)
        
        # Compare the two groups
        comparisons = {}
        n_signatures = min(treated_patterns.shape[1], control_patterns.shape[1])
        
        for s in range(n_signatures):
            try:
                treated_levels = treated_patterns[:, s, -1]
                control_levels = control_patterns[:, s, -1]
                
                if len(treated_levels) > 0 and len(control_levels) > 0:
                    stat, p_val = mannwhitneyu(treated_levels, control_levels, 
                                             alternative='two-sided')
                    
                    comparisons[s] = {
                        'treated_mean': np.mean(treated_levels),
                        'control_mean': np.mean(control_levels),
                        'difference': np.mean(treated_levels) - np.mean(control_levels),
                        'p_value': p_val,
                        'effect_size': (np.mean(treated_levels) - np.mean(control_levels)) / 
                                      np.sqrt((np.var(treated_levels) + np.var(control_levels)) / 2)
                    }
            except Exception as e:
                continue
        
        return comparisons

    def compare_treated_vs_untreated_patterns(self, window=12):
        """
        Override parent method to use matched controls instead of random sampling
        """
        return self.compare_treated_vs_matched_controls(window)
    
    def learn_treatment_responsive_patterns_with_matched_controls(self, window=12):
        """
        Learn treatment responsive patterns using matched controls instead of random sampling
        """
        if self.matched_control_indices is None:
            print("No matched controls provided, using original method")
            return self.learn_treatment_responsive_patterns()
        
        treated_patterns = self.treatment_patterns['pre_treatment_signatures']
        
        # Extract matched control patterns using proper age-based time points
        control_patterns = []
        for ctrl_idx in self.matched_control_indices:
            try:
                # Get the actual time point for this matched control (not random!)
                ctrl_eid = self.processed_ids[ctrl_idx]
                
                # Try to get age from covariate_dicts if available
                if hasattr(self, 'covariate_dicts') and self.covariate_dicts is not None:
                    age_at_enroll = self.covariate_dicts.get('age_at_enroll', {}).get(int(ctrl_eid))
                else:
                    # Fallback to covariates DataFrame
                    try:
                        age_at_enroll = self.covariates[self.covariates['eid'] == ctrl_eid]['age_at_enroll'].iloc[0]
                    except:
                        age_at_enroll = None
                
                if age_at_enroll is not None and not np.isnan(age_at_enroll):
                    t0 = int(age_at_enroll - 30)  # Convert to time index
                    if t0 >= window and t0 < self.signatures.shape[2] - window:
                        pattern = self.signatures[ctrl_idx, :, t0-window:t0]
                        if pattern.shape == (self.signatures.shape[1], window):
                            control_patterns.append(pattern)
            except Exception as e:
                continue
        
        if len(control_patterns) == 0:
            print("Warning: No valid control patterns found, using original method")
            return self.learn_treatment_responsive_patterns()
        
        control_patterns = np.array(control_patterns)
        
        # Now use the same logic as the original method but with matched controls
        n_signatures = treated_patterns.shape[1]
        concerning_patterns = {}
        treatment_readiness_signatures = []
        
        # Analyze each signature
        for sig_idx in range(n_signatures):
            treated_sig = treated_patterns[:, sig_idx, :]  # N x window
            control_sig = control_patterns[:, sig_idx, :]  # N_control x window
            
            # Calculate trends
            treated_trends = np.polyfit(np.arange(window), treated_sig.T, 1)[0]  # Slopes
            control_trends = np.polyfit(np.arange(window), control_sig.T, 1)[0]  # Slopes
            
            # Concerning trends (upward in treated)
            concerning_trend_fraction = np.mean(treated_trends > 0)
            
            # Accelerating patterns (increasing acceleration)
            treated_accel = np.polyfit(np.arange(window), treated_sig.T, 2)[0]  # Quadratic terms
            accelerating_fraction = np.mean(treated_accel > 0)
            
            concerning_patterns[sig_idx] = {
                'concerning_trend_fraction': concerning_trend_fraction,
                'accelerating_fraction': accelerating_fraction,
                'treated_trend_mean': np.mean(treated_trends),
                'control_trend_mean': np.mean(control_trends),
                'treated_trend_std': np.std(treated_trends),
                'control_trend_std': np.std(control_trends)
            }
            
            # Calculate treatment readiness score (separation between treated and controls)
            treated_final = treated_sig[:, -1]  # Final levels
            control_final = control_sig[:, -1]  # Final levels
            
            if len(treated_final) > 0 and len(control_final) > 0:
                separation = abs(np.mean(treated_final) - np.mean(control_final)) / np.std(np.concatenate([treated_final, control_final]))
                treatment_readiness_signatures.append((sig_idx, separation))
        
        # Sort by readiness score
        treatment_readiness_signatures.sort(key=lambda x: x[1], reverse=True)
        
        # Early vs late treatment analysis (using matched controls)
        early_vs_late = self._analyze_early_vs_late_treatment_with_matched_controls()
        
        return {
            'concerning_patterns': concerning_patterns,
            'treatment_readiness_signatures': treatment_readiness_signatures,
            'early_vs_late': early_vs_late
        }
    
    def _analyze_early_vs_late_treatment_with_matched_controls(self):
        """
        Analyze early vs late treatment differences using matched controls
        """
        if self.matched_control_indices is None:
            return {'signature_differences': []}
        
        # Get treatment times and split into early/late
        treatment_times = np.array(self.treatment_patterns['treatment_times'])
        median_time = np.median(treatment_times)
        
        early_mask = treatment_times <= median_time
        late_mask = treatment_times > median_time
        
        early_patterns = self.treatment_patterns['pre_treatment_signatures'][early_mask]
        late_patterns = self.treatment_patterns['pre_treatment_signatures'][late_mask]
        
        # Get matched controls for early and late groups
        early_controls = []
        late_controls = []
        
        for i, ctrl_idx in enumerate(self.matched_control_indices):
            try:
                ctrl_eid = self.processed_ids[ctrl_idx]
                if hasattr(self, 'covariate_dicts') and self.covariate_dicts is not None:
                    age_at_enroll = self.covariate_dicts.get('age_at_enroll', {}).get(int(ctrl_eid))
                else:
                    try:
                        age_at_enroll = self.covariates[self.covariates['eid'] == ctrl_eid]['age_at_enroll'].iloc[0]
                    except:
                        age_at_enroll = None
                
                if age_at_enroll is not None and not np.isnan(age_at_enroll):
                    t0 = int(age_at_enroll - 30)
                    if t0 >= 12 and t0 < self.signatures.shape[2] - 12:
                        pattern = self.signatures[ctrl_idx, :, t0-12:t0]
                        if pattern.shape == (self.signatures.shape[1], 12):
                            # Assign to early or late based on the matched treated patient
                            if i < len(early_mask) and early_mask[i]:
                                early_controls.append(pattern)
                            elif i < len(late_mask) and late_mask[i]:
                                late_controls.append(pattern)
            except:
                continue
        
        early_controls = np.array(early_controls)
        late_controls = np.array(late_controls)
        
        # Compare early vs late
        signature_differences = []
        n_signatures = min(early_patterns.shape[1], late_patterns.shape[1])
        
        for sig_idx in range(n_signatures):
            try:
                early_mean = np.mean(early_patterns[:, sig_idx, -1])
                late_mean = np.mean(late_patterns[:, sig_idx, -1])
                diff = early_mean - late_mean
                
                # Statistical test
                from scipy.stats import mannwhitneyu
                stat, p_val = mannwhitneyu(
                    early_patterns[:, sig_idx, -1], 
                    late_patterns[:, sig_idx, -1], 
                    alternative='two-sided'
                )
                
                signature_differences.append({
                    'signature': sig_idx,
                    'difference': diff,
                    'p_value': p_val,
                    'early_mean': early_mean,
                    'late_mean': late_mean
                })
            except:
                continue
        
        # Sort by absolute difference
        signature_differences.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        return {'signature_differences': signature_differences}
    
    def plot_learned_patterns_with_matched_controls(self):
        """
        Plot learned patterns using matched controls instead of random sampling
        """
        if not hasattr(self, 'matched_control_indices') or self.matched_control_indices is None:
            print("Warning: No matched controls available, falling back to original method")
            return self.plot_learned_patterns()
        
        # Get treated patterns (pre-treatment signatures)
        treated_patterns = self.treatment_patterns['pre_treatment_signatures']
        
        # Get matched control patterns at the same time points
        control_patterns = []
        for ctrl_idx in self.matched_control_indices:
            try:
                # Use the same time window as treated patients (12 months before treatment)
                # For controls, we'll use a similar age-based time point
                ctrl_eid = self.processed_ids[ctrl_idx]
                
                # Try to get age from covariate_dicts if available
                if hasattr(self, 'covariate_dicts') and self.covariate_dicts is not None:
                    age_at_enroll = self.covariate_dicts.get('age_at_enroll', {}).get(int(ctrl_eid))
                else:
                    # Fallback to covariates DataFrame
                    try:
                        age_at_enroll = self.covariates[self.covariates['eid'] == ctrl_eid]['age_at_enroll'].iloc[0]
                    except:
                        age_at_enroll = None
                
                if age_at_enroll is not None and not np.isnan(age_at_enroll):
                    t0 = int(age_at_enroll - 30)  # Convert to time index
                    if t0 >= 12 and t0 < self.signatures.shape[2] - 12:
                        pattern = self.signatures[ctrl_idx, :, t0-12:t0]
                        if pattern.shape == (self.signatures.shape[1], 12):
                            control_patterns.append(pattern)
            except Exception as e:
                continue
        
        if len(control_patterns) == 0:
            print("Warning: No valid control patterns found, using original method")
            return self.plot_learned_patterns()
        
        control_patterns = np.array(control_patterns)
        
        # Create the same plots but with matched controls
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Learned Treatment Patterns from Observational Data (Matched Controls)', fontsize=16)
        
        # Plot 1: Pre-treatment signature clusters (same as original)
        self._plot_pre_treatment_clusters(axes[0, 0])
        
        # Plot 2: Most predictive signatures (using matched controls)
        self._plot_most_predictive_signatures_matched(axes[0, 1], treated_patterns, control_patterns)
        
        # Plot 3: Signature differences (using matched controls)
        self._plot_signature_differences_matched(axes[0, 2], treated_patterns, control_patterns)
        
        # Plot 4: Treatment initiation ages (same as original)
        self._plot_treatment_initiation_ages(axes[1, 0])
        
        # Plot 5: Concerning trends (same as original)
        self._plot_concerning_trends(axes[1, 1])
        
        # Plot 6: Early vs late treatment (same as original)
        self._plot_early_vs_late_treatment(axes[1, 2])
        
        plt.tight_layout()
        return fig
    
    def _plot_pre_treatment_clusters(self, ax):
        """Plot pre-treatment signature clusters"""
        cluster_analysis = self.discover_treatment_initiation_patterns()
        if cluster_analysis is None:
            ax.text(0.5, 0.5, 'No cluster data available', ha='center', va='center', transform=ax.transAxes)
            return
            
        n_clusters = cluster_analysis['n_clusters']
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        
        for c in range(n_clusters):
            pattern = cluster_analysis['cluster_patterns'][c]
            n_patients = pattern['n_patients']
            mean_age = pattern['mean_treatment_age']
            
            # Plot average signature for top signature
            top_sig = pattern['signature_pattern'][0, :]  # First signature
            ax.plot(range(-12, 0), top_sig, color=colors[c], linewidth=2,
                    label=f'Cluster {c} (n={n_patients}, age={mean_age:.1f})')
        
        ax.set_xlabel('Months Before Treatment')
        ax.set_ylabel('Signature Loading')
        ax.set_title('Pre-Treatment Signature Clusters')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_treatment_initiation_ages(self, ax):
        """Plot treatment initiation age distribution"""
        treatment_ages = np.array(self.treatment_patterns['treatment_times']) + self.time_start_age
        ax.hist(treatment_ages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Age at Treatment Initiation')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Observed Treatment Initiation Ages')
        ax.grid(True, alpha=0.3)
    
    def _plot_concerning_trends(self, ax):
        """Plot concerning trends before treatment using matched controls"""
        responsive_patterns = self.learn_treatment_responsive_patterns_with_matched_controls()
        if responsive_patterns is None:
            ax.text(0.5, 0.5, 'No trend data available', ha='center', va='center', transform=ax.transAxes)
            return
            
        concerning = responsive_patterns['concerning_patterns']
        signatures = list(concerning.keys())[:8]
        trend_fractions = [concerning[s]['concerning_trend_fraction'] for s in signatures]
        
        ax.bar(range(len(signatures)), trend_fractions, alpha=0.7, color='orange')
        ax.set_xticks(range(len(signatures)))
        ax.set_xticklabels([f'Sig {s}' for s in signatures])
        ax.set_ylabel('Fraction with Upward Trend')
        ax.set_title('Concerning Trends Before Treatment (Matched Controls)')
        ax.grid(True, alpha=0.3)
    
    def _plot_early_vs_late_treatment(self, ax):
        """Plot early vs late treatment comparison using matched controls"""
        responsive_patterns = self.learn_treatment_responsive_patterns_with_matched_controls()
        if responsive_patterns is None:
            ax.text(0.5, 0.5, 'No early/late data available', ha='center', va='center', transform=ax.transAxes)
            return
            
        early_late = responsive_patterns['early_vs_late']['signature_differences']
        sig_diffs = early_late[:6]  # First 6 signatures
        
        differences = [d['difference'] for d in sig_diffs]
        p_values = [d['p_value'] for d in sig_diffs]
        signatures = [d['signature'] for d in sig_diffs]
        
        colors = ['green' if p < 0.05 else 'gray' for p in p_values]
        ax.bar(range(len(signatures)), differences, color=colors, alpha=0.7)
        ax.set_xticks(range(len(signatures)))
        ax.set_xticklabels([f'Sig {s}' for s in signatures])
        ax.set_ylabel('Difference (Early - Late Treaters)')
        ax.set_title('Early vs Late Treatment Signatures (Matched Controls)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    def _plot_most_predictive_signatures_matched(self, ax, treated_patterns, control_patterns):
        """Plot most predictive signatures using matched controls"""
        # Calculate treatment readiness scores using matched controls
        n_signatures = treated_patterns.shape[1]
        readiness_scores = []
        
        for sig_idx in range(n_signatures):
            treated_levels = treated_patterns[:, sig_idx, -1]  # Final levels
            control_levels = control_patterns[:, sig_idx, -1]  # Final levels
            
            # Calculate separation score (higher = more predictive)
            if len(treated_levels) > 0 and len(control_levels) > 0:
                separation = abs(np.mean(treated_levels) - np.mean(control_levels)) / np.std(np.concatenate([treated_levels, control_levels]))
                readiness_scores.append(separation)
            else:
                readiness_scores.append(0)
        
        # Get top 5 signatures
        top_indices = np.argsort(readiness_scores)[::-1][:5]
        top_scores = [readiness_scores[i] for i in top_indices]
        
        ax.bar(range(len(top_indices)), top_scores, color='skyblue')
        ax.set_xlabel('Signature')
        ax.set_ylabel('Treatment Readiness Score')
        ax.set_title('Most Predictive Signatures (Matched Controls)')
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels([f'Sig {i}' for i in top_indices])
        ax.grid(True, alpha=0.3)
    
    def _plot_signature_differences_matched(self, ax, treated_patterns, control_patterns):
        """Plot signature differences using matched controls"""
        n_signatures = min(treated_patterns.shape[1], control_patterns.shape[1])
        differences = []
        
        for sig_idx in range(n_signatures):
            treated_mean = np.mean(treated_patterns[:, sig_idx, -1])
            control_mean = np.mean(control_patterns[:, sig_idx, -1])
            diff = treated_mean - control_mean
            differences.append(diff)
        
        colors = ['red' if d > 0 else 'blue' for d in differences]
        ax.bar(range(n_signatures), differences, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Signature')
        ax.set_ylabel('Difference (Treated - Matched Controls)')
        ax.set_title('Signature Differences: Treated vs Matched Controls')
        ax.set_xticks(range(n_signatures))
        ax.set_xticklabels([f'Sig {i}' for i in range(n_signatures)])
        ax.grid(True, alpha=0.3)

def bayesian_map_propensity_response(treated_signatures, control_signatures, outcomes=None):
    """
    MAP estimation for Bayesian propensity-response model
    """
    
    print("   Using MAP optimization for Bayesian analysis...")
    
    # Prepare data
    n_treated = len(treated_signatures)
    n_control = len(control_signatures)
    
    # Extract final signature levels
    treated_levels = treated_signatures[:, :, -1]  # N_t x K
    control_levels = control_signatures[:, :, -1]  # N_c x K
    
    # Combine into single dataset
    all_signatures = np.vstack([treated_levels, control_levels])
    treatment_status = np.concatenate([np.ones(n_treated), np.zeros(n_control)])
    
    n_patients, n_signatures = all_signatures.shape
    
    # Check if outcomes are valid before defining the function
    use_response_model = False
    if outcomes is not None:
        if len(outcomes) != len(treated_signatures):
            print(f"   Warning: outcomes shape {outcomes.shape} doesn't match treated cohort size {len(treated_signatures)}")
            print(f"   Skipping response model analysis")
        else:
            use_response_model = True
    
    def neg_log_posterior(params):
        """Negative log posterior"""
        
        # Stage 1: Propensity model parameters
        prop_intercept = params[0]
        prop_coefs = params[1:n_signatures+1]
        
        # Propensity scores
        logits = prop_intercept + np.dot(all_signatures, prop_coefs)
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # Treatment likelihood
        treatment_ll = (treatment_status * np.log(probs + 1e-10) + 
                       (1 - treatment_status) * np.log(1 - probs + 1e-10))
        
        # Priors (Normal(0,1))
        prior_ll = -0.5 * (prop_intercept**2 + np.sum(prop_coefs**2))
        
        total_ll = np.sum(treatment_ll) + prior_ll
        
        # Stage 2: Response model (if outcomes provided and valid)
        if use_response_model:
            resp_intercept = params[n_signatures+1]
            resp_coefs = params[n_signatures+2:2*n_signatures+2]
            treatment_effect = params[2*n_signatures+2]
            
            # Response model (only for treated patients)
            resp_logits = resp_intercept + np.dot(treated_levels, resp_coefs) + treatment_effect
            resp_probs = 1 / (1 + np.exp(-np.clip(resp_logits, -500, 500)))
            
            # Outcome likelihood
            outcome_ll = (outcomes * np.log(resp_probs + 1e-10) + 
                         (1 - outcomes) * np.log(1 - resp_probs + 1e-10))
            
            # Additional priors
            resp_prior_ll = -0.5 * (resp_intercept**2 + np.sum(resp_coefs**2) + treatment_effect**2)
            
            total_ll += np.sum(outcome_ll) + resp_prior_ll
        
        return -total_ll  # Negative for minimization
    
    # Initialize parameters
    if use_response_model:
        n_params = 2 * n_signatures + 3  # prop + resp parameters
    else:
        n_params = n_signatures + 1  # prop parameters only
    
    init_params = np.random.normal(0, 0.1, n_params)
    
    # Optimize
    try:
        result = minimize(neg_log_posterior, init_params, method='BFGS', 
                         options={'maxiter': 1000})
        
        # Extract MAP estimates
        map_results = {
            'propensity_intercept': result.x[0],
            'propensity_coefficients': result.x[1:n_signatures+1],
            'converged': result.success,
            'neg_log_posterior': result.fun,
            'signatures_most_predictive': np.argsort(np.abs(result.x[1:n_signatures+1]))[::-1][:5],
            'optimization_result': result
        }
        
        if use_response_model:
            map_results.update({
                'response_intercept': result.x[n_signatures+1],
                'response_coefficients': result.x[n_signatures+2:2*n_signatures+2],
                'treatment_effect': result.x[2*n_signatures+2],
                'signatures_most_responsive': np.argsort(np.abs(result.x[n_signatures+2:2*n_signatures+2]))[::-1][:5]
            })
        
        print(f"   MAP optimization converged: {result.success}")
        print(f"   Top 5 signatures for treatment propensity: {map_results['signatures_most_predictive']}")
        
        if use_response_model:
            print(f"   Estimated treatment effect: {map_results['treatment_effect']:.3f}")
            print(f"   Top 5 signatures for treatment response: {map_results['signatures_most_responsive']}")
        
        return map_results
        
    except Exception as e:
        print(f"   MAP optimization failed: {e}")
        return None

def validate_trial_reproduction(treated_outcomes, control_outcomes, follow_up_times):
    """
    Calculate HR and compare to known trial results
    
    Parameters:
    - treated_outcomes: Binary outcome data for treated patients (1=event, 0=no event)
    - control_outcomes: Binary outcome data for control patients (1=event, 0=no event)
    - follow_up_times: Follow-up times for both groups (in years)
    
    Returns:
    - Dictionary with validation results including HR, CI, p-value, and comparison to trials
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
        import warnings
        warnings.filterwarnings('ignore')
    except ImportError:
        print("Warning: lifelines not available, using basic survival analysis")
        return _basic_survival_validation(treated_outcomes, control_outcomes, follow_up_times)
    
    # Prepare data for Cox model
    import pandas as pd
    
    # Create survival dataset
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
    
    # Fit Cox proportional hazards model
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    
    # Extract results
    summary = cph.print_summary()
    
    # Get hazard ratio and confidence intervals
    hr = np.exp(cph.params_['treatment'])
    
    # Get confidence intervals - handle different lifelines versions
    try:
        hr_ci_lower = np.exp(cph.confidence_intervals_.loc['treatment', 'treatment_lower'])
        hr_ci_upper = np.exp(cph.confidence_intervals_.loc['treatment', 'treatment_upper'])
    except KeyError:
        # Try alternative column names
        try:
            hr_ci_lower = np.exp(cph.confidence_intervals_.loc['treatment', 'lower 0.95'])
            hr_ci_upper = np.exp(cph.confidence_intervals_.loc['treatment', 'upper 0.95'])
        except KeyError:
            # Fallback to basic confidence interval calculation
            hr_ci_lower = hr * 0.8  # Rough estimate
            hr_ci_upper = hr * 1.2  # Rough estimate
    
    p_value = cph.summary.loc['treatment', 'p']
    
    # Calculate concordance index (C-index)
    c_index = cph.concordance_index_
    
    # Compare to known trial results
    # Statin trials typically show HR ~0.75 for cardiovascular events
    expected_hr = 0.75
    hr_difference = hr - expected_hr
    hr_ratio = hr / expected_hr
    
    # Test if our HR is significantly different from expected
    # Use confidence interval overlap
    ci_overlaps_expected = (hr_ci_lower <= expected_hr <= hr_ci_upper)
    
    # Calculate power (simplified)
    # Power increases with sample size and effect size
    total_events = np.sum(all_outcomes)
    power_estimate = min(0.95, total_events / 100)  # Rough estimate
    
    # Validation metrics
    validation_results = {
        'hazard_ratio': hr,
        'hr_ci_lower': hr_ci_lower,
        'hr_ci_upper': hr_ci_upper,
        'p_value': p_value,
        'concordance_index': c_index,
        'expected_hr': expected_hr,
        'hr_difference': hr_difference,
        'hr_ratio': hr_ratio,
        'ci_overlaps_expected': ci_overlaps_expected,
        'power_estimate': power_estimate,
        'n_treated': n_treated,
        'n_control': n_control,
        'total_events': total_events,
        'cox_model': cph,
        'validation_passed': ci_overlaps_expected and p_value < 0.05
    }
    
    # Print validation summary
    print("\n=== TRIAL REPRODUCTION VALIDATION ===")
    print(f"Hazard Ratio: {hr:.3f} (95% CI: {hr_ci_lower:.3f}-{hr_ci_upper:.3f})")
    print(f"P-value: {p_value:.4f}")
    print(f"Expected HR from trials: {expected_hr:.3f}")
    print(f"Difference from expected: {hr_difference:.3f}")
    print(f"CI overlaps expected: {ci_overlaps_expected}")
    print(f"Concordance Index: {c_index:.3f}")
    print(f"Total events: {total_events}")
    print(f"Validation passed: {validation_results['validation_passed']}")
    
    if validation_results['validation_passed']:
        print("✓ Matching validation successful - results consistent with trial data")
    else:
        print("⚠ Matching validation failed - results inconsistent with trial data")
        if not ci_overlaps_expected:
            print("  - Confidence interval does not overlap expected HR")
        if p_value >= 0.05:
            print("  - Effect not statistically significant")
    
    return validation_results

def _basic_survival_validation(treated_outcomes, control_outcomes, follow_up_times):
    """
    Basic survival validation when lifelines is not available
    """
    from scipy.stats import chi2_contingency
    
    # Simple chi-square test for event rates
    treated_events = np.sum(treated_outcomes)
    control_events = np.sum(control_outcomes)
    treated_total = len(treated_outcomes)
    control_total = len(control_outcomes)
    
    # Create contingency table
    contingency_table = np.array([
        [treated_events, treated_total - treated_events],
        [control_events, control_total - control_events]
    ])
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate risk ratio (approximate HR)
    treated_rate = treated_events / treated_total
    control_rate = control_events / control_total
    risk_ratio = treated_rate / control_rate if control_rate > 0 else np.inf
    
    # Calculate event rates
    treated_event_rate = treated_events / treated_total
    control_event_rate = control_events / control_total
    
    validation_results = {
        'risk_ratio': risk_ratio,
        'p_value': p_value,
        'treated_event_rate': treated_event_rate,
        'control_event_rate': control_event_rate,
        'n_treated': treated_total,
        'n_control': control_total,
        'total_events': treated_events + control_events,
        'validation_passed': p_value < 0.05 and risk_ratio < 1.0
    }
    
    print("\n=== BASIC TRIAL REPRODUCTION VALIDATION ===")
    print(f"Risk Ratio: {risk_ratio:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Treated event rate: {treated_event_rate:.3f}")
    print(f"Control event rate: {control_event_rate:.3f}")
    print(f"Validation passed: {validation_results['validation_passed']}")
    
    return validation_results

def comprehensive_treatment_analysis(signature_loadings, processed_ids, 
                                   statin_prescriptions, covariates,
                                   covariate_dicts=None, sig_indices=None,
                                   outcomes=None, event_idx=None, event_indices=None, gp_scripts=None):
    """
    Complete comprehensive analysis pipeline
    
    Parameters:
    - signature_loadings: N x K x T signature array
    - processed_ids: Array of patient IDs
    - statin_prescriptions: DataFrame with prescription data
    - covariates: DataFrame with covariate data
    - covariate_dicts: Dictionary format covariates for matching
    - sig_indices: Optional signature indices to use
    - outcomes: Optional outcome array Y[patient, event_type, time] for validation
    - event_idx: Index of the specific event type to validate (e.g., single disease)
    - event_indices: List of indices for composite events (e.g., [112, 113, 114, 115, 116] for ASCVD)
    - gp_scripts: Optional GP scripts DataFrame
    
    Returns:
    - Dictionary with all analysis results
    """
    
    print("=== COMPREHENSIVE TREATMENT PATTERN ANALYSIS ===\n")
    
    # Step 1: Extract treatment patterns
    print("1. Extracting treatment patterns...")
    learner = ObservationalTreatmentPatternLearner(
        signature_loadings, processed_ids, statin_prescriptions, covariates, gp_scripts=gp_scripts
    )
    
    treated_eids = learner.treatment_patterns['treated_patients']
    treated_times = learner.treatment_patterns['treatment_times']
    never_treated_eids = learner.treatment_patterns['never_treated']
    
    # Filter to only patients who have prescription data (the 222K in gp_scripts)
    # This excludes the 178K patients with no prescription records
    if gp_scripts is not None:
        all_patients_with_prescriptions = set(gp_scripts['eid'].unique())
    else:
        # Fallback to statin prescriptions if gp_scripts not provided
        all_patients_with_prescriptions = set(statin_prescriptions['eid'].unique())
    
    treated_patients_set = set(treated_eids)
    
    # Use clean control definition: all GP patients minus ALL statin users
    all_statins_eids = set(statin_prescriptions['eid'].unique())
    valid_never_treated = [eid for eid in all_patients_with_prescriptions 
                      if eid not in all_statins_eids and eid in processed_ids]
    
    print(f"   Found {len(treated_eids)} treated patients")
    print(f"   Found {len(never_treated_eids)} total never-treated patients")
    print(f"   Found {len(valid_never_treated)} never-treated patients with signature data")
    
    # Update the never_treated list to only valid ones
    never_treated_eids = valid_never_treated
    
    # Step 2: Build features for matching
    print("\n2. Building features for matching...")
    
    if covariate_dicts is None:
        # Create simple covariate dict from DataFrame
        covariate_dicts = {}
        if 'birth_year' in covariates.columns:
            covariate_dicts['age'] = dict(zip(covariates['eid'], 
                                            2025 - covariates['birth_year']))
    
    # Build features for treated patients
    treated_features, treated_indices, treated_eids_matched = build_features(
        treated_eids, treated_times, processed_ids, signature_loadings, 
        covariate_dicts, sig_indices=sig_indices, is_treated=True, treatment_dates=treated_times
    )
    
    # Build features for control patients using proper age-based time points
    # Get age at enrollment for controls and convert to time indices
    control_eids_for_matching = never_treated_eids[:len(treated_eids)*2]  # 2:1 ratio
    control_t0s = []
    valid_control_eids = []
    
    for eid in control_eids_for_matching:
        try:
            # Get age at enrollment for this control
            age_at_enroll = covariate_dicts['age_at_enroll'].get(int(eid))
            if age_at_enroll is not None and not np.isnan(age_at_enroll):
                # Convert age to time index (age - 30, since time grid starts at age 30)
                t0 = int(age_at_enroll - 30)
                if t0 >= 10:  # Need at least 10 years of history for matching
                    control_t0s.append(t0)
                    valid_control_eids.append(eid)
        except:
            continue
    
    control_features, control_indices, control_eids_matched = build_features(
        valid_control_eids, control_t0s, processed_ids, 
        signature_loadings, covariate_dicts, sig_indices=sig_indices, is_treated=False, treatment_dates=None
    )
    
    print(f"   Built features for {len(treated_features)} treated patients")
    print(f"   Built features for {len(control_features)} control patients")
    
    # Step 3: Perform matching
    print("\n3. Performing nearest neighbor matching...")
    
    (matched_treated_indices, matched_control_indices, 
     matched_treated_eids, matched_control_eids) = perform_nearest_neighbor_matching(
        treated_features, control_features, treated_indices, control_indices,
        treated_eids_matched, control_eids_matched
    )
    
    print(f"   Successfully matched {len(matched_treated_indices)} pairs")
    
    # Step 4: Assess matching balance
    print("\n4. Assessing matching balance...")
    
    # Get the actual covariate data for the matched patients
    matched_treated_eids = [processed_ids[i] for i in matched_treated_indices]
    matched_control_eids = [processed_ids[i] for i in matched_control_indices]
    
    # Extract key covariates for balance assessment
    key_covariates = ['age_at_enroll', 'tchol', 'hdl', 'SBP', 'pce_goff', 'sex', 
                     'dm2_prev', 'antihtnbase', 'dm1_prev']
    
    balance_stats = {}
    for cov_name in key_covariates:
        if cov_name in covariate_dicts:
            # Get values for treated and control groups
            treated_values = []
            control_values = []
            
            for eid in matched_treated_eids:
                val = covariate_dicts[cov_name].get(int(eid))
                if val is not None and not np.isnan(val):
                    treated_values.append(val)
            
            for eid in matched_control_eids:
                val = covariate_dicts[cov_name].get(int(eid))
                if val is not None and not np.isnan(val):
                    control_values.append(val)
            
            if len(treated_values) > 0 and len(control_values) > 0:
                # Calculate SMD
                def compute_smd(x1, x0):
                    m1, m0 = np.nanmean(x1), np.nanmean(x0)
                    s1, s0 = np.nanstd(x1), np.nanstd(x0)
                    return np.abs(m1 - m0) / np.sqrt((s1**2 + s0**2) / 2)
                
                smd = compute_smd(treated_values, control_values)
                
                balance_stats[cov_name] = {
                    'treated_mean': np.mean(treated_values),
                    'control_mean': np.mean(control_values),
                    'treated_std': np.std(treated_values),
                    'control_std': np.std(control_values),
                    'smd': smd,
                    'n_treated': len(treated_values),
                    'n_control': len(control_values)
                }
    
    # Print balance summary
    print("\n=== MATCHING BALANCE ASSESSMENT ===")
    print(f"{'Covariate':<15} {'Treated Mean':<12} {'Control Mean':<12} {'SMD':<8}")
    print("-" * 50)
    
    for cov_name, stats in balance_stats.items():
        print(f"{cov_name:<15} {stats['treated_mean']:<12.3f} {stats['control_mean']:<12.3f} {stats['smd']:<8.3f}")
    
    # Check how many covariates have SMD < 0.1 (good balance)
    well_balanced = sum(1 for stats in balance_stats.values() if stats['smd'] < 0.1)
    print(f"\nCovariates with SMD < 0.1 (good balance): {well_balanced}/{len(balance_stats)}")
    
    if well_balanced == len(balance_stats):
        print("✓ Excellent balance achieved!")
    elif well_balanced >= len(balance_stats) * 0.8:
        print("✓ Good balance achieved")
    else:
        print("⚠ Balance could be improved")

    # Step 5: Enhanced observational pattern learning
    print("\n5. Learning observational patterns with matched controls...")
    
    enhanced_learner = EnhancedObservationalLearner(
        signature_loadings, processed_ids, statin_prescriptions, covariates,
        matched_control_indices=matched_control_indices, gp_scripts=gp_scripts
    )
    
    # Pass covariate_dicts to the enhanced learner for proper age access
    enhanced_learner.covariate_dicts = covariate_dicts
    
    # Learn patterns using matched controls
    cluster_analysis = enhanced_learner.discover_treatment_initiation_patterns()
    responsive_patterns = enhanced_learner.learn_treatment_responsive_patterns_with_matched_controls()
    matched_comparison = enhanced_learner.compare_treated_vs_matched_controls()
    predictor = enhanced_learner.build_treatment_readiness_predictor()
    
    if predictor:
        print(f"   Treatment readiness model AUC: {predictor['cv_auc']:.3f}")
    
    # Step 6: Bayesian MAP causal inference
    print("\n6. Performing Bayesian MAP propensity-response analysis...")
    
    # Extract signature patterns for matched cohorts
    all_treated_patterns = enhanced_learner.treatment_patterns['pre_treatment_signatures']

    # Subset to only the matched treated patients
    matched_treated_patterns = all_treated_patterns[:len(matched_treated_indices)]
    print(f"   Subsetting treated patterns: {all_treated_patterns.shape} -> {matched_treated_patterns.shape}")

    # Extract control patterns using proper time points (not random)
    control_patterns = []
    for ctrl_idx in matched_control_indices:
        try:
            # Get the actual time point for this matched control
            ctrl_eid = processed_ids[ctrl_idx]
            age_at_enroll = covariate_dicts['age_at_enroll'].get(int(ctrl_eid))
            if age_at_enroll is not None and not np.isnan(age_at_enroll):
                t0 = int(age_at_enroll - 30)  # Convert to time index
                if t0 >= 12 and t0 < signature_loadings.shape[2] - 12:
                    pattern = signature_loadings[ctrl_idx, :, t0-12:t0]
                    if pattern.shape == (signature_loadings.shape[1], 12):
                        control_patterns.append(pattern)
        except:
            continue

    control_patterns = np.array(control_patterns[:len(matched_treated_patterns)])

    # SUBSET OUTCOMES TO MATCHED TREATED PATIENTS
    if outcomes is not None:
        # Convert to numpy if needed
        if hasattr(outcomes, 'detach'):
            outcomes_np = outcomes.detach().cpu().numpy()
        else:
            outcomes_np = outcomes
        
        # Get the original treated indices (before matching)
        original_treated_indices = treated_indices  # This should be available from Step 2
        
        # Subset outcomes to only the matched treated patients
        matched_treated_outcomes_full = outcomes_np[original_treated_indices]
        print(f"   Subsetting outcomes: {outcomes_np.shape} -> {matched_treated_outcomes_full.shape}")
        
        # Extract specific outcomes for the event we're analyzing
        if event_indices is not None:
            # For composite events (like ASCVD), check if any event occurred
            matched_treated_outcomes = np.any(matched_treated_outcomes_full[:, event_indices, :] > 0, axis=(1, 2))
        elif event_idx is not None:
            # For single event
            matched_treated_outcomes = np.any(matched_treated_outcomes_full[:, event_idx, :] > 0, axis=1)
        else:
            # For any event
            matched_treated_outcomes = np.any(matched_treated_outcomes_full > 0, axis=(1, 2))
        
        print(f"   Extracted binary outcomes shape: {matched_treated_outcomes.shape}")
    else:
        matched_treated_outcomes = None

    # Run MAP-based Bayesian analysis with subsetted outcomes
    map_results = bayesian_map_propensity_response(
        matched_treated_patterns, control_patterns, outcomes=matched_treated_outcomes
    )
    
    if map_results and map_results['converged']:
        print(f"   MAP optimization successful!")
    else:
        print(f"   MAP optimization had issues")
        map_results = None
    
    # Step 7: Trial reproduction validation
    print("\n7. Validating matching with trial reproduction...")
    
    # If outcomes are provided, validate our matching approach
    validation_results = None
    if event_indices is not None:
        print(f"   Validating composite event with indices {event_indices}")
    elif event_idx is not None:
        print(f"   Validating single event type at index {event_idx}")
    else:
        print("   Validating any event occurrence")
    if outcomes is not None:
        # Convert PyTorch tensor to numpy if needed
        if hasattr(outcomes, 'detach'):
            outcomes = outcomes.detach().cpu().numpy()
        
        # Extract outcomes for matched cohorts
        treated_outcomes_matched = []
        control_outcomes_matched = []
        follow_up_times_matched = []
        
        # Get outcomes for treated patients
        for treated_idx in matched_treated_indices:
            # Get the treatment time for this patient
            treated_eid = processed_ids[treated_idx]
            treatment_time = None
            
            # Find treatment time from the original treatment patterns
            for i, eid in enumerate(enhanced_learner.treatment_patterns['treated_patients']):
                if eid == treated_eid:
                    treatment_time = enhanced_learner.treatment_patterns['treatment_times'][i]
                    break
            
            if treatment_time is not None:
                # Check if patient has outcome data
                if treated_idx < outcomes.shape[0]:
                    # Look for specific event type after treatment time
                    if event_indices is not None:
                        # Composite event - check if any of the specified events occurred
                        post_treatment_outcomes = outcomes[treated_idx, event_indices, int(treatment_time):]
                        post_treatment_outcomes = np.any(post_treatment_outcomes > 0, axis=0)
                    elif event_idx is not None:
                        # Single event
                        post_treatment_outcomes = outcomes[treated_idx, event_idx, int(treatment_time):]
                    else:
                        # If no specific event specified, use any event
                        post_treatment_outcomes = outcomes[treated_idx, :, int(treatment_time):]
                        post_treatment_outcomes = np.any(post_treatment_outcomes > 0, axis=0)
                    
                    event_occurred = np.any(post_treatment_outcomes > 0)
                    
                    if event_occurred:
                        # Find time to first event
                        event_times = np.where(post_treatment_outcomes > 0)[0]  # Time dimension
                        time_to_event = event_times[0] if len(event_times) > 0 else 5.0
                    else:
                        # Censored at end of follow-up
                        time_to_event = min(5.0, outcomes.shape[2] - int(treatment_time))
                    
                    treated_outcomes_matched.append(int(event_occurred))
                    follow_up_times_matched.append(time_to_event)
        
        # Get outcomes for control patients
        for control_idx in matched_control_indices:
            control_eid = processed_ids[control_idx]
            
            # For controls, use their age-based time point as "treatment" time
            age_at_enroll = covariate_dicts['age_at_enroll'].get(int(control_eid))
            if age_at_enroll is not None and not np.isnan(age_at_enroll):
                control_time = int(age_at_enroll - 30)  # Convert to time index
                
                if control_idx < outcomes.shape[0] and control_time < outcomes.shape[2]:
                    # Look for specific event type after control time
                    if event_indices is not None:
                        # Composite event - check if any of the specified events occurred
                        post_control_outcomes = outcomes[control_idx, event_indices, control_time:]
                        post_control_outcomes = np.any(post_control_outcomes > 0, axis=0)
                    elif event_idx is not None:
                        # Single event
                        post_control_outcomes = outcomes[control_idx, event_idx, control_time:]
                    else:
                        # If no specific event specified, use any event
                        post_control_outcomes = outcomes[control_idx, :, control_time:]
                        post_control_outcomes = np.any(post_control_outcomes > 0, axis=0)
                    
                    event_occurred = np.any(post_control_outcomes > 0)
                    
                    if event_occurred:
                        # Find time to first event
                        event_times = np.where(post_control_outcomes > 0)[0]
                        time_to_event = event_times[0] if len(event_times) > 0 else 5.0
                    else:
                        # Censored at end of follow-up
                        time_to_event = min(5.0, outcomes.shape[2] - control_time)
                    
                    control_outcomes_matched.append(int(event_occurred))
                    follow_up_times_matched.append(time_to_event)
        
        if len(treated_outcomes_matched) > 10 and len(control_outcomes_matched) > 10:
            validation_results = validate_trial_reproduction(
                np.array(treated_outcomes_matched),
                np.array(control_outcomes_matched),
                np.array(follow_up_times_matched)
            )
        else:
            print("   Insufficient outcome data for validation")
    else:
        print("   No outcome data provided for validation")
    
    # Step 8: Visualization with matched controls
    print("\n8. Creating visualizations with matched controls...")
    
    # Override the plotting to use matched controls
    enhanced_learner.matched_control_indices = matched_control_indices
    enhanced_learner.matched_treated_indices = matched_treated_indices
    
    # Create custom plotting that uses matched controls
    fig = enhanced_learner.plot_learned_patterns_with_matched_controls()
    
    # Compile results
    results = {
        'enhanced_learner': enhanced_learner,
        'matched_treated_indices': matched_treated_indices,
        'matched_control_indices': matched_control_indices,
        'cluster_analysis': cluster_analysis,
        'responsive_patterns': responsive_patterns,
        'matched_comparison': matched_comparison,
        'predictor': predictor,
        'bayesian_map_results': map_results,
        'validation_results': validation_results,
        'visualization': fig
    }
    
    print("\n=== ANALYSIS COMPLETE ===")
    
    return results

# Example usage function
def run_comprehensive_analysis(signature_loadings, processed_ids, statin_prescriptions, 
                             covariates, gp_scripts=None, **kwargs):
    """
    Convenience function to run the complete analysis
    """
    return comprehensive_treatment_analysis(
        signature_loadings, processed_ids, statin_prescriptions, 
        covariates, gp_scripts=gp_scripts, **kwargs
    )

if __name__ == "__main__":
    print("Comprehensive Treatment Analysis Pipeline")
    print("Import this module and use run_comprehensive_analysis() function")