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

def build_features(eids, t0s, processed_ids, thetas, covariate_dicts, sig_indices=None):
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
    
    for eid, t0 in zip(eids, t0s):
        try:
            idx = np.where(processed_ids == int(eid))[0][0]
        except Exception:
            continue
        if t0 < window:
            continue  # Not enough history
        t0_int = int(t0)
        sig_traj = thetas[idx, sig_indices, t0_int-window:t0_int].flatten()
        if sig_traj.shape[0] != expected_length:
            continue  # Skip if not the right length
        
        # Check for NaN in signature trajectory
        if np.any(np.isnan(sig_traj)):
            continue  # Skip if signature has NaN values
        
        # Extract comprehensive covariates with proper NaN handling
        age = covariate_dicts['age_at_enroll'].get(int(eid), 57)
        if np.isnan(age) or age is None:
            age = 57  # Fallback age
            
        sex = covariate_dicts['sex'].get(int(eid), 0)
        if np.isnan(sex) or sex is None:
            sex = 0
        sex = int(sex)
        
        dm2 = covariate_dicts['dm2_prev'].get(int(eid), 0)
        if np.isnan(dm2) or dm2 is None:
            dm2 = 0
            
        antihtn = covariate_dicts['antihtnbase'].get(int(eid), 0)
        if np.isnan(antihtn) or antihtn is None:
            antihtn = 0
            
        dm1 = covariate_dicts['dm1_prev'].get(int(eid), 0)
        if np.isnan(dm1) or dm1 is None:
            dm1 = 0
            
        smoke = encode_smoking(covariate_dicts['smoke'].get(int(eid), None))
        if np.any(np.isnan(smoke)):
            smoke = [0, 0, 0]  # Default to "Never" smoking
            
        ldl_prs = covariate_dicts.get('ldl_prs', {}).get(int(eid), 0)
        if np.isnan(ldl_prs) or ldl_prs is None:
            ldl_prs = 0
            
        cad_prs = covariate_dicts.get('cad_prs', {}).get(int(eid), 0)
        if np.isnan(cad_prs) or cad_prs is None:
            cad_prs = 0
            
        tchol = covariate_dicts.get('tchol', {}).get(int(eid), 0)
        if np.isnan(tchol) or tchol is None:
            tchol = 0
            
        hdl = covariate_dicts.get('hdl', {}).get(int(eid), 0)
        if np.isnan(hdl) or hdl is None:
            hdl = 0
            
        sbp = covariate_dicts.get('SBP', {}).get(int(eid), 0)
        if np.isnan(sbp) or sbp is None:
            sbp = 0
            
        pce_goff = covariate_dicts.get('pce_goff', {}).get(int(eid), 0.09)
        if np.isnan(pce_goff) or pce_goff is None:
            pce_goff = 0.09
        
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
        
        # Extract matched control patterns
        control_patterns = []
        for ctrl_idx in self.matched_control_indices:
            try:
                # Sample a similar time window for controls
                valid_times = list(range(window, min(40, self.signatures.shape[2] - window)))
                if len(valid_times) == 0:
                    continue
                    
                sample_time = np.random.choice(valid_times)
                pattern = self.signatures[ctrl_idx, :, sample_time-window:sample_time]
                
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

def bayesian_map_propensity_response(treated_signatures, control_signatures, outcomes=None):
    """
    MAP estimation for Bayesian propensity-response model
    Uses optimization instead of MCMC - leveraging your existing MAP approach
    
    Parameters:
    - treated_signatures: Signature patterns for treated patients (N_t x K x T)
    - control_signatures: Signature patterns for control patients (N_c x K x T)
    - outcomes: Optional outcome data for treated patients
    
    Returns:
    - map_results: Dictionary with MAP estimates and diagnostics
    """
    
    print("   Using MAP optimization for Bayesian analysis...")
    
    # Prepare data like your clustering code
    n_treated = len(treated_signatures)
    n_control = len(control_signatures)
    
    # Extract final signature levels
    treated_levels = treated_signatures[:, :, -1]  # N_t x K
    control_levels = control_signatures[:, :, -1]  # N_c x K
    
    # Combine into single dataset
    all_signatures = np.vstack([treated_levels, control_levels])
    treatment_status = np.concatenate([np.ones(n_treated), np.zeros(n_control)])
    
    n_patients, n_signatures = all_signatures.shape
    
    def neg_log_posterior(params):
        """Negative log posterior - same optimization pattern as your MAP work"""
        
        # Stage 1: Propensity model parameters
        prop_intercept = params[0]
        prop_coefs = params[1:n_signatures+1]
        
        # Propensity scores
        logits = prop_intercept + np.dot(all_signatures, prop_coefs)
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))  # Stable sigmoid
        
        # Treatment likelihood
        treatment_ll = (treatment_status * np.log(probs + 1e-10) + 
                       (1 - treatment_status) * np.log(1 - probs + 1e-10))
        
        # Priors (Normal(0,1))
        prior_ll = -0.5 * (prop_intercept**2 + np.sum(prop_coefs**2))
        
        total_ll = np.sum(treatment_ll) + prior_ll
        
        # Stage 2: Response model (if outcomes provided)
        if outcomes is not None:
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
    if outcomes is not None:
        n_params = 2 * n_signatures + 3  # prop + resp parameters
    else:
        n_params = n_signatures + 1  # prop parameters only
    
    init_params = np.random.normal(0, 0.1, n_params)
    
    # Optimize (like your clustering approach)
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
        
        if outcomes is not None:
            map_results.update({
                'response_intercept': result.x[n_signatures+1],
                'response_coefficients': result.x[n_signatures+2:2*n_signatures+2],
                'treatment_effect': result.x[2*n_signatures+2],
                'signatures_most_responsive': np.argsort(np.abs(result.x[n_signatures+2:2*n_signatures+2]))[::-1][:5]
            })
        
        print(f"   MAP optimization converged: {result.success}")
        print(f"   Top 5 signatures for treatment propensity: {map_results['signatures_most_predictive']}")
        
        if outcomes is not None:
            print(f"   Estimated treatment effect: {map_results['treatment_effect']:.3f}")
            print(f"   Top 5 signatures for treatment response: {map_results['signatures_most_responsive']}")
        
        return map_results
        
    except Exception as e:
        print(f"   MAP optimization failed: {e}")
        return None

def comprehensive_treatment_analysis(signature_loadings, processed_ids, 
                                   statin_prescriptions, covariates,
                                   covariate_dicts=None, sig_indices=None,
                                   outcomes=None, gp_scripts=None):
    """
    Complete comprehensive analysis pipeline
    
    Parameters:
    - signature_loadings: N x K x T signature array
    - processed_ids: Array of patient IDs
    - statin_prescriptions: DataFrame with prescription data
    - covariates: DataFrame with covariate data
    - covariate_dicts: Dictionary format covariates for matching
    - sig_indices: Optional signature indices to use
    - outcomes: Optional outcome data for Bayesian analysis
    
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
    
    # Controls = patients with prescriptions but NOT statins, AND have signature data
    valid_never_treated = [eid for eid in all_patients_with_prescriptions 
                          if eid not in treated_patients_set and eid in processed_ids]
    
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
        covariate_dicts, sig_indices=sig_indices
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
        signature_loadings, covariate_dicts, sig_indices=sig_indices
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
    
    # Step 4: Enhanced observational pattern learning
    print("\n4. Learning observational patterns with matched controls...")
    
    enhanced_learner = EnhancedObservationalLearner(
        signature_loadings, processed_ids, statin_prescriptions, covariates,
        matched_control_indices=matched_control_indices, gp_scripts=gp_scripts
    )
    
    # Learn patterns
    cluster_analysis = enhanced_learner.discover_treatment_initiation_patterns()
    responsive_patterns = enhanced_learner.learn_treatment_responsive_patterns()
    matched_comparison = enhanced_learner.compare_treated_vs_matched_controls()
    predictor = enhanced_learner.build_treatment_readiness_predictor()
    
    if predictor:
        print(f"   Treatment readiness model AUC: {predictor['cv_auc']:.3f}")
    
    # Step 5: Bayesian MAP causal inference
    print("\n5. Performing Bayesian MAP propensity-response analysis...")
    
    # Extract signature patterns for matched cohorts
    treated_patterns = enhanced_learner.treatment_patterns['pre_treatment_signatures']
    
    # Extract control patterns
    control_patterns = []
    for ctrl_idx in matched_control_indices:
        try:
            sample_time = np.random.randint(12, min(40, signature_loadings.shape[2] - 12))
            pattern = signature_loadings[ctrl_idx, :, sample_time-12:sample_time]
            if pattern.shape == (signature_loadings.shape[1], 12):
                control_patterns.append(pattern)
        except:
            continue
    
    control_patterns = np.array(control_patterns[:len(treated_patterns)])
    
    # Run MAP-based Bayesian analysis
    map_results = bayesian_map_propensity_response(
        treated_patterns, control_patterns, outcomes=outcomes
    )
    
    if map_results and map_results['converged']:
        print(f"   MAP optimization successful!")
    else:
        print(f"   MAP optimization had issues")
        map_results = None
    
    # Step 6: Visualization
    print("\n6. Creating visualizations...")
    fig = enhanced_learner.plot_learned_patterns()
    
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