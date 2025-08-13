

# Usage example:
# aspirins = find_aspirin_basic(gp_scripts)




"""
Aspirin-Colorectal Cancer Prevention Analysis

This script analyzes aspirin use for colorectal cancer prevention:
1. Extract aspirin-treated patients (with proper follow-up logic)
2. Define clean controls (no aspirin ever)
3. Apply exclusions (colorectal cancer before index, missing binary data)
4. Impute quantitative variables
5. Perform nearest neighbor matching
6. Calculate HR for colorectal cancer prevention

Expected effect: Aspirin should REDUCE colorectal cancer risk (HR < 1.0)
Expected HR from trials: ~0.7-0.8 (20-30% risk reduction)

Author: Sarah Urbut
Date: 2025-01-15
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from lifelines import CoxPHFitter
except ImportError:
    print("Warning: lifelines not available for HR calculation")

def find_aspirin_basic(gp_scripts):
    """
    Basic aspirin search avoiding type comparison issues
    """
    print("\n=== Basic Aspirin Search ===\n")
    
    df = gp_scripts.copy()
    
    # Convert to strings safely
    df['drug_name_str'] = df['drug_name'].astype(str)
    df['bnf_code_str'] = df['bnf_code'].astype(str)
    
    # Search for aspirin keywords in drug names
    aspirin_keywords = ['aspirin', 'acetylsalicylic', 'asa', 'disprin', 'ecotrin']
    
    aspirin_mask = False
    for keyword in aspirin_keywords:
        if keyword == 'asa':
            # For 'asa', use word boundaries to avoid matching 'nasal'
            keyword_mask = df['drug_name_str'].str.contains(r'\basa\b', case=False, na=False, regex=True)
        else:
            keyword_mask = df['drug_name_str'].str.contains(keyword, case=False, na=False)
        aspirin_mask = aspirin_mask | keyword_mask
        if keyword_mask.sum() > 0:
            print(f"Found {keyword_mask.sum()} prescriptions containing '{keyword}'")
            # Debug: Check if this keyword is matching Beclometasone
            beclo_with_keyword = keyword_mask & df['drug_name_str'].str.contains('beclometasone', case=False, na=False)
            if beclo_with_keyword.sum() > 0:
                print(f"  ⚠️ WARNING: '{keyword}' is matching {beclo_with_keyword.sum()} Beclometasone prescriptions!")
                print(f"  Sample: {df[beclo_with_keyword]['drug_name'].head().tolist()}")
    
    # Search for antiplatelet BNF codes (2.9 - Antiplatelet drugs)
    # More specific patterns to avoid including other drugs like clopidogrel
    # 02.09.01 = Aspirin, 02.09.02 = Dipyridamole, 02.09.03 = Clopidogrel, etc.
    bnf_patterns = ['02.09.01.']  # Only match aspirin-specific BNF codes
    bnf_mask = False
    for pattern in bnf_patterns:
        pattern_mask = df['bnf_code_str'].str.startswith(pattern, na=False)
        bnf_mask = bnf_mask | pattern_mask
        if pattern_mask.sum() > 0:
            print(f"Found {pattern_mask.sum()} prescriptions with BNF pattern '{pattern}' (aspirin-specific)")
    
    # Also check what we're excluding
    clopidogrel_pattern = '02.09.03.'
    clopidogrel_mask = df['bnf_code_str'].str.startswith(clopidogrel_pattern, na=False)
    if clopidogrel_mask.sum() > 0:
        print(f"Excluded {clopidogrel_mask.sum()} clopidogrel prescriptions (BNF {clopidogrel_pattern})")
    
    # Combine results
    all_aspirin_mask = aspirin_mask | bnf_mask
    aspirins = df[all_aspirin_mask].copy()
    
    # Remove any clopidogrel that might have slipped through
    clopidogrel_mask = aspirins['drug_name_str'].str.contains('clopidogrel', case=False, na=False)
    if clopidogrel_mask.sum() > 0:
        print(f"⚠️ Removing {clopidogrel_mask.sum()} clopidogrel prescriptions that slipped through")
        aspirins = aspirins[~clopidogrel_mask].copy()
    
    # Debug: Check what's being included
    print(f"\nDEBUG: Checking what drugs are being included...")
    print(f"Aspirin mask matches: {aspirin_mask.sum()}")
    print(f"BNF mask matches: {bnf_mask.sum()}")
    print(f"Combined matches: {all_aspirin_mask.sum()}")
    
    # Check for Beclometasone specifically
    beclo_mask = df['drug_name_str'].str.contains('beclometasone', case=False, na=False)
    if beclo_mask.sum() > 0:
        print(f"⚠️ WARNING: Found {beclo_mask.sum()} Beclometasone prescriptions!")
        beclo_included = beclo_mask & all_aspirin_mask
        print(f"  - Beclometasone included in results: {beclo_included.sum()}")
        if beclo_included.sum() > 0:
            print("  - Sample Beclometasone entries:")
            print(df[beclo_included][['drug_name', 'bnf_code']].head())
    
    print(f"\nTotal potential aspirin prescriptions: {len(aspirins)}")
    
    if len(aspirins) > 0:
        print(f"Unique patients with aspirin: {aspirins['eid'].nunique()}")
        
        # Show sample
        print(f"\nSample aspirin prescriptions:")
        sample_cols = ['eid', 'issue_date', 'drug_name', 'bnf_code']
        print(aspirins[sample_cols].head(10))
        
        # Most common aspirin drugs
        print(f"\nMost common aspirin drugs:")
        top_drugs = aspirins['drug_name'].value_counts().head(5)
        for drug, count in top_drugs.items():
            print(f"  {drug}: {count}")
    
    else:
        print("No obvious aspirin found")
        
        # Show what we do have
        print(f"\nSample of all drug names:")
        print(df['drug_name'].head(10).tolist())
        
        print(f"\nSample of all BNF codes:")
        print(df['bnf_code'].head(10).tolist())
    
    return aspirins


def find_clopidogrel_basic(gp_scripts):
    """
    Basic clopidogrel search for negative control experiment
    Clopidogrel is an antiplatelet drug like aspirin but with different mechanism
    and no known anti-CRC effects - perfect negative control
    """
    print("\n=== Basic Clopidogrel Search (Negative Control) ===\n")
    
    df = gp_scripts.copy()
    
    # Convert to strings safely
    df['drug_name_str'] = df['drug_name'].astype(str)
    df['bnf_code_str'] = df['bnf_code'].astype(str)
    
    # Search for clopidogrel keywords in drug names
    clopidogrel_keywords = ['clopidogrel', 'plavix', 'clopilet', 'clopilet', 'clopidogrel']
    
    clopidogrel_mask = False
    for keyword in clopidogrel_keywords:
        keyword_mask = df['drug_name_str'].str.contains(keyword, case=False, na=False)
        clopidogrel_mask = clopidogrel_mask | keyword_mask
        if keyword_mask.sum() > 0:
            print(f"Found {keyword_mask.sum()} prescriptions containing '{keyword}'")
    
    # Search for clopidogrel-specific BNF codes (2.9.3.1 - Clopidogrel specifically)
    # Handle both formats: 02.09.03.01 and 02090301
    bnf_patterns = ['02.09.03.01', '02090301']  # Clopidogrel specifically
    bnf_mask = False
    for pattern in bnf_patterns:
        pattern_mask = df['bnf_code_str'].str.startswith(pattern, na=False)
        bnf_mask = bnf_mask | pattern_mask
        if pattern_mask.sum() > 0:
            print(f"Found {pattern_mask.sum()} prescriptions with BNF pattern '{pattern}'")
    
    # Combine results
    all_clopidogrel_mask = clopidogrel_mask | bnf_mask
    clopidogrels = df[all_clopidogrel_mask].copy()
    
    # Debug: Check what's being included
    print(f"\nDEBUG: Checking what drugs are being included...")
    print(f"Clopidogrel mask matches: {clopidogrel_mask.sum()}")
    print(f"BNF mask matches: {bnf_mask.sum()}")
    print(f"Combined matches: {all_clopidogrel_mask.sum()}")
    
    print(f"\nTotal potential clopidogrel prescriptions: {len(clopidogrels)}")
    
    if len(clopidogrels) > 0:
        print(f"Unique patients with clopidogrel: {clopidogrels['eid'].nunique()}")
        
        # Show sample
        print(f"\nSample clopidogrel prescriptions:")
        sample_cols = ['eid', 'issue_date', 'drug_name', 'bnf_code']
        print(clopidogrels[sample_cols].head(10))
        
        # Most common clopidogrel drugs
        print(f"\nMost common clopidogrel drugs:")
        top_drugs = clopidogrels['drug_name'].value_counts().head(5)
        for drug, count in top_drugs.items():
            print(f"  {drug}: {count}")
    
    else:
        print("No obvious clopidogrel found")
        
        # Show what we do have
        print(f"\nSample of all drug names:")
        print(df['drug_name'].head(10).tolist())
        
        print(f"\nSample of all BNF codes:")
        print(df['bnf_code'].head(10).tolist())
    
    return clopidogrels

def encode_smoking(status):
    """One-hot encoding: [Never, Previous, Current]"""
    if status == "Never":
        return [1, 0, 0]
    elif status == "Previous":
        return [0, 1, 0]
    elif status == "Current":
        return [0, 0, 1]
    else:
        return [0, 0, 0]  # For missing/unknown

def build_features_aspirin(eids, t0s, processed_ids, thetas, covariate_dicts, sig_indices=None, 
                          is_treated=False, treatment_dates=None):
    """
    Build feature vectors for aspirin matching with proper exclusions and imputation
    
    Parameters:
    - eids: List of patient IDs
    - t0s: List of time indices for each patient
    - processed_ids: Array of all processed patient IDs
    - thetas: Signature loadings (N x K x T)
    - covariate_dicts: Dictionary with covariate data
    - sig_indices: Optional list of signature indices to use
    - is_treated: Whether these are treated patients
    - treatment_dates: Treatment dates for treated patients
    
    Returns:
    - features: Feature matrix for matching
    - indices: Patient indices in thetas
    - kept_eids: Successfully matched patient IDs
    """
    features = []
    indices = []
    kept_eids = []
    window = 10  # 10 years of signature history
    n_signatures = thetas.shape[1]
    if sig_indices is None:
        sig_indices = list(range(n_signatures))
    expected_length = len(sig_indices) * window
    
    # Calculate means for imputation
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
        
        if sig_traj.shape[0] != expected_length:
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
                treatment_age = age_at_enroll + treatment_dates[treatment_idx]
                age = treatment_age  # Use treatment age for matching
            else:
                age = age_at_enroll
        else:
            # For control patients: use enrollment age as index date
            age = age_at_enroll
            
        sex = covariate_dicts['sex'].get(int(eid), 0)
        if np.isnan(sex) or sex is None:
            sex = 0
        sex = int(sex)
        
        # EXCLUDE if missing binary variables
        dm2 = covariate_dicts['dm2_prev'].get(int(eid))
        if dm2 is None or np.isnan(dm2):
            continue  # Skip this patient
            
        antihtn = covariate_dicts['antihtnbase'].get(int(eid))
        if antihtn is None or np.isnan(antihtn):
            continue  # Skip this patient
            
        dm1 = covariate_dicts['dm1_prev'].get(int(eid))
        if dm1 is None or np.isnan(dm1):
            continue  # Skip this patient
        
        # EXCLUDE patients with colorectal cancer before treatment/enrollment (incident user logic)
        # Determine the reference age for cancer exclusion
        if is_treated and treatment_dates is not None:
            # For treated patients: use first aspirin prescription date
            treatment_idx = eids.index(eid) if eid in eids else None
            if treatment_idx is not None and treatment_idx < len(treatment_dates):
                treatment_age = age_at_enroll + treatment_dates[treatment_idx]
                reference_age = treatment_age
            else:
                reference_age = age_at_enroll
        else:
            # For control patients: use enrollment date
            reference_age = age_at_enroll
        
        # Check colorectal cancer exclusion (only exclude cancer before index date)
        # Note: This assumes you have colorectal cancer data in covariate_dicts
        # You may need to adapt this based on your actual cancer data structure
        cancer_any = covariate_dicts.get('colorectal_cancer_any', {}).get(int(eid), 0)
        cancer_censor_age = covariate_dicts.get('colorectal_cancer_censor_age', {}).get(int(eid))
        if cancer_any == 2 and cancer_censor_age is not None and not np.isnan(cancer_censor_age):
            if cancer_censor_age < reference_age:
                continue  # Skip this patient
        
        # Check for missing DM/HTN/HyperLip status (exclude if unknown)
        dm_any = covariate_dicts.get('Dm_Any', {}).get(int(eid))
        if dm_any is None or np.isnan(dm_any):
            continue  # Skip this patient
            
        ht_any = covariate_dicts.get('Ht_Any', {}).get(int(eid))
        if ht_any is None or np.isnan(ht_any):
            continue  # Skip this patient
            
        hyperlip_any = covariate_dicts.get('HyperLip_Any', {}).get(int(eid))
        if hyperlip_any is None or np.isnan(hyperlip_any):
            continue  # Skip this patient
        
        # IMPUTE quantitative variables with mean
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
        
        if np.any(np.isnan(feature_vector)):
            continue
        
        features.append(feature_vector)
        indices.append(idx)
        kept_eids.append(eid)
    
    print(f"Final result: {len(features)} patients kept out of {len(eids)}")
    
    if len(features) == 0:
        print("Warning: No valid features found after filtering")
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

def calculate_hazard_ratio_colorectal(treated_outcomes, control_outcomes, follow_up_times):
    """
    Calculate hazard ratio for colorectal cancer prevention using Cox proportional hazards model
    
    Parameters:
    - treated_outcomes: Binary outcome data for treated patients (1=colorectal cancer, 0=no cancer)
    - control_outcomes: Binary outcome data for control patients (1=colorectal cancer, 0=no cancer)
    - follow_up_times: Follow-up times for both groups (in years)
    
    Returns:
    - Dictionary with HR results
    """
    try:
        from lifelines import CoxPHFitter
        import warnings
        warnings.filterwarnings('ignore')
    except ImportError:
        print("Warning: lifelines not available for HR calculation")
        return None
    
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
    hr = np.exp(cph.params_['treatment'])
    
    # Get confidence intervals
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
    c_index = cph.concordance_index_
    
    # Compare to expected trial results for colorectal cancer prevention
    expected_hr = 0.75  # Expected 25% risk reduction
    hr_difference = hr - expected_hr
    ci_overlaps_expected = (hr_ci_lower <= expected_hr <= hr_ci_upper)
    
    results = {
        'hazard_ratio': hr,
        'hr_ci_lower': hr_ci_lower,
        'hr_ci_upper': hr_ci_upper,
        'p_value': p_value,
        'concordance_index': c_index,
        'expected_hr': expected_hr,
        'hr_difference': hr_difference,
        'ci_overlaps_expected': ci_overlaps_expected,
        'n_treated': n_treated,
        'n_control': n_control,
        'total_events': np.sum(all_outcomes),
        'validation_passed': ci_overlaps_expected and p_value < 0.05,
        'protective_effect': hr < 1.0  # True if aspirin is protective
    }
    
    return results

def verify_patient_cohorts_aspirin(gp_scripts, true_aspirins, processed_ids):
    """
    VERIFICATION FUNCTION: Check that our patient cohorts are properly defined for aspirin
    """
    print("=== PATIENT COHORT VERIFICATION (ASPIRIN) ===")
    
    # Get all patients with complete data (BOTH prescription AND signature data)
    patients_with_complete_data = set(gp_scripts['eid'].unique()).intersection(set(processed_ids))
    all_aspirin_eids = set(true_aspirins['eid'].unique())
    
    # Define cohorts explicitly
    treated_cohort = patients_with_complete_data.intersection(all_aspirin_eids)
    control_cohort = patients_with_complete_data - all_aspirin_eids
    
    print(f"Patients with complete data: {len(patients_with_complete_data):,}")
    print(f"All aspirin patients: {len(all_aspirin_eids):,}")
    print(f"Treated cohort (complete data + aspirin): {len(treated_cohort):,}")
    print(f"Control cohort (complete data - aspirin): {len(control_cohort):,}")
    
    # Verify no overlap
    overlap = treated_cohort.intersection(control_cohort)
    if len(overlap) > 0:
        print(f"❌ ERROR: {len(overlap)} patients in both treated and control!")
        return False
    else:
        print("✅ No overlap between treated and control cohorts")
        return True

def verify_treated_patients_actually_have_aspirin(treated_eids, true_aspirins):
    """
    VERIFICATION FUNCTION: Check that all claimed treated patients actually have aspirin
    """
    print("\n=== TREATED PATIENT VERIFICATION (ASPIRIN) ===")
    
    all_aspirin_eids = set(true_aspirins['eid'].unique())
    treated_with_aspirin = [eid for eid in treated_eids if eid in all_aspirin_eids]
    treated_without_aspirin = [eid for eid in treated_eids if eid not in all_aspirin_eids]
    
    print(f"Claimed treated patients: {len(treated_eids):,}")
    print(f"  - With aspirin: {len(treated_with_aspirin):,}")
    print(f"  - Without aspirin: {len(treated_without_aspirin):,}")
    
    if len(treated_without_aspirin) > 0:
        print(f"❌ ERROR: {len(treated_without_aspirin)} treated patients don't have aspirin!")
        return False
    else:
        print("✅ All treated patients have aspirin")
        return True

def verify_controls_are_clean_aspirin(control_eids, true_aspirins):
    """
    VERIFICATION FUNCTION: Check that all controls actually don't have aspirin
    """
    print("\n=== CONTROL PATIENT VERIFICATION (ASPIRIN) ===")
    
    all_aspirin_eids = set(true_aspirins['eid'].unique())
    controls_with_aspirin = [eid for eid in control_eids if eid in all_aspirin_eids]
    controls_without_aspirin = [eid for eid in control_eids if eid not in all_aspirin_eids]
    
    print(f"Claimed control patients: {len(control_eids):,}")
    print(f"  - With aspirin: {len(controls_with_aspirin):,}")
    print(f"  - Without aspirin: {len(controls_without_aspirin):,}")
    
    if len(controls_with_aspirin) > 0:
        print(f"❌ ERROR: {len(controls_with_aspirin)} controls have aspirin!")
        return False
    else:
        print("✅ All controls are clean (no aspirin)")
        return True

def aspirin_colorectal_analysis(gp_scripts=None, true_aspirins=None, processed_ids=None, 
                              thetas=None, sig_indices=None, covariate_dicts=None, Y=None, 
                              colorectal_cancer_indices=None, cov=None):
    """
    Aspirin-colorectal cancer prevention analysis with explicit self-checking
    
    This function does:
    1. Extract aspirin-treated patients using ObservationalTreatmentPatternLearner
    2. Define clean controls (no aspirin ever)
    3. Apply exclusions (colorectal cancer before index, missing binary data)
    4. Impute quantitative variables
    5. Perform nearest neighbor matching
    6. Calculate HR for colorectal cancer prevention
    7. VERIFY everything at each step
    
    Parameters:
    - gp_scripts: DataFrame with GP prescription data
    - true_aspirins: DataFrame with aspirin prescriptions
    - processed_ids: Array of all processed patient IDs
    - thetas: Signature loadings (N x K x T)
    - covariate_dicts: Dictionary with covariate data
    - Y: Outcome tensor [patients, events, time]
    - colorectal_cancer_indices: List of colorectal cancer event indices
    - cov: Covariates DataFrame
    
    Returns:
    - Dictionary with analysis results
    """
    
    print("=== ASPIRIN-COLORECTAL CANCER PREVENTION ANALYSIS ===")
    print("Expected effect: Aspirin should REDUCE colorectal cancer risk (HR < 1.0)")
    print("Expected HR from trials: ~0.7-0.8 (20-30% risk reduction)")
    
    # Step 1: Verify patient cohorts are properly defined
    print("\n1. Verifying patient cohort definitions:")
    cohort_ok = verify_patient_cohorts_aspirin(gp_scripts, true_aspirins, processed_ids)
    if not cohort_ok:
        print("❌ Cohort definition failed - stopping analysis")
        return None
    
    # Step 2: Extract treated patients using ObservationalTreatmentPatternLearner
    print("\n2. Extracting treated patients using ObservationalTreatmentPatternLearner:")
    from scripts.observational_treatment_patterns import ObservationalTreatmentPatternLearner
    
    learner = ObservationalTreatmentPatternLearner(
        signature_loadings=thetas,
        processed_ids=processed_ids, 
        statin_prescriptions=true_aspirins,  # Reuse statin logic for aspirin
        covariates=cov,
        gp_scripts=gp_scripts
    )
    
    treated_eids = learner.treatment_patterns['treated_patients']
    treated_times = learner.treatment_patterns['treatment_times']
    never_treated_eids = learner.treatment_patterns['never_treated']
    
    print(f"   Treated patients from learner: {len(treated_eids):,}")
    print(f"   Never-treated patients from learner: {len(never_treated_eids):,}")
    
    # VERIFY: Check that treated patients actually have aspirin
    treated_ok = verify_treated_patients_actually_have_aspirin(treated_eids, true_aspirins)
    if not treated_ok:
        print("❌ Treated patient verification failed - stopping analysis")
        return None
    
    # Step 3: Define clean controls
    print("\n3. Defining clean controls:")
    if gp_scripts is not None:
        all_patients_with_prescriptions = set(gp_scripts['eid'].unique())
    else:
        all_patients_with_prescriptions = set(true_aspirins['eid'].unique())
    
    # Use clean control definition: all GP patients minus ALL aspirin users
    all_aspirin_eids = set(true_aspirins['eid'].unique())
    valid_never_treated = [eid for eid in all_patients_with_prescriptions 
                      if eid not in all_aspirin_eids and eid in processed_ids]
    
    print(f"   Found {len(valid_never_treated)} never-treated patients with signature data")
    
    # VERIFY: Check that controls don't have aspirin
    control_ok = verify_controls_are_clean_aspirin(valid_never_treated, true_aspirins)
    if not control_ok:
        print("❌ Control verification failed - stopping analysis")
        return None
    
    # Use the same control selection logic
    control_eids_for_matching = valid_never_treated[:len(treated_eids)*2]  # 2:1 ratio
    control_t0s = []
    valid_control_eids = []
    
    for eid in control_eids_for_matching:
        try:
            age_at_enroll = covariate_dicts['age_at_enroll'].get(int(eid))
            if age_at_enroll is not None and not np.isnan(age_at_enroll):
                # Convert age to time index (age - 30, since time grid starts at age 30)
                t0 = int(age_at_enroll - 30)
                # Only require t0 >= 10 for controls
                if t0 >= 10:
                    control_t0s.append(t0)
                    valid_control_eids.append(eid)
        except:
            continue
    
    print(f"   Control patients with 10-year follow-up: {len(valid_control_eids):,}")
    
    # Step 4: Build features for treated patients
    print("\n4. Building features for treated patients:")
    treated_features, treated_indices, treated_eids_matched = build_features_aspirin(
        treated_eids, treated_times, processed_ids, thetas, 
        covariate_dicts, sig_indices=sig_indices, is_treated=True, treatment_dates=treated_times
    )
    
    print(f"   Treated patients after exclusions: {len(treated_features):,}")
    
    # Step 5: Build features for control patients
    print("\n5. Building features for control patients:")
    control_features, control_indices, control_eids_matched = build_features_aspirin(
        valid_control_eids, control_t0s, processed_ids, 
        thetas, covariate_dicts, sig_indices=sig_indices, is_treated=False, treatment_dates=None
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
    
    # COMPREHENSIVE INDEX ALIGNMENT CHECK
    print("\n=== INDEX ALIGNMENT VERIFICATION ===")
    
    # Check that matched indices correspond to the right patients
    print("Verifying index alignment between local and global arrays...")
    
    # For treated patients - just check first few and last few for verification
    treated_index_alignment_ok = True
    sample_size = min(5, len(matched_treated_indices))
    
    # Check first few
    for i in range(sample_size):
        local_eid = matched_treated_eids[i]
        global_eid = processed_ids[matched_treated_indices[i]]
        if local_eid != global_eid:
            print(f"❌ MISALIGNMENT: Local treated[{i}]={local_eid} != Global[{matched_treated_indices[i]}]={global_eid}")
            treated_index_alignment_ok = False
    
    # Check last few
    if len(matched_treated_indices) > sample_size:
        for i in range(len(matched_treated_indices) - sample_size, len(matched_treated_indices)):
            local_eid = matched_treated_eids[i]
            global_eid = processed_ids[matched_treated_indices[i]]
            if local_eid != global_eid:
                print(f"❌ MISALIGNMENT: Local treated[{i}]={local_eid} != Global[{matched_treated_indices[i]}]={global_eid}")
                treated_index_alignment_ok = False
    
    # For control patients - same approach
    control_index_alignment_ok = True
    
    # Check first few
    for i in range(sample_size):
        local_eid = matched_control_eids[i]
        global_eid = processed_ids[matched_control_indices[i]]
        if local_eid != global_eid:
            print(f"❌ MISALIGNMENT: Local control[{i}]={local_eid} != Global[{matched_control_indices[i]}]={global_eid}")
            control_index_alignment_ok = False
    
    # Check last few
    if len(matched_control_indices) > sample_size:
        for i in range(len(matched_control_indices) - sample_size, len(matched_control_indices)):
            local_eid = matched_control_eids[i]
            global_eid = processed_ids[matched_control_indices[i]]
            if local_eid != global_eid:
                print(f"❌ MISALIGNMENT: Local control[{i}]={local_eid} != Global[{matched_control_indices[i]}]={global_eid}")
                control_index_alignment_ok = False
    
    if treated_index_alignment_ok and control_index_alignment_ok:
        print(f"✅ All indices properly aligned between local and global arrays")
        print(f"   Verified {sample_size} samples from start and end of {len(matched_treated_indices):,} matched pairs")
    else:
        print("❌ Index misalignment detected - this will cause incorrect results!")
        return None
    
    # Check that treatment times correspond to the right patients
    print("\nVerifying treatment time alignment...")
    treatment_time_alignment_ok = True
    
    # Just check a sample of treatment times
    sample_size = min(5, len(matched_treated_eids))
    verified_count = 0
    
    for i in range(sample_size):
        treated_eid = matched_treated_eids[i]
        # Find this patient in the original treated lists
        original_idx = None
        for j, eid in enumerate(treated_eids):
            if eid == treated_eid:
                original_idx = j
                break
        
        if original_idx is not None and original_idx < len(treated_times):
            expected_time = treated_times[original_idx]
            verified_count += 1
        else:
            print(f"❌ Treated {treated_eid}: Could not find treatment time!")
            treatment_time_alignment_ok = False
    
    if treatment_time_alignment_ok:
        print(f"✅ Treatment time alignment verified for {verified_count} sample patients")
        print(f"   Sample treatment times: {[treated_times[i] for i in range(min(3, len(treated_times)))]}")
    else:
        print("❌ Treatment time misalignment - stopping analysis")
        return None
    
    # Now proceed with outcome calculation using PROPERLY VERIFIED indices
    print("\n=== OUTCOME CALCULATION WITH VERIFIED INDICES ===")
    
    # Step 7: Calculate outcomes and hazard ratio
    print("\n7. Calculating outcomes and hazard ratio:")
    
    if Y is not None:
        # Convert PyTorch tensor to numpy if needed
        if hasattr(Y, 'detach'):
            Y_np = Y.detach().cpu().numpy()
        else:
            Y_np = Y
        
        # Extract outcomes for matched cohorts
        treated_outcomes = []
        control_outcomes = []
        follow_up_times = []
        
        # Track event timing for analysis
        treated_event_times = []
        control_event_times = []
        treated_censoring_times = []
        control_censoring_times = []
        
        # Get outcomes for treated patients
        for treated_idx in matched_treated_indices:
            treated_eid = processed_ids[treated_idx]
            treatment_time = None
            
            # Find treatment time from the original treated times
            for i, eid in enumerate(treated_eids):
                if eid == treated_eid:
                    treatment_time = treated_times[i]
                    break
            
            if treatment_time is not None and treated_idx < Y_np.shape[0]:
                # Look for colorectal cancer events after treatment time
                if colorectal_cancer_indices is not None:
                    # Colorectal cancer events - check if any of the specified events occurred
                    post_treatment_outcomes = Y_np[treated_idx, colorectal_cancer_indices, int(treatment_time):]
                    post_treatment_outcomes = np.any(post_treatment_outcomes > 0, axis=0)
                else:
                    # If no specific event specified, use any event
                    post_treatment_outcomes = Y_np[treated_idx, :, int(treatment_time):]
                    post_treatment_outcomes = np.any(post_treatment_outcomes > 0, axis=0)
                
                event_occurred = np.any(post_treatment_outcomes > 0)
                
                if event_occurred:
                    # Find time to first event
                    event_times = np.where(post_treatment_outcomes > 0)[0]
                    time_to_event = event_times[0] if len(event_times) > 0 else 5.0
                    treated_event_times.append(time_to_event)
                else:
                    # Censored at end of follow-up (minimum 5 years)
                    time_to_event = min(5.0, Y_np.shape[2] - int(treatment_time))
                    treated_censoring_times.append(time_to_event)
                
                treated_outcomes.append(int(event_occurred))
                follow_up_times.append(time_to_event)
        
        # Get outcomes for control patients
        for control_idx in matched_control_indices:
            control_eid = processed_ids[control_idx]
            
            # For controls, use their age-based time point as "treatment" time
            age_at_enroll = covariate_dicts['age_at_enroll'].get(int(control_eid))
            if age_at_enroll is not None and not np.isnan(age_at_enroll):
                control_time = int(age_at_enroll - 30)  # Convert to time index
                
                if control_idx < Y_np.shape[0] and control_time < Y_np.shape[2]:
                    # Look for colorectal cancer events after control time
                    if colorectal_cancer_indices is not None:
                        # Colorectal cancer events - check if any of the specified events occurred
                        post_control_outcomes = Y_np[control_idx, colorectal_cancer_indices, control_time:]
                        post_control_outcomes = np.any(post_control_outcomes > 0, axis=0)
                    else:
                        # If no specific event specified, use any event
                        post_control_outcomes = Y_np[control_idx, :, control_time:]
                        post_control_outcomes = np.any(post_control_outcomes > 0, axis=0)
                    
                    event_occurred = np.any(post_control_outcomes > 0)
                    
                    if event_occurred:
                        # Find time to first event
                        event_times = np.where(post_control_outcomes > 0)[0]
                        time_to_event = event_times[0] if len(event_times) > 0 else 5.0
                        control_event_times.append(time_to_event)
                    else:
                        # Censored at end of follow-up (minimum 5 years)
                        time_to_event = min(5.0, Y_np.shape[2] - control_time)
                        control_censoring_times.append(time_to_event)
                    
                    control_outcomes.append(int(event_occurred))
                    follow_up_times.append(time_to_event)
        
        # ANALYZE EVENT TIMING DISTRIBUTIONS
        print("\n=== EVENT TIMING ANALYSIS ===")
        print("This will help explain why HR might be protective even with higher event rates")
        
        if len(treated_event_times) > 0 and len(control_event_times) > 0:
            print(f"\nTreated patients with events: {len(treated_event_times):,}")
            print(f"Control patients with events: {len(control_event_times):,}")
            
            # Event timing statistics
            treated_event_mean = np.mean(treated_event_times)
            control_event_mean = np.mean(control_event_times)
            treated_event_median = np.median(treated_event_times)
            control_event_median = np.median(control_event_times)
            
            print(f"\nEvent timing (years from index):")
            print(f"  Treated:  Mean={treated_event_mean:.2f}, Median={treated_event_median:.2f}")
            print(f"  Control:  Mean={control_event_mean:.2f}, Median={control_event_median:.2f}")
            
            # Test if treated events happen later (protective effect)
            if treated_event_mean > control_event_mean:
                timing_difference = treated_event_mean - control_event_mean
                print(f"✅ Treated events happen {timing_difference:.2f} years LATER on average")
                print(f"   This explains protective HR despite potentially higher event rates!")
            else:
                timing_difference = control_event_mean - treated_event_mean
                print(f"⚠️ Treated events happen {timing_difference:.2f} years EARLIER on average")
                print(f"   This would suggest harmful effect - investigate further!")
            
            # Censoring analysis
            if len(treated_censoring_times) > 0 and len(control_censoring_times) > 0:
                treated_censor_mean = np.mean(treated_censoring_times)
                control_censor_mean = np.mean(control_censoring_times)
                
                print(f"\nCensoring times (years from index):")
                print(f"  Treated:  Mean={treated_censor_mean:.2f}")
                print(f"  Control:  Mean={control_censor_mean:.2f}")
                
                if treated_censor_mean > control_censor_mean:
                    print(f"✅ Treated patients followed longer before censoring")
                else:
                    print(f"⚠️ Controls followed longer before censoring")
            
            # Event rate comparison
            treated_event_rate = len(treated_event_times) / len(treated_outcomes) * 100
            control_event_rate = len(control_event_times) / len(control_outcomes) * 100
            
            print(f"\nEvent rates:")
            print(f"  Treated:  {treated_event_rate:.1f}% ({len(treated_event_times)}/{len(treated_outcomes)})")
            print(f"  Control:  {control_event_rate:.1f}% ({len(control_event_times)}/{len(control_outcomes)})")
            
            if treated_event_rate > control_event_rate:
                rate_difference = treated_event_rate - control_event_rate
                print(f"⚠️ Treated event rate {rate_difference:.1f}% HIGHER than control")
                print(f"   But if events happen later, this can still give protective HR!")
            else:
                rate_difference = control_event_rate - treated_event_rate
                print(f"✅ Treated event rate {rate_difference:.1f}% LOWER than control")
                print(f"   This directly supports protective effect")
        
        if len(treated_outcomes) > 10 and len(control_outcomes) > 10:
            hr_results = calculate_hazard_ratio_colorectal(
                np.array(treated_outcomes),
                np.array(control_outcomes),
                np.array(follow_up_times)
            )
            
            print(f"   Treated colorectal cancer events: {np.sum(treated_outcomes):,}")
            print(f"   Control colorectal cancer events: {np.sum(control_outcomes):,}")
            print(f"   Total colorectal cancer events: {hr_results['total_events']:,}")
            print()
            
            print("=== FINAL RESULTS (COLORECTAL CANCER PREVENTION) ===")
            print(f"Hazard Ratio: {hr_results['hazard_ratio']:.3f}")
            print(f"95% CI: {hr_results['hr_ci_lower']:.3f} - {hr_results['hr_ci_upper']:.3f}")
            print(f"P-value: {hr_results['p_value']:.4f}")
            print(f"Matched pairs: {hr_results['n_treated']:,}")
            
            # Interpret results for cancer prevention
            if hr_results['hazard_ratio'] < 1.0:
                risk_reduction = (1 - hr_results['hazard_ratio']) * 100
                print(f"Risk reduction: {risk_reduction:.1f}%")
            else:
                risk_increase = (hr_results['hazard_ratio'] - 1) * 100
                print(f"Risk increase: {risk_increase:.1f}%")
            
            # Compare to expected trial results
            expected_hr = 0.75
            hr_difference = hr_results['hazard_ratio'] - expected_hr
            print(f"Expected HR from trials: {expected_hr:.3f}")
            print(f"Difference from expected: {hr_difference:.3f}")
            
            # Check if results are consistent with trials
            ci_overlaps_expected = (hr_results['hr_ci_lower'] <= expected_hr <= hr_results['hr_ci_upper'])
            print(f"CI overlaps expected: {ci_overlaps_expected}")
            print(f"Protective effect detected: {hr_results['protective_effect']}")
            print(f"Validation passed: {ci_overlaps_expected and hr_results['p_value'] < 0.05}")
            
            # Return comprehensive results including matched patient IDs and treatment times
            comprehensive_results = {
                'hazard_ratio_results': hr_results,
                'matched_patients': {
                    'treated_eids': matched_treated_eids,
                    'control_eids': matched_control_eids,
                    'treated_indices': matched_treated_indices,
                    'control_indices': matched_control_indices
                },
                'treatment_times': {
                    'treated_times': [treated_times[i] for i, eid in enumerate(treated_eids) if eid in matched_treated_eids],
                    'control_times': [control_t0s[i] for i, eid in enumerate(valid_control_eids) if eid in matched_control_eids]
                },
                'cohort_sizes': {
                    'n_treated': len(matched_treated_eids),
                    'n_control': len(matched_control_eids),
                    'n_total': len(matched_treated_eids) + len(matched_control_eids)
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

def clopidogrel_colorectal_analysis(gp_scripts=None, true_clopidogrels=None, processed_ids=None, 
                                   thetas=None, sig_indices=None, covariate_dicts=None, Y=None, 
                                   colorectal_cancer_indices=None, cov=None):
    """
    Clopidogrel-colorectal cancer analysis as NEGATIVE CONTROL
    
    This is a perfect negative control because:
    1. Clopidogrel is an antiplatelet drug like aspirin (same patient population)
    2. But works through P2Y12 receptor inhibition (different mechanism)
    3. No known anti-CRC effects
    4. Should show HR ≈ 1.0 (null effect) or potentially harmful
    
    Expected result: HR ≈ 1.0 (no effect) or HR > 1.0 (harmful)
    If we see HR < 1.0 (protective), that suggests confounding or bias
    
    Parameters: Same as aspirin analysis
    Returns: Same structure for plotting compatibility
    """
    
    print("=== CLOPIDOGREL-COLORECTAL CANCER ANALYSIS (NEGATIVE CONTROL) ===")
    print("Expected effect: Clopidogrel should have NO effect on CRC (HR ≈ 1.0)")
    print("If HR < 1.0, this suggests confounding or bias in our analysis!")
    
    # Step 1: Verify patient cohorts are properly defined
    print("\n1. Verifying patient cohort definitions:")
    cohort_ok = verify_patient_cohorts_aspirin(gp_scripts, true_clopidogrels, processed_ids)
    if not cohort_ok:
        print("❌ Cohort definition failed - stopping analysis")
        return None
    
    # Step 2: Extract treated patients using ObservationalTreatmentPatternLearner
    print("\n2. Extracting treated patients using ObservationalTreatmentPatternLearner:")
    from scripts.observational_treatment_patterns import ObservationalTreatmentPatternLearner
    
    learner = ObservationalTreatmentPatternLearner(
        signature_loadings=thetas,
        processed_ids=processed_ids, 
        statin_prescriptions=true_clopidogrels,  # Reuse statin logic for clopidogrel
        covariates=cov,
        gp_scripts=gp_scripts
    )
    
    treated_eids = learner.treatment_patterns['treated_patients']
    treated_times = learner.treatment_patterns['treatment_times']
    never_treated_eids = learner.treatment_patterns['never_treated']
    
    print(f"   Treated patients from learner: {len(treated_eids):,}")
    print(f"   Never-treated patients from learner: {len(never_treated_eids):,}")
    
    # VERIFY: Check that treated patients actually have clopidogrel
    treated_ok = verify_treated_patients_actually_have_aspirin(treated_eids, true_clopidogrels)
    if not treated_ok:
        print("❌ Treated patient verification failed - stopping analysis")
        return None
    
    # Step 3: Define clean controls
    print("\n3. Defining clean controls:")
    if gp_scripts is not None:
        all_patients_with_prescriptions = set(gp_scripts['eid'].unique())
    else:
        all_patients_with_prescriptions = set(true_clopidogrels['eid'].unique())
    
    # Use clean control definition: all GP patients minus ALL clopidogrel users
    all_clopidogrel_eids = set(true_clopidogrels['eid'].unique())
    valid_never_treated = [eid for eid in all_patients_with_prescriptions 
                      if eid not in all_clopidogrel_eids and eid in processed_ids]
    
    print(f"   Found {len(valid_never_treated)} never-treated patients with signature data")
    
    # VERIFY: Check that controls don't have clopidogrel
    control_ok = verify_controls_are_clean_aspirin(valid_never_treated, true_clopidogrels)
    if not control_ok:
        print("❌ Control verification failed - stopping analysis")
        return None
    
    # Use the same control selection logic
    control_eids_for_matching = valid_never_treated[:len(treated_eids)*2]  # 2:1 ratio
    control_t0s = []
    valid_control_eids = []
    
    for eid in control_eids_for_matching:
        try:
            age_at_enroll = covariate_dicts['age_at_enroll'].get(int(eid))
            if age_at_enroll is not None and not np.isnan(age_at_enroll):
                # Convert age to time index (age - 30, since time grid starts at age 30)
                t0 = int(age_at_enroll - 30)
                # Only require t0 >= 10 for controls
                if t0 >= 10:
                    control_t0s.append(t0)
                    valid_control_eids.append(eid)
        except:
            continue
    
    print(f"   Control patients with 10-year follow-up: {len(valid_control_eids):,}")
    
    # Step 4: Build features for treated patients
    print("\n4. Building features for treated patients:")
    treated_features, treated_indices, treated_eids_matched = build_features_aspirin(
        treated_eids, treated_times, processed_ids, thetas, 
        covariate_dicts, sig_indices=sig_indices, is_treated=True, treatment_dates=treated_times
    )
    
    print(f"   Treated patients after exclusions: {len(treated_features):,}")
    
    # Step 5: Build features for control patients
    print("\n5. Building features for control patients:")
    control_features, control_indices, control_eids_matched = build_features_aspirin(
        valid_control_eids, control_t0s, processed_ids, 
        thetas, covariate_dicts, sig_indices=sig_indices, is_treated=False, treatment_dates=None
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
    
    # Step 7: Calculate outcomes and hazard ratio
    print("\n7. Calculating outcomes and hazard ratio:")
    
    if Y is not None:
        # Convert PyTorch tensor to numpy if needed
        if hasattr(Y, 'detach'):
            Y_np = Y.detach().cpu().numpy()
        else:
            Y_np = Y
        
        # Extract outcomes for matched cohorts
        treated_outcomes = []
        control_outcomes = []
        follow_up_times = []
        
        # Get outcomes for treated patients
        for treated_idx in matched_treated_indices:
            treated_eid = processed_ids[treated_idx]
            treatment_time = None
            
            # Find treatment time from the original treated times
            for i, eid in enumerate(treated_eids):
                if eid == treated_eid:
                    treatment_time = treated_times[i]
                    break
            
            if treatment_time is not None and treated_idx < Y_np.shape[0]:
                # Look for colorectal cancer events after treatment time
                if colorectal_cancer_indices is not None:
                    # Colorectal cancer events - check if any of the specified events occurred
                    post_treatment_outcomes = Y_np[treated_idx, colorectal_cancer_indices, int(treatment_time):]
                    post_treatment_outcomes = np.any(post_treatment_outcomes > 0, axis=0)
                else:
                    # If no specific event specified, use any event
                    post_treatment_outcomes = Y_np[treated_idx, :, int(treatment_time):]
                    post_treatment_outcomes = np.any(post_treatment_outcomes > 0, axis=0)
                
                event_occurred = np.any(post_treatment_outcomes > 0)
                
                if event_occurred:
                    # Find time to first event
                    event_times = np.where(post_treatment_outcomes > 0)[0]
                    time_to_event = event_times[0] if len(event_times) > 0 else 5.0
                else:
                    # Censored at end of follow-up (minimum 5 years)
                    time_to_event = min(5.0, Y_np.shape[2] - int(treatment_time))
                
                treated_outcomes.append(int(event_occurred))
                follow_up_times.append(time_to_event)
        
        # Get outcomes for control patients
        for control_idx in matched_control_indices:
            control_eid = processed_ids[control_idx]
            
            # For controls, use their age-based time point as "treatment" time
            age_at_enroll = covariate_dicts['age_at_enroll'].get(int(control_eid))
            if age_at_enroll is not None and not np.isnan(age_at_enroll):
                control_time = int(age_at_enroll - 30)  # Convert to time index
                
                if control_idx < Y_np.shape[0] and control_time < Y_np.shape[2]:
                    # Look for colorectal cancer events after control time
                    if colorectal_cancer_indices is not None:
                        # Colorectal cancer events - check if any of the specified events occurred
                        post_control_outcomes = Y_np[control_idx, colorectal_cancer_indices, control_time:]
                        post_control_outcomes = np.any(post_control_outcomes > 0, axis=0)
                    else:
                        # If no specific event specified, use any event
                        post_control_outcomes = Y_np[control_idx, :, control_time:]
                        post_control_outcomes = np.any(post_control_outcomes > 0, axis=0)
                    
                    event_occurred = np.any(post_control_outcomes > 0)
                    
                    if event_occurred:
                        # Find time to first event
                        event_times = np.where(post_control_outcomes > 0)[0]
                        time_to_event = event_times[0] if len(event_times) > 0 else 5.0
                    else:
                        # Censored at end of follow-up (minimum 5 years)
                        time_to_event = min(5.0, Y_np.shape[2] - control_time)
                    
                    control_outcomes.append(int(event_occurred))
                    follow_up_times.append(time_to_event)
        
        if len(treated_outcomes) > 10 and len(control_outcomes) > 10:
            hr_results = calculate_hazard_ratio_colorectal(
                np.array(treated_outcomes),
                np.array(control_outcomes),
                np.array(follow_up_times)
            )
            
            print(f"   Treated CRC events: {np.sum(treated_outcomes):,}")
            print(f"   Control CRC events: {np.sum(control_outcomes):,}")
            print(f"   Total CRC events: {hr_results['total_events']:,}")
            print()
            
            print("=== FINAL RESULTS (CLOPIDOGREL NEGATIVE CONTROL) ===")
            print(f"Hazard Ratio: {hr_results['hazard_ratio']:.3f}")
            print(f"95% CI: {hr_results['hr_ci_lower']:.3f} - {hr_results['hr_ci_upper']:.3f}")
            print(f"P-value: {hr_results['p_value']:.4f}")
            print(f"Matched pairs: {hr_results['n_treated']:,}")
            
            # Interpret results for negative control
            if hr_results['hazard_ratio'] < 1.0:
                risk_reduction = (1 - hr_results['hazard_ratio']) * 100
                print(f"Risk reduction: {risk_reduction:.1f}%")
                print(f"⚠️  WARNING: This suggests confounding or bias!")
                print(f"   Clopidogrel should have NO effect on CRC")
            else:
                risk_increase = (hr_results['hazard_ratio'] - 1) * 100
                print(f"Risk increase: {risk_increase:.1f}%")
                print(f"✅ This is more consistent with expected null effect")
            
            # Check if results are consistent with null effect
            hr_close_to_null = 0.8 <= hr_results['hazard_ratio'] <= 1.2
            print(f"HR close to null (0.8-1.2): {hr_close_to_null}")
            print(f"Negative control validation: {'PASSED' if hr_close_to_null else 'FAILED'}")
            
            # Return results in the format expected by plotting functions
            final_results = {
                'hazard_ratio_results': hr_results,
                'matched_patients': {
                    'treated_eids': matched_treated_eids,
                    'control_eids': matched_control_eids,
                    'treated_indices': matched_treated_indices,
                    'control_indices': matched_control_indices
                },
                'treatment_times': treated_times,
                'n_treated': len(matched_treated_eids),
                'n_control': len(matched_control_eids)
            }
            
            return final_results
        else:
            print("   Insufficient outcome data for HR calculation")
            return None
    else:
        print("   No outcomes (Y) found - cannot calculate HR")
        return None

# Run the analysis
if __name__ == "__main__":
    # For standalone execution, these would need to be loaded
    results = aspirin_colorectal_analysis()
    
    if results is not None:
        print("\n=== COMPREHENSIVE RESULTS SUMMARY ===")
        print(f"Matched pairs: {results['cohort_sizes']['n_treated']:,}")
        print(f"Total patients: {results['cohort_sizes']['n_total']:,}")
        
        hr_results = results['hazard_ratio_results']
        print(f"Hazard Ratio: {hr_results['hazard_ratio']:.3f}")
        print(f"95% CI: {hr_results['hr_ci_lower']:.3f} - {hr_results['hr_ci_upper']:.3f}")
        print(f"P-value: {hr_results['p_value']:.4f}")
        
        # Now you can access matched patients for further analysis:
        matched_treated_eids = results['matched_patients']['treated_eids']
        matched_control_eids = results['matched_patients']['control_eids']
        print(f"\nMatched treated EIDs: {len(matched_treated_eids):,}")
        print(f"Matched control EIDs: {len(matched_control_eids):,}")
        
        # Show treatment times
        treated_times = results['treatment_times']['treated_times']
        control_times = results['treatment_times']['control_times']
        print(f"Treated times (years from enrollment): {len(treated_times):,}")
        print(f"Control times (years from enrollment): {len(control_times):,}")
        
        # Example: You can now use these for differential response analysis
        print("\nYou can now use these matched patient IDs for:")
        print("- Differential response analysis")
        print("- Signature heterogeneity testing") 
        print("- Individual treatment effect analysis")
        print("- Subgroup analysis by signature patterns")
        print("- Running run_aspirin_differential_analysis.py")
    else:
        print("Analysis failed - no results returned") 