#!/usr/bin/env python3
"""
Differential Drug Response Analysis Based on Genetic Signatures

This script analyzes how genetic signatures (PRS) and thetas influence 
drug response, detecting enrichment for differential response patterns.

Author: Sarah Urbut
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DifferentialResponseAnalyzer:
    """
    Analyzes differential drug response based on genetic signatures and thetas.
    Detects enrichment for differential response patterns.
    """
    
    def __init__(self, thetas, processed_ids, prs_matrix, prs_labels, 
                 disease_names, covariate_dicts):
        """
        Initialize the analyzer.
        
        Args:
            thetas: Signature loadings (N, n_signatures, n_timepoints)
            processed_ids: Patient IDs corresponding to thetas
            prs_matrix: PRS matrix (N, n_prs)
            prs_labels: List of PRS names
            disease_names: List of disease names
            covariate_dicts: Dictionary of covariate mappings
        """
        self.thetas = thetas
        self.processed_ids = processed_ids
        self.prs_matrix = prs_matrix
        self.prs_labels = prs_labels
        self.disease_names = disease_names
        self.covariate_dicts = covariate_dicts
        
        # Create patient ID to index mapping
        self.eid_to_idx = {eid: idx for idx, eid in enumerate(processed_ids)}
        
        print(f"Initialized Differential Response Analyzer")
        print(f"  - {len(processed_ids)} patients")
        print(f"  - {thetas.shape[1]} signatures")
        print(f"  - {thetas.shape[2]} timepoints")
        print(f"  - {len(prs_labels)} PRS traits")
        print(f"  - {len(disease_names)} disease outcomes")
    
    def analyze_drug_response_by_genetics(self, drug_prescriptions, outcome_indices, 
                                        drug_name="Drug", outcome_name="Outcome"):
        """
        Analyze how genetic signatures influence drug response.
        
        Args:
            drug_prescriptions: DataFrame with drug prescription data
            outcome_indices: List of outcome indices to analyze
            drug_name: Name of the drug for labeling
            outcome_name: Name of the outcome for labeling
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n=== {drug_name.upper()} RESPONSE BY GENETICS ANALYSIS ===")
        print(f"Outcome: {outcome_name}")
        
        # Get treated and control patients
        treated_eids = set(drug_prescriptions['eid'].unique())
        all_eids = set(self.processed_ids)
        control_eids = all_eids - treated_eids
        
        print(f"Treated patients: {len(treated_eids):,}")
        print(f"Control patients: {len(control_eids):,}")
        
        # Filter to patients with complete data
        treated_with_data = self._filter_complete_patients(treated_eids)
        control_with_data = self._filter_complete_patients(control_eids)
        
        print(f"Treated with complete data: {len(treated_with_data):,}")
        print(f"Controls with complete data: {len(control_with_data):,}")
        
        # Analyze each outcome
        results = {}
        for outcome_idx in outcome_indices:
            outcome_name = self.disease_names[outcome_idx]
            print(f"\n--- Analyzing {outcome_name} (Index {outcome_idx}) ---")
            
            outcome_results = self._analyze_single_outcome(
                treated_with_data, control_with_data, outcome_idx, outcome_name
            )
            results[outcome_idx] = outcome_results
        
        return results
    
    def _filter_complete_patients(self, eid_list):
        """Filter patients to those with complete data."""
        complete_patients = []
        
        for eid in eid_list:
            if eid not in self.eid_to_idx:
                continue
                
            # Check if we have PRS data
            idx = self.eid_to_idx[eid]
            if idx >= self.prs_matrix.shape[0]:
                continue
                
            # Check if we have theta data
            if idx >= self.thetas.shape[0]:
                continue
                
            # Check if we have key covariates
            age = self.covariate_dicts.get('age_at_enroll', {}).get(int(eid))
            if age is None or np.isnan(age):
                continue
                
            complete_patients.append(eid)
        
        return complete_patients
    
    def _analyze_single_outcome(self, treated_eids, control_eids, outcome_idx, outcome_name):
        """Analyze a single outcome for differential response."""
        
        # Get patient indices
        treated_indices = [self.eid_to_idx[eid] for eid in treated_eids]
        control_indices = [self.eid_to_idx[eid] for eid in control_eids]
        
        # Extract thetas for these patients
        treated_thetas = self.thetas[treated_indices, :, :]
        control_thetas = self.thetas[control_indices, :, :]
        
        # Extract PRS for these patients
        treated_prs = self.prs_matrix[treated_indices, :]
        control_prs = self.prs_matrix[control_indices, :]
        
        # Calculate baseline differences
        baseline_diff = self._calculate_baseline_differences(
            treated_thetas, control_thetas, treated_prs, control_prs
        )
        
        # Analyze genetic enrichment
        genetic_enrichment = self._analyze_genetic_enrichment(
            treated_thetas, control_thetas, treated_prs, control_prs, outcome_idx
        )
        
        # Analyze signature-response interactions
        signature_interactions = self._analyze_signature_interactions(
            treated_thetas, control_thetas, treated_prs, control_prs, outcome_idx
        )
        
        # Predict differential response
        response_prediction = self._predict_differential_response(
            treated_thetas, control_thetas, treated_prs, control_prs, outcome_idx
        )
        
        return {
            'baseline_differences': baseline_diff,
            'genetic_enrichment': genetic_enrichment,
            'signature_interactions': signature_interactions,
            'response_prediction': response_prediction,
            'n_treated': len(treated_eids),
            'n_control': len(control_eids)
        }
    
    def _calculate_baseline_differences(self, treated_thetas, control_thetas, 
                                      treated_prs, control_prs):
        """Calculate baseline differences between treated and control groups."""
        
        # Calculate mean thetas across time
        treated_mean_thetas = np.mean(treated_thetas, axis=2)  # (n_patients, n_signatures)
        control_mean_thetas = np.mean(control_thetas, axis=2)
        
        # Calculate differences
        theta_diff = np.mean(treated_mean_thetas, axis=0) - np.mean(control_mean_thetas, axis=0)
        
        # Calculate PRS differences
        prs_diff = np.mean(treated_prs, axis=0) - np.mean(control_prs, axis=0)
        
        # Statistical significance
        theta_pvals = []
        for sig_idx in range(treated_mean_thetas.shape[1]):
            _, pval = stats.ttest_ind(
                treated_mean_thetas[:, sig_idx], 
                control_mean_thetas[:, sig_idx]
            )
            theta_pvals.append(pval)
        
        prs_pvals = []
        for prs_idx in range(treated_prs.shape[1]):
            _, pval = stats.ttest_ind(
                treated_prs[:, prs_idx], 
                control_prs[:, prs_idx]
            )
            prs_pvals.append(pval)
        
        return {
            'theta_differences': theta_diff,
            'theta_pvalues': np.array(theta_pvals),
            'prs_differences': prs_diff,
            'prs_pvalues': np.array(prs_pvals)
        }
    
    def _analyze_genetic_enrichment(self, treated_thetas, control_thetas, 
                                   treated_prs, control_prs, outcome_idx):
        """Analyze genetic enrichment for differential response."""
        
        # Calculate response metrics for treated patients
        treated_response = self._calculate_response_metrics(treated_thetas, outcome_idx)
        
        # Calculate response metrics for controls
        control_response = self._calculate_response_metrics(control_thetas, outcome_idx)
        
        # Analyze PRS-response relationships in treated group
        prs_response_correlations = []
        prs_response_pvals = []
        
        for prs_idx in range(treated_prs.shape[1]):
            correlation, pval = stats.pearsonr(
                treated_prs[:, prs_idx], 
                treated_response
            )
            prs_response_correlations.append(correlation)
            prs_response_pvals.append(pval)
        
        # Analyze signature-response relationships in treated group
        sig_response_correlations = []
        sig_response_pvals = []
        
        treated_mean_thetas = np.mean(treated_thetas, axis=2)
        for sig_idx in range(treated_mean_thetas.shape[1]):
            correlation, pval = stats.pearsonr(
                treated_mean_thetas[:, sig_idx], 
                treated_response
            )
            sig_response_correlations.append(correlation)
            sig_response_pvals.append(pval)
        
        # Calculate enrichment scores
        enrichment_scores = self._calculate_enrichment_scores(
            treated_prs, treated_response, control_prs, control_response
        )
        
        return {
            'prs_response_correlations': np.array(prs_response_correlations),
            'prs_response_pvalues': np.array(prs_response_pvals),
            'signature_response_correlations': np.array(sig_response_correlations),
            'signature_response_pvalues': np.array(sig_response_pvals),
            'enrichment_scores': enrichment_scores,
            'treated_response': treated_response,
            'control_response': control_response
        }
    
    def _calculate_response_metrics(self, thetas, outcome_idx):
        """Calculate response metrics from thetas."""
        # For now, use the mean theta value as a simple response metric
        # This could be enhanced with more sophisticated response calculations
        return np.mean(thetas, axis=(1, 2))
    
    def _calculate_enrichment_scores(self, treated_prs, treated_response, 
                                   control_prs, control_response):
        """Calculate enrichment scores for genetic variants."""
        
        # Calculate response variance in treated vs control
        treated_response_var = np.var(treated_response)
        control_response_var = np.var(control_response)
        
        # Calculate PRS-response variance in treated vs control
        enrichment_scores = []
        
        for prs_idx in range(treated_prs.shape[1]):
            # Calculate PRS-response relationship strength
            treated_corr, _ = stats.pearsonr(treated_prs[:, prs_idx], treated_response)
            control_corr, _ = stats.pearsonr(control_prs[:, prs_idx], control_response)
            
            # Enrichment = stronger relationship in treated group
            enrichment = abs(treated_corr) - abs(control_corr)
            enrichment_scores.append(enrichment)
        
        return np.array(enrichment_scores)
    
    def _analyze_signature_interactions(self, treated_thetas, control_thetas, 
                                      treated_prs, control_prs, outcome_idx):
        """Analyze interactions between signatures and genetic variants."""
        
        # Calculate signature-PRS interactions in treated group
        interaction_strengths = []
        interaction_pvals = []
        
        treated_mean_thetas = np.mean(treated_thetas, axis=2)
        
        for sig_idx in range(treated_mean_thetas.shape[1]):
            for prs_idx in range(treated_prs.shape[1]):
                # Calculate interaction term
                interaction_term = treated_mean_thetas[:, sig_idx] * treated_prs[:, prs_idx]
                
                # Test if interaction predicts response
                response = self._calculate_response_metrics(treated_thetas, outcome_idx)
                correlation, pval = stats.pearsonr(interaction_term, response)
                
                interaction_strengths.append(correlation)
                interaction_pvals.append(pval)
        
        # Reshape to signature x PRS matrix
        n_signatures = treated_mean_thetas.shape[1]
        n_prs = treated_prs.shape[1]
        
        interaction_matrix = np.array(interaction_strengths).reshape(n_signatures, n_prs)
        pvalue_matrix = np.array(interaction_pvals).reshape(n_signatures, n_prs)
        
        return {
            'interaction_strengths': interaction_matrix,
            'interaction_pvalues': pvalue_matrix
        }
    
    def _predict_differential_response(self, treated_thetas, control_thetas, 
                                    treated_prs, control_prs, outcome_idx):
        """Predict differential response using machine learning."""
        
        # Prepare features: combine thetas and PRS
        treated_features = np.concatenate([
            np.mean(treated_thetas, axis=2),  # Mean thetas
            treated_prs,  # PRS values
        ], axis=1)
        
        control_features = np.concatenate([
            np.mean(control_thetas, axis=2),  # Mean thetas
            control_prs,  # PRS values
        ], axis=1)
        
        # Prepare labels: 1 for treated, 0 for control
        treated_labels = np.ones(treated_features.shape[0])
        control_labels = np.zeros(control_features.shape[0])
        
        # Combine data
        X = np.vstack([treated_features, control_features])
        y = np.concatenate([treated_labels, control_labels])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest to predict treatment group
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
        
        # Train on full data
        rf.fit(X_scaled, y)
        
        # Feature importance
        feature_importance = rf.feature_importances_
        
        # Split importance between thetas and PRS
        n_signatures = treated_thetas.shape[1]
        n_prs = treated_prs.shape[1]
        
        theta_importance = feature_importance[:n_signatures]
        prs_importance = feature_importance[n_signatures:]
        
        return {
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'theta_importance': theta_importance,
            'prs_importance': prs_importance,
            'feature_importance': feature_importance
        }
    
    def create_visualizations(self, results, drug_name, outcome_name):
        """Create comprehensive visualizations of the analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Differential Response Analysis: {drug_name} vs {outcome_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Baseline differences in signatures
        outcome_results = list(results.values())[0]  # Take first outcome
        baseline = outcome_results['baseline_differences']
        
        # Signature differences
        sig_diffs = baseline['theta_differences']
        sig_pvals = baseline['theta_pvalues']
        significant_sigs = sig_pvals < 0.05
        
        axes[0, 0].bar(range(len(sig_diffs)), sig_diffs, 
                       color=['red' if sig else 'gray' for sig in significant_sigs])
        axes[0, 0].set_title('Signature Differences (Treated - Control)')
        axes[0, 0].set_xlabel('Signature Index')
        axes[0, 0].set_ylabel('Difference')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PRS differences
        prs_diffs = baseline['prs_differences']
        prs_pvals = baseline['prs_pvalues']
        significant_prs = prs_pvals < 0.05
        
        axes[0, 1].bar(range(len(prs_diffs)), prs_diffs,
                       color=['red' if prs else 'gray' for prs in significant_prs])
        axes[0, 1].set_title('PRS Differences (Treated - Control)')
        axes[0, 1].set_xlabel('PRS Index')
        axes[0, 1].set_ylabel('Difference')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Genetic enrichment
        enrichment = outcome_results['genetic_enrichment']
        enrichment_scores = enrichment['enrichment_scores']
        
        axes[0, 2].bar(range(len(enrichment_scores)), enrichment_scores, color='skyblue')
        axes[0, 2].set_title('Genetic Enrichment Scores')
        axes[0, 2].set_xlabel('PRS Index')
        axes[0, 2].set_ylabel('Enrichment Score')
        axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Signature-PRS interactions
        interactions = outcome_results['signature_interactions']
        interaction_matrix = interactions['interaction_strengths']
        
        im = axes[1, 0].imshow(interaction_matrix, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('Signature-PRS Interactions')
        axes[1, 0].set_xlabel('PRS Index')
        axes[1, 0].set_ylabel('Signature Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Feature importance for prediction
        prediction = outcome_results['response_prediction']
        theta_importance = prediction['theta_importance']
        prs_importance = prediction['prs_importance']
        
        # Top signatures
        top_sig_indices = np.argsort(theta_importance)[-10:]
        axes[1, 1].barh(range(len(top_sig_indices)), 
                        theta_importance[top_sig_indices], color='lightcoral')
        axes[1, 1].set_title('Top Signature Importance')
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_ylabel('Signature Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Top PRS
        top_prs_indices = np.argsort(prs_importance)[-10:]
        axes[1, 2].barh(range(len(top_prs_indices)), 
                        prs_importance[top_prs_indices], color='lightgreen')
        axes[1, 2].set_title('Top PRS Importance')
        axes[1, 2].set_xlabel('Importance')
        axes[1, 2].set_ylabel('PRS Index')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        self._print_summary_statistics(results, drug_name, outcome_name)
    
    def _print_summary_statistics(self, results, drug_name, outcome_name):
        """Print summary statistics of the analysis."""
        
        print(f"\n{'='*80}")
        print(f"SUMMARY: {drug_name} vs {outcome_name}")
        print(f"{'='*80}")
        
        for outcome_idx, outcome_results in results.items():
            outcome_name = self.disease_names[outcome_idx]
            print(f"\n--- {outcome_name} (Index {outcome_idx}) ---")
            
            # Baseline differences
            baseline = outcome_results['baseline_differences']
            n_sig_diff = np.sum(baseline['theta_pvalues'] < 0.05)
            n_prs_diff = np.sum(baseline['prs_pvalues'] < 0.05)
            
            print(f"  Baseline Differences:")
            print(f"    - {n_sig_diff}/{len(baseline['theta_pvalues'])} signatures significantly different")
            print(f"    - {n_prs_diff}/{len(baseline['prs_pvalues'])} PRS significantly different")
            
            # Genetic enrichment
            enrichment = outcome_results['genetic_enrichment']
            n_sig_corr = np.sum(enrichment['signature_response_pvalues'] < 0.05)
            n_prs_corr = np.sum(enrichment['prs_response_pvalues'] < 0.05)
            
            print(f"  Genetic Enrichment:")
            print(f"    - {n_sig_corr}/{len(enrichment['signature_response_pvalues'])} signatures correlate with response")
            print(f"    - {n_prs_corr}/{len(enrichment['prs_response_pvalues'])} PRS correlate with response")
            
            # Interactions
            interactions = outcome_results['signature_interactions']
            n_interactions = np.sum(interactions['interaction_pvalues'] < 0.05)
            total_interactions = interactions['interaction_pvalues'].size
            
            print(f"  Signature-PRS Interactions:")
            print(f"    - {n_interactions}/{total_interactions} interactions significant")
            
            # Prediction performance
            prediction = outcome_results['response_prediction']
            print(f"  Response Prediction:")
            print(f"    - CV R²: {prediction['cv_r2_mean']:.3f} ± {prediction['cv_r2_std']:.3f}")
            
            # Top features
            top_sig_idx = np.argmax(prediction['theta_importance'])
            top_prs_idx = np.argmax(prediction['prs_importance'])
            
            print(f"    - Top signature: {top_sig_idx} (importance: {prediction['theta_importance'][top_sig_idx]:.3f})")
            print(f"    - Top PRS: {top_prs_idx} (importance: {prediction['prs_importance'][top_prs_idx]:.3f})")
    
    def save_results(self, results, drug_name, output_file=None):
        """Save analysis results to file."""
        
        if output_file is None:
            output_file = f"differential_response_{drug_name.lower()}_results.pkl"
        
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to: {output_file}")
        
        # Also save summary to CSV
        summary_data = []
        
        for outcome_idx, outcome_results in results.items():
            outcome_name = self.disease_names[outcome_idx]
            
            baseline = outcome_results['baseline_differences']
            enrichment = outcome_results['genetic_enrichment']
            prediction = outcome_results['response_prediction']
            
            summary_data.append({
                'outcome_index': outcome_idx,
                'outcome_name': outcome_name,
                'n_treated': outcome_results['n_treated'],
                'n_control': outcome_results['n_control'],
                'n_significant_signatures': np.sum(baseline['theta_pvalues'] < 0.05),
                'n_significant_prs': np.sum(baseline['prs_pvalues'] < 0.05),
                'n_signature_response_correlations': np.sum(enrichment['signature_response_pvalues'] < 0.05),
                'n_prs_response_correlations': np.sum(enrichment['prs_response_pvalues'] < 0.05),
                'cv_r2_mean': prediction['cv_r2_mean'],
                'cv_r2_std': prediction['cv_r2_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_file = output_file.replace('.pkl', '_summary.csv')
        summary_df.to_csv(csv_file, index=False)
        print(f"Summary saved to: {csv_file}")


def main():
    """Main function to run the differential response analysis."""
    
    print("Differential Drug Response Analysis")
    print("=" * 50)
    
    # Load your data (you'll need to adjust paths)
    print("Loading data...")
    
    # Load thetas
    thetas = np.load("/Users/sarahurbut/aladynoulli2/pyScripts/thetas.npy")
    processed_ids = np.load("/Users/sarahurbut/aladynoulli2/pyScripts/processed_patient_ids.npy").astype(int)
    
    # Load PRS matrix
    import torch
    G = torch.load("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/G_matrix.pt")
    G = G.detach().cpu().numpy()
    
    # Load PRS names
    prs_names = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/prs_names_with_head.csv')
    prs_labels = prs_names['Names'].tolist()
    
    # Load disease names
    disease_names_df = pd.read_csv("/Users/sarahurbut/aladynoulli2/pyScripts/disease_names.csv")
    disease_names = disease_names_df.iloc[:, 1].tolist()
    
    # Load covariates
    cov = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/matched_pce_df_400k.csv')
    cov.columns = cov.columns.str.strip()
    cov = cov.rename(columns={cov.columns[0]: 'eid'})
    cov['eid'] = cov['eid'].astype(int)
    
    # Create covariate dictionaries
    cov['enrollment'] = pd.to_datetime(cov['Enrollment_Date'], errors='coerce')
    cov['age_at_enroll'] = cov['enrollment'].dt.year - cov['birth_year']
    
    covariate_dicts = {
        'age_at_enroll': dict(zip(cov['eid'], cov['age_at_enroll'])),
        'sex': dict(zip(cov['eid'], cov['Sex'])),
        'dm2_prev': dict(zip(cov['eid'], cov['prev_dm'])),
        'antihtnbase': dict(zip(cov['eid'], cov['prev_ht'])),
        'smoke': dict(zip(cov['eid'], cov['SmokingStatusv2'])),
        'tchol': dict(zip(cov['eid'], cov['tchol'])),
        'hdl': dict(zip(cov['eid'], cov['hdl'])),
        'sbp': dict(zip(cov['eid'], cov['SBP']))
    }
    
    print("Data loaded successfully!")
    
    # Initialize analyzer
    analyzer = DifferentialResponseAnalyzer(
        thetas=thetas,
        processed_ids=processed_ids,
        prs_matrix=G,
        prs_labels=prs_labels,
        disease_names=disease_names,
        covariate_dicts=covariate_dicts
    )
    
    # Load drug prescription data
    print("\nLoading drug prescription data...")
    gp_scripts = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/gp_scripts.txt', sep='\t')
    
    # Find statins
    statin_keywords = ['statin', 'atorva', 'simva', 'rosuva', 'prava', 'fluva']
    statin_mask = gp_scripts['drug_name'].str.contains('|'.join(statin_keywords), case=False, na=False)
    statins = gp_scripts[statin_mask]
    
    # Find aspirin
    aspirin_keywords = ['aspirin', 'asa', 'disprin']
    aspirin_mask = gp_scripts['drug_name'].str.contains('|'.join(aspirin_keywords), case=False, na=False)
    aspirin = gp_scripts[aspirin_mask]
    
    # Find metformin
    metformin_keywords = ['metformin', 'glucophage']
    metformin_mask = gp_scripts['drug_name'].str.contains('|'.join(metformin_keywords), case=False, na=False)
    metformin = gp_scripts[metformin_mask]
    
    print(f"Found {len(statins)} statin prescriptions")
    print(f"Found {len(aspirin)} aspirin prescriptions")
    print(f"Found {len(metformin)} metformin prescriptions")
    
    # Run analyses
    print("\nRunning differential response analyses...")
    
    # 1. Statins vs CAD
    print("\n1. Analyzing Statins vs CAD...")
    cad_indices = [112, 113, 114, 115, 116]  # ASCVD composite
    statin_results = analyzer.analyze_drug_response_by_genetics(
        statins, cad_indices, "Statins", "CAD"
    )
    
    # 2. Aspirin vs Colorectal Cancer
    print("\n2. Analyzing Aspirin vs Colorectal Cancer...")
    crc_indices = [10, 11]  # Colorectal cancer
    aspirin_results = analyzer.analyze_drug_response_by_genetics(
        aspirin, crc_indices, "Aspirin", "Colorectal Cancer"
    )
    
    # 3. Metformin vs CAD
    print("\n3. Analyzing Metformin vs CAD...")
    metformin_results = analyzer.analyze_drug_response_by_genetics(
        metformin, cad_indices, "Metformin", "CAD"
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    analyzer.create_visualizations(statin_results, "Statins", "CAD")
    analyzer.create_visualizations(aspirin_results, "Aspirin", "Colorectal Cancer")
    analyzer.create_visualizations(metformin_results, "Metformin", "CAD")
    
    # Save results
    print("\nSaving results...")
    analyzer.save_results(statin_results, "Statins")
    analyzer.save_results(aspirin_results, "Aspirin")
    analyzer.save_results(metformin_results, "Metformin")
    
    print("\nAnalysis complete!")
    print("Check the generated files for detailed results.")


if __name__ == "__main__":
    main()


def analyze_heterogeneous_treatment_effects(drug_prescriptions, outcome_idx, outcome_name,
                                         thetas, disease_names, Y_tensor, covariate_dicts, 
                                         processed_ids):
    """
    Analyze heterogeneous treatment effects AFTER signature-based matching.
    This tests: "Among treated patients, after controlling for confounding via signatures,
    do some still respond differently based on their biology?"
    """
    print(f"\n=== HETEROGENEOUS TREATMENT EFFECTS ANALYSIS ===")
    print(f"Outcome: {outcome_name}")
    
    # Step 1: Get treated and control patients
    treated_eids = set(drug_prescriptions['eid'].unique())
    all_eids = set(processed_ids)
    control_eids = all_eids - treated_eids
    
    print(f"Treated patients: {len(treated_eids):,}")
    print(f"Control patients: {len(control_eids):,}")
    
    # Step 2: Build features for matching (signatures + key covariates)
    treated_features = build_matching_features(treated_eids, thetas, covariate_dicts, processed_ids)
    control_features = build_matching_features(control_eids, thetas, covariate_dicts, processed_ids)
    
    # Step 3: Perform signature-based matching
    matched_pairs = perform_signature_matching(treated_features, control_features)
    
    print(f"Matched pairs: {len(matched_pairs):,}")
    
    # Check if we got any matches
    if len(matched_pairs) == 0:
        print("Warning: No matched pairs found. Returning early.")
        return {
            'individual_effects': None,
            'signature_heterogeneity': None,
            'n_matched_treated': 0,
            'error': 'No matched pairs found'
        }
    
    # Step 4: Among matched treated patients, test for heterogeneous effects
    heterogeneous_effects = test_heterogeneous_effects(matched_pairs, outcome_idx, thetas, Y_tensor, processed_ids)
    
    # Check if we got valid results
    if heterogeneous_effects is None or 'error' in heterogeneous_effects:
        print("Warning: Heterogeneous effects analysis failed")
        return {
            'individual_effects': None,
            'signature_heterogeneity': None,
            'n_matched_treated': len(matched_pairs),
            'error': 'Heterogeneous effects analysis failed'
        }
    
    # Return consistent dictionary format
    return {
        'individual_effects': heterogeneous_effects['individual_effects'],
        'signature_heterogeneity': heterogeneous_effects['signature_heterogeneity'],
        'n_matched_treated': heterogeneous_effects['n_matched_treated'],
        'matched_pairs': matched_pairs
    }

def build_matching_features(eid_list, thetas, covariate_dicts, processed_ids):
    """Build features for matching: signatures + key covariates."""
    features_list = []
    eid_mapping = []
    
    # Create the mapping here
    eid_to_idx = {eid: idx for idx, eid in enumerate(processed_ids)}
    
    for eid in eid_list:
        if eid not in eid_to_idx:
            continue
            
        idx = eid_to_idx[eid]
        if idx >= thetas.shape[0]:
            continue
        
        # Get signature features (mean across time)
        sig_features = np.mean(thetas[idx, :, :], axis=1)  # (n_signatures,)
        
        # Get key covariates
        age = covariate_dicts.get('age_at_enroll', {}).get(int(eid), 0)
        sex = covariate_dicts.get('sex', {}).get(int(eid), 0)
        # Add other key covariates as needed
        
        # Combine features
        patient_features = np.concatenate([sig_features, [age, sex]])
        features_list.append(patient_features)
        eid_mapping.append(eid)
    
    return np.array(features_list), eid_mapping

def perform_signature_matching(treated_features, control_features):
    """Perform nearest neighbor matching on signature + covariate space."""
    from sklearn.neighbors import NearestNeighbors
    
    print(f"Debug: Treated features shape: {treated_features[0].shape}")
    print(f"Debug: Control features shape: {control_features[0].shape}")
    print(f"Debug: Treated EIDs: {len(treated_features[1])}")
    print(f"Debug: Control EIDs: {len(control_features[1])}")
    
    # Check if we have enough data
    if treated_features[0].shape[0] == 0 or control_features[0].shape[0] == 0:
        print("Warning: No features to match")
        return []
    
    # Scale features
    scaler = StandardScaler()
    treated_scaled = scaler.fit_transform(treated_features[0])
    control_scaled = scaler.transform(control_features[0])
    
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nbrs.fit(control_scaled)
    
    distances, indices = nbrs.kneighbors(treated_scaled)
    
    print(f"Debug: Found {len(distances)} potential matches")
    
    # Create matched pairs with more lenient distance threshold
    matched_pairs = []
    for i, (treated_idx, control_idx, distance) in enumerate(zip(range(len(treated_scaled)), indices.flatten(), distances.flatten())):
        # Use a more lenient threshold - adjust this value as needed
        if distance < 2.0:  # Increased from 0.5
            matched_pairs.append({
                'treated_eid': treated_features[1][treated_idx],
                'control_eid': control_features[1][control_idx],
                'distance': distance
            })
    
    print(f"Debug: Created {len(matched_pairs)} matched pairs with distance < 2.0")
    
    # If still no matches, try even more lenient threshold
    if len(matched_pairs) == 0:
        print("Warning: No matches with distance < 2.0, trying distance < 5.0")
        for i, (treated_idx, control_idx, distance) in enumerate(zip(range(len(treated_scaled)), indices.flatten(), distances.flatten())):
            if distance < 5.0:
                matched_pairs.append({
                    'treated_eid': treated_features[1][treated_idx],
                    'control_eid': control_features[1][control_idx],
                    'distance': distance
                })
        print(f"Debug: Created {len(matched_pairs)} matched pairs with distance < 5.0")
    
    return matched_pairs

def test_heterogeneous_effects(matched_pairs, outcome_idx, thetas, Y_tensor, processed_ids):
    """
    Among matched treated patients, test if there's heterogeneity in treatment effects
    based on their underlying biology (signatures).
    """
    
    # Create the mapping here
    eid_to_idx = {eid: idx for idx, eid in enumerate(processed_ids)}
    
    # Get matched treated patients
    matched_treated_eids = [pair['treated_eid'] for pair in matched_pairs]
    matched_treated_indices = [eid_to_idx[eid] for eid in matched_treated_eids]
    
    print(f"Debug: Found {len(matched_treated_indices)} matched treated patients")
    
    # Check if we have enough data
    if len(matched_treated_indices) < 2:
        print("Warning: Not enough matched treated patients for heterogeneity analysis")
        return {
            'individual_effects': None,
            'signature_heterogeneity': None,
            'n_matched_treated': len(matched_treated_indices),
            'error': 'Insufficient matched patients'
        }
    
    # Get their signature patterns
    matched_treated_thetas = thetas[matched_treated_indices, :, :]
    
    # Get their outcomes - convert to numpy if it's a torch tensor
    if hasattr(Y_tensor, 'detach'):
        # It's a torch tensor
        matched_treated_outcomes = Y_tensor[matched_treated_indices, outcome_idx, :].detach().cpu().numpy()
    else:
        # It's already numpy
        matched_treated_outcomes = Y_tensor[matched_treated_indices, outcome_idx, :]
    
    print(f"Debug: Theta shape: {matched_treated_thetas.shape}, Outcomes shape: {matched_treated_outcomes.shape}")
    
    # Calculate treatment effect for each patient
    # This is the key: what's their individual "response"?
    individual_treatment_effects = calculate_individual_treatment_effects(
        matched_treated_thetas, matched_treated_outcomes
    )
    
    print(f"Debug: Treatment effects shape: {individual_treatment_effects.shape}")
    
    # Now test: do patients with different signature patterns have different treatment effects?
    signature_heterogeneity = test_signature_heterogeneity(
        matched_treated_thetas, individual_treatment_effects
    )
    
    return {
        'individual_effects': individual_treatment_effects,
        'signature_heterogeneity': signature_heterogeneity,
        'n_matched_treated': len(matched_treated_indices)
    }

def calculate_individual_treatment_effects(thetas, outcomes):
    """
    Calculate individual treatment effect for each patient.
    This is the "response" metric.
    """
    # Option 1: Event rate reduction (if you have before/after data)
    # Option 2: Time to event
    # Option 3: Event count
    
    # For now, using event count as proxy for treatment effect
    # (lower = better response)
    treatment_effects = np.sum(outcomes, axis=1)
    
    return treatment_effects

def test_signature_heterogeneity(thetas, treatment_effects):
    """
    Test if patients with different signature patterns have different treatment effects.
    """
    
    # Check if we have enough data
    if thetas is None or treatment_effects is None:
        return {'error': 'No data provided'}
    
    if len(treatment_effects) < 2:
        return {'error': f'Insufficient data: only {len(treatment_effects)} patients'}
    
    # Calculate signature features
    mean_signatures = np.mean(thetas, axis=2)  # (n_patients, n_signatures)
    
    print(f"Debug: Mean signatures shape: {mean_signatures.shape}")
    
    # Test each signature for heterogeneity
    heterogeneity_results = {}
    
    for sig_idx in range(mean_signatures.shape[1]):
        sig_name = f"Signature_{sig_idx}"
        
        try:
            # Test correlation between signature and treatment effect
            if len(mean_signatures[:, sig_idx]) >= 2 and len(treatment_effects) >= 2:
                correlation, pval = stats.pearsonr(mean_signatures[:, sig_idx], treatment_effects)
            else:
                correlation, pval = np.nan, np.nan
            
            # Test if high vs low signature patients have different effects
            median_sig = np.median(mean_signatures[:, sig_idx])
            high_sig_mask = mean_signatures[:, sig_idx] > median_sig
            low_sig_mask = mean_signatures[:, sig_idx] <= median_sig
            
            high_effects = treatment_effects[high_sig_mask]
            low_effects = treatment_effects[low_sig_mask]
            
            if len(high_effects) > 0 and len(low_effects) > 0:
                t_stat, t_pval = stats.ttest_ind(high_effects, low_effects)
            else:
                t_stat, t_pval = np.nan, np.nan
            
            heterogeneity_results[sig_name] = {
                'correlation': correlation,
                'correlation_pval': pval,
                'high_vs_low_tstat': t_stat,
                'high_vs_low_pval': t_pval,
                'high_effects_mean': np.mean(high_effects) if len(high_effects) > 0 else np.nan,
                'low_effects_mean': np.mean(low_effects) if len(low_effects) > 0 else np.nan,
                'significant': pval < 0.05 or t_pval < 0.05
            }
        except Exception as e:
            print(f"Warning: Error processing signature {sig_idx}: {e}")
            heterogeneity_results[sig_name] = {
                'correlation': np.nan,
                'correlation_pval': np.nan,
                'high_vs_low_tstat': np.nan,
                'high_vs_low_pval': np.nan,
                'high_effects_mean': np.nan,
                'low_effects_mean': np.nan,
                'significant': False,
                'error': str(e)
            }
    
    return heterogeneity_results

def create_heterogeneous_effects_visualization(heterogeneous_results, outcome_name, drug_name):
    """
    Create comprehensive visualization explaining heterogeneous treatment effects.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Conceptual explanation plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.text(0.1, 0.9, f'HETEROGENEOUS TREATMENT EFFECTS\n{drug_name} on {outcome_name}', 
              fontsize=14, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.8, 'What we\'re testing:', fontsize=12, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.7, '• After matching on signatures to control\n  confounding...', fontsize=10, transform=ax1.transAxes)
    ax1.text(0.1, 0.6, '• Do patients with different signature\n  patterns respond differently to treatment?', fontsize=10, transform=ax1.transAxes)
    ax1.text(0.1, 0.5, '• This could reveal personalized\n  medicine opportunities!', fontsize=10, transform=ax1.transAxes)
    ax1.text(0.1, 0.3, 'Key Insight:', fontsize=12, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.2, 'High-risk patients (high Signature_5)\nmay get GREATEST absolute benefit\nfrom treatment!', fontsize=10, color='red', transform=ax1.transAxes)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Signature correlation heatmap
    ax2 = plt.subplot(2, 3, 2)
    correlations = []
    p_values = []
    sig_names = []
    
    for sig_name, result in heterogeneous_results['signature_heterogeneity'].items():
        if 'error' not in result:
            correlations.append(result['correlation'])
            p_values.append(result['correlation_pval'])
            sig_names.append(sig_name)
    
    # Create correlation matrix for heatmap
    corr_matrix = np.array(correlations).reshape(-1, 1)
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    ax2.set_yticks(range(len(sig_names)))
    ax2.set_yticklabels(sig_names)
    ax2.set_xticks([])
    ax2.set_title('Signature-Treatment Effect\nCorrelations', fontweight='bold')
    
    # Add correlation values as text
    for i, corr in enumerate(correlations):
        color = 'white' if abs(corr) > 0.15 else 'black'
        ax2.text(0, i, f'{corr:.3f}', ha='center', va='center', color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Correlation')
    
    # 3. Top signatures by effect size
    ax3 = plt.subplot(2, 3, 3)
    
    # Sort by absolute correlation
    sig_effects = [(sig_names[i], abs(correlations[i]), correlations[i]) 
                   for i in range(len(correlations))]
    sig_effects.sort(key=lambda x: x[1], reverse=True)
    
    top_sigs = [x[0] for x in sig_effects[:10]]
    top_corrs = [x[2] for x in sig_effects[:10]]
    colors = ['red' if x < 0 else 'blue' for x in top_corrs]
    
    bars = ax3.barh(range(len(top_sigs)), top_corrs, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(top_sigs)))
    ax3.set_yticklabels(top_sigs)
    ax3.set_xlabel('Correlation with Treatment Effect')
    ax3.set_title('Top 10 Signatures by\nEffect Size', fontweight='bold')
    ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    # 4. Signature_5 detailed analysis (if it exists)
    ax4 = plt.subplot(2, 3, 4)
    sig5_result = heterogeneous_results['signature_heterogeneity'].get('Signature_5')
    
    if sig5_result and 'error' not in sig5_result:
        # Create conceptual plot
        x_pos = [0, 1]
        high_effect = sig5_result['high_effects_mean']
        low_effect = sig5_result['low_effects_mean']
        
        bars = ax4.bar(x_pos, [high_effect, low_effect], 
                       color=['red', 'blue'], alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(['High\nSignature_5\n(High CVD Risk)', 'Low\nSignature_5\n(Low CVD Risk)'])
        ax4.set_ylabel('Treatment Effect Size')
        ax4.set_title('Signature_5: CVD Risk vs\nTreatment Response', fontweight='bold')
        
        # Add effect difference
        effect_diff = high_effect - low_effect
        ax4.text(0.5, max(high_effect, low_effect) + 0.01, 
                f'Difference: {effect_diff:.3f}', 
                ha='center', fontweight='bold', color='red')
        
        # Add p-value
        pval = sig5_result['high_vs_low_pval']
        ax4.text(0.5, 0.5 * (high_effect + low_effect), 
                f'p = {pval:.2e}', ha='center', fontweight='bold')
    
    # 5. Treatment effect distribution
    ax5 = plt.subplot(2, 3, 5)
    if heterogeneous_results['individual_effects'] is not None:
        effects = heterogeneous_results['individual_effects']
        ax5.hist(effects, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(np.mean(effects), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(effects):.3f}')
        ax5.set_xlabel('Individual Treatment Effect')
        ax5.set_ylabel('Number of Patients')
        ax5.set_title('Distribution of Individual\nTreatment Effects', fontweight='bold')
        ax5.legend()
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.text(0.1, 0.9, 'ANALYSIS SUMMARY', fontsize=14, fontweight='bold', transform=ax6.transAxes)
    
    n_matched = heterogeneous_results['n_matched_treated']
    n_significant = sum(1 for r in heterogeneous_results['signature_heterogeneity'].values() 
                       if 'error' not in r and r.get('significant', False))
    
    ax6.text(0.1, 0.8, f'Matched Patients: {n_matched:,}', fontsize=12, transform=ax6.transAxes)
    ax6.text(0.1, 0.7, f'Significant Signatures: {n_significant}/21', fontsize=12, transform=ax6.transAxes)
    
    if sig5_result and 'error' not in sig5_result:
        ax6.text(0.1, 0.6, 'Key Finding:', fontsize=12, fontweight='bold', transform=ax6.transAxes)
        ax6.text(0.1, 0.5, f'Signature_5 correlation:', fontsize=10, transform=ax6.transAxes)
        ax6.text(0.1, 0.4, f'{sig5_result["correlation"]:.3f}', fontsize=10, color='red', transform=ax6.transAxes)
        ax6.text(0.1, 0.3, f'p = {sig5_result["correlation_pval"]:.2e}', fontsize=10, transform=ax6.transAxes)
    
    ax6.text(0.1, 0.1, 'Interpretation: High CVD risk\npatients may get greatest\nabsolute benefit from treatment!', 
              fontsize=10, color='red', transform=ax6.transAxes)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Heterogeneous Treatment Effects: {drug_name} on {outcome_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def create_signature_balance_plot(matched_pairs, thetas, processed_ids, outcome_name):
    """
    Create a plot showing signature balance between treated and matched controls.
    This helps assess matching quality and identify any remaining confounding.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract matched treated and control indices
    treated_eids = [pair['treated_eid'] for pair in matched_pairs]
    control_eids = [pair['control_eid'] for pair in matched_pairs]
    
    # Create mapping
    eid_to_idx = {eid: idx for idx, eid in enumerate(processed_ids)}
    
    # Get indices
    treated_indices = [eid_to_idx[eid] for eid in treated_eids if eid in eid_to_idx]
    control_indices = [eid_to_idx[eid] for eid in control_eids if eid in eid_to_idx]
    
    if len(treated_indices) == 0 or len(control_indices) == 0:
        print("Warning: No valid indices found for plotting")
        return None
    
    # Calculate mean signature loadings for each group
    treated_sig_means = thetas[treated_indices, :, :].mean(axis=(0, 2))  # (n_signatures,)
    control_sig_means = thetas[control_indices, :, :].mean(axis=(0, 2))  # (n_signatures,)
    
    # Calculate differences
    sig_differences = treated_sig_means - control_sig_means
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Raw differences
    x_pos = np.arange(len(sig_differences))
    colors = ['red' if abs(diff) > 0.01 else 'blue' for diff in sig_differences]
    
    bars1 = ax1.bar(x_pos, sig_differences, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax1.axhline(0.01, color='red', linestyle='--', alpha=0.7, label='±0.01 threshold')
    ax1.axhline(-0.01, color='red', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Signature Index')
    ax1.set_ylabel('Difference (Treated - Matched Controls)')
    ax1.set_title(f'Signature Balance After Matching: {outcome_name}\nRed bars = potentially problematic differences', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add signature labels for significant differences
    for i, diff in enumerate(sig_differences):
        if abs(diff) > 0.01:
            ax1.text(i, diff + (0.002 if diff > 0 else -0.002), 
                    f'Sig_{i}', ha='center', va='bottom' if diff > 0 else 'top', 
                    fontsize=8, fontweight='bold')
    
    # Plot 2: Standardized differences (like SMD)
    treated_sig_stds = thetas[treated_indices, :, :].std(axis=(0, 2))
    control_sig_stds = thetas[control_indices, :, :].std(axis=(0, 2))
    
    # Calculate standardized mean differences
    smds = np.abs(sig_differences) / np.sqrt((treated_sig_stds**2 + control_sig_stds**2) / 2)
    
    bars2 = ax2.bar(x_pos, smds, color='orange', alpha=0.7, edgecolor='black')
    ax2.axhline(0.1, color='red', linestyle='--', alpha=0.7, label='SMD=0.1 threshold')
    
    ax2.set_xlabel('Signature Index')
    ax2.set_ylabel('Standardized Mean Difference (SMD)')
    ax2.set_title('Standardized Signature Differences\nSMD > 0.1 indicates potential imbalance', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add signature labels for high SMD
    for i, smd in enumerate(smds):
        if smd > 0.1:
            ax2.text(i, smd + 0.01, f'Sig_{i}', ha='center', va='bottom', 
                    fontsize=8, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n=== SIGNATURE BALANCE SUMMARY ===")
    print(f"Total signatures: {len(sig_differences)}")
    print(f"Signatures with |difference| > 0.01: {sum(abs(sig_differences) > 0.01)}")
    print(f"Signatures with SMD > 0.1: {sum(smds > 0.1)}")
    
    # Highlight problematic signatures
    problematic_sigs = []
    for i, (diff, smd) in enumerate(zip(sig_differences, smds)):
        if abs(diff) > 0.01 or smd > 0.1:
            problematic_sigs.append((i, diff, smd))
    
    if problematic_sigs:
        print(f"\nPotentially problematic signatures:")
        for sig_idx, diff, smd in problematic_sigs:
            print(f"  Signature_{sig_idx}: diff={diff:.4f}, SMD={smd:.3f}")
    
    return fig