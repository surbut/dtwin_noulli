#!/usr/bin/env python3
"""
Run True Differential Treatment Effects Analysis for Aspirin-Colorectal Cancer Prevention

This script takes your aspirin matching results and runs the true differential analysis
to discover which signature patterns predict differential cancer prevention benefits.

Author: Sarah Urbut  
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from true_differential_treatment_effects import run_true_differential_analysis

def extract_matching_results_from_aspirin_analysis(aspirin_results):
    """
    Extract the necessary components from your aspirin analysis results
    for differential treatment effects analysis.
    
    Parameters:
    - aspirin_results: Results from aspirin_colorectal_analysis()
        Expected structure: dict_keys(['hazard_ratio_results', 'matched_patients', 'cohort_sizes', 'matching_features'])
    
    Returns:
    - Dictionary with extracted components needed for differential analysis
    """
    
    print("=== EXTRACTING ASPIRIN MATCHING RESULTS ===")
    print(f"Available keys: {list(aspirin_results.keys())}")
    
    # Extract from the new comprehensive results structure
    if 'matched_patients' in aspirin_results:
        matched_patients = aspirin_results['matched_patients']
        
        # Get the matched EIDs
        matched_treated_eids = matched_patients['treated_eids']
        matched_control_eids = matched_patients['control_eids']
        
        print(f"Found {len(matched_treated_eids)} matched treated patients")
        print(f"Found {len(matched_control_eids)} matched control patients")
        
        # Create matched pairs list in the format needed for differential analysis
        matched_pairs = []
        for treated_eid, control_eid in zip(matched_treated_eids, matched_control_eids):
            matched_pairs.append({
                'treated_eid': treated_eid,
                'control_eid': control_eid,
                'distance': 0  # Distance not needed for this analysis
            })
        
        print(f"Created {len(matched_pairs)} matched pairs")
        
        # For treated patient info, we need to get this from the original analysis
        # Since the aspirin analysis uses ObservationalTreatmentPatternLearner,
        # we need to reconstruct or pass this information
        
        # For now, we'll use the matched treated EIDs and assume we can get times
        # This might need adjustment based on your specific setup
        treated_eids = matched_treated_eids  # These are the treated patients we have
        
        # Get treatment times from the new structure
        if 'treatment_times' in aspirin_results:
            treated_times = aspirin_results['treatment_times']['treated_times']
            print(f"Found {len(treated_times)} treatment times")
        else:
            print("Warning: Treatment times not found in results")
            treated_times = []
        
        return {
            'matched_pairs': matched_pairs,
            'treated_eids': treated_eids,
            'treated_times': treated_times,  # Empty - needs to be added to aspirin analysis
            'n_matched_pairs': len(matched_pairs)
        }
    
    else:
        print("❌ Could not find 'matched_patients' key in aspirin_results")
        print("Available keys:", list(aspirin_results.keys()))
        return None

def run_aspirin_colorectal_differential_analysis(aspirin_results, thetas, Y, processed_ids, 
                                               covariate_dicts, colorectal_indices=[10, 11]):
    """
    Run the complete differential treatment effects analysis for aspirin-colorectal cancer.
    
    Parameters:
    - aspirin_results: Results from your aspirin_colorectal_analysis()
    - thetas: Signature loadings
    - Y: Outcome tensor
    - processed_ids: Patient IDs
    - covariate_dicts: Covariate dictionaries
    - colorectal_indices: Indices for colorectal cancer events
    
    Returns:
    - Complete differential analysis results
    """
    
    print("="*80)
    print("ASPIRIN-COLORECTAL CANCER DIFFERENTIAL TREATMENT EFFECTS")
    print("="*80)
    print("This analysis will discover which signature patterns predict")
    print("differential aspirin benefits for colorectal cancer prevention!")
    print("="*80)
    
    # Step 1: Extract matching results
    matching_data = extract_matching_results_from_aspirin_analysis(aspirin_results)
    
    if matching_data is None or matching_data['n_matched_pairs'] < 50:
        print("❌ Insufficient matching data for differential analysis")
        print(f"   Found only {matching_data['n_matched_pairs'] if matching_data else 0} matched pairs")
        print("   Need at least 50 pairs for meaningful analysis")
        return None
    
    print(f"✅ Successfully extracted {matching_data['n_matched_pairs']} matched pairs")
    
    # Step 2: Run differential analysis for each colorectal cancer type
    all_results = {}
    
    for crc_idx in colorectal_indices:
        crc_name = f"Colorectal_Cancer_{crc_idx}"
        print(f"\n{'='*60}")
        print(f"ANALYZING: {crc_name}")
        print(f"{'='*60}")
        
        # Run the true differential analysis
        differential_results = run_true_differential_analysis(
            matched_pairs=matching_data['matched_pairs'],
            treated_eids=matching_data['treated_eids'],
            treated_times=matching_data['treated_times'],
            Y_tensor=Y,
            outcome_idx=crc_idx,
            outcome_name=crc_name,
            thetas=thetas,
            processed_ids=processed_ids,
            covariate_dicts=covariate_dicts,
            drug_name="Aspirin"
        )
        
        if differential_results is not None:
            all_results[crc_name] = differential_results
            
            # Print key findings
            summary = differential_results['summary']
            print(f"\n🎯 KEY FINDINGS FOR {crc_name}:")
            print(f"   📊 Patients analyzed: {summary['n_patients']}")
            print(f"   💊 Overall aspirin effect: {summary['overall_treatment_effect']:.4f} events/year")
            print(f"   🧬 Signatures with differential effects: {summary['n_signatures_with_differential_effects']}")
            
            if summary['n_signatures_with_differential_effects'] > 0:
                top_sig = summary['most_differential_signature']
                print(f"   🏆 Most differential signature: {top_sig[0]} (effect size: {top_sig[1]['effect_size']:.3f})")
                
                if top_sig[1]['effect_size'] > 0.3:  # Large effect size
                    print(f"   🚀 POTENTIAL BREAKTHROUGH: Strong differential effect found!")
                    print(f"      This could enable personalized aspirin recommendations for CRC prevention!")
    
    # Step 3: Create combined analysis if multiple cancer types
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"COMBINED COLORECTAL CANCER ANALYSIS")
        print(f"{'='*80}")
        
        # Find signatures that are differential across multiple cancer types
        common_differential_signatures = find_common_differential_signatures(all_results)
        
        if len(common_differential_signatures) > 0:
            print(f"🔬 SIGNATURES DIFFERENTIAL ACROSS MULTIPLE CRC TYPES:")
            for sig_name, effect_sizes in common_differential_signatures.items():
                print(f"   {sig_name}: {effect_sizes}")
                print(f"      → Could represent core aspirin-cancer prevention biology!")
    
    # Step 4: Clinical interpretation
    print(f"\n{'='*80}")
    print(f"CLINICAL IMPLICATIONS")
    print(f"{'='*80}")
    
    total_differential_signatures = sum(r['summary']['n_signatures_with_differential_effects'] 
                                      for r in all_results.values())
    
    if total_differential_signatures == 0:
        print("📋 CLINICAL CONCLUSION:")
        print("   • All patients appear to benefit similarly from aspirin for CRC prevention")
        print("   • Current population-based guidelines are appropriate")
        print("   • No personalization needed based on signature patterns")
        
    else:
        print("🎯 CLINICAL CONCLUSION:")
        print("   • DIFFERENTIAL ASPIRIN BENEFITS FOUND!")
        print("   • Some patients benefit MORE from aspirin for CRC prevention")
        print("   • Signature-based personalization could be valuable")
        print("   • Could reduce unnecessary aspirin use (and bleeding risk)")
        print("   • Could improve CRC prevention in high-responder patients")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Validate findings in independent cohort")
        print(f"   2. Investigate biological mechanisms of differential signatures")
        print(f"   3. Develop clinical decision tool")
        print(f"   4. Design personalized aspirin prevention trial")
    
    return all_results

def find_common_differential_signatures(all_results):
    """
    Find signatures that show differential effects across multiple cancer types.
    """
    
    signature_effects = {}
    
    for cancer_type, results in all_results.items():
        differential_results = results['differential_results']
        
        for sig_name, sig_result in differential_results.items():
            if sig_result['significant_difference']:
                if sig_name not in signature_effects:
                    signature_effects[sig_name] = {}
                signature_effects[sig_name][cancer_type] = sig_result['effect_size']
    
    # Keep only signatures differential in multiple cancer types
    common_signatures = {sig: effects for sig, effects in signature_effects.items() 
                        if len(effects) > 1}
    
    return common_signatures

def create_aspirin_summary_report(all_results, output_file="aspirin_differential_summary.txt"):
    """
    Create a comprehensive summary report of the aspirin differential analysis.
    """
    
    with open(output_file, 'w') as f:
        f.write("ASPIRIN-COLORECTAL CANCER DIFFERENTIAL TREATMENT EFFECTS ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY:\n")
        f.write("-"*40 + "\n")
        
        total_patients = sum(r['summary']['n_patients'] for r in all_results.values())
        total_differential = sum(r['summary']['n_signatures_with_differential_effects'] 
                               for r in all_results.values())
        
        f.write(f"Total patients analyzed: {total_patients}\n")
        f.write(f"Cancer types analyzed: {len(all_results)}\n")
        f.write(f"Total differential signatures found: {total_differential}\n\n")
        
        for cancer_type, results in all_results.items():
            f.write(f"\n{cancer_type}:\n")
            f.write("-" * len(cancer_type) + "-\n")
            
            summary = results['summary']
            f.write(f"  Patients: {summary['n_patients']}\n")
            f.write(f"  Overall treatment effect: {summary['overall_treatment_effect']:.4f}\n")
            f.write(f"  Differential signatures: {summary['n_signatures_with_differential_effects']}\n")
            
            if summary['n_signatures_with_differential_effects'] > 0:
                top_sig = summary['most_differential_signature']
                f.write(f"  Top signature: {top_sig[0]} (effect size: {top_sig[1]['effect_size']:.3f})\n")
        
        f.write(f"\nDetailed results saved in analysis objects.\n")
    
    print(f"📝 Summary report saved to: {output_file}")

# Example usage function
def main_aspirin_differential_analysis(aspirin_results, thetas, Y, processed_ids, 
                                     covariate_dicts, colorectal_indices=[10, 11]):
    """
    Main function to run the complete aspirin differential analysis.
    """
    
    # Run the analysis
    results = run_aspirin_colorectal_differential_analysis(
        aspirin_results=aspirin_results,
        thetas=thetas,
        Y=Y,
        processed_ids=processed_ids,
        covariate_dicts=covariate_dicts,
        colorectal_indices=colorectal_indices
    )
    
    if results is not None:
        # Create summary report
        create_aspirin_summary_report(results)
        
        # Show visualizations
        for cancer_type, result_data in results.items():
            if 'visualization' in result_data and result_data['visualization'] is not None:
                plt.show()
    
    return results

if __name__ == "__main__":
    print("Aspirin-Colorectal Cancer Differential Treatment Effects Analysis")
    print("Use main_aspirin_differential_analysis() function")