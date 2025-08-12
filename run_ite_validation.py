#!/usr/bin/env python3
"""
Run ITE Validation Test

This script loads the necessary data and runs the complete validation test
for the fixed Individual Treatment Effect (ITE) calculation.

Author: Sarah Urbut
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add scripts directory to path
sys.path.append('/Users/sarahurbut/dtwin_noulli/scripts')

# Import the validation test functions
from test_fixed_ite_calculation import test_statin_ite_calculation, create_ite_validation_visualization

def load_data_and_run_statin_analysis():
    """
    Load all necessary data and run the statin analysis to get comprehensive results.
    """
    
    print("="*80)
    print("LOADING DATA FOR ITE VALIDATION TEST")
    print("="*80)
    
    # Load core data
    print("Loading core data files...")
    try:
        thetas = np.load('/Users/sarahurbut/dtwin_noulli/data/thetas.npy')
        processed_ids = np.load('/Users/sarahurbut/dtwin_noulli/data/processed_ids.npy')
        Y = np.load('/Users/sarahurbut/dtwin_noulli/data/Y.npy')
        covariate_dicts = np.load('/Users/sarahurbut/dtwin_noulli/data/covariate_dicts.npy', allow_pickle=True).item()
        
        print(f"✅ Loaded thetas: {thetas.shape}")
        print(f"✅ Loaded processed_ids: {processed_ids.shape}")
        print(f"✅ Loaded Y: {Y.shape}")
        print(f"✅ Loaded covariate_dicts: {len(covariate_dicts)} covariates")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
        
    # Import and run the statin analysis to get comprehensive results
    print(f"\n{'='*60}")
    print("RUNNING STATIN ANALYSIS TO GET COMPREHENSIVE RESULTS")
    print(f"{'='*60}")
    
    try:
        from simple_treatment_analysis import simple_treatment_analysis
        
        print("Running simple_treatment_analysis()...")
        statin_results = simple_treatment_analysis()
        
        if statin_results is None:
            print("❌ Statin analysis returned None")
            return None
            
        print(f"✅ Successfully ran statin analysis")
        print(f"Results keys: {list(statin_results.keys())}")
        
        # Show key metrics
        hr_results = statin_results['hazard_ratio_results']
        print(f"Hazard Ratio: {hr_results['hazard_ratio']:.3f}")
        print(f"95% CI: {hr_results['hr_ci_lower']:.3f} - {hr_results['hr_ci_upper']:.3f}")
        print(f"P-value: {hr_results['p_value']:.4f}")
        print(f"Matched pairs: {statin_results['cohort_sizes']['n_treated']:,}")
        
    except Exception as e:
        print(f"❌ Error running statin analysis: {e}")
        return None
    
    # Now run the ITE validation test
    print(f"\n{'='*60}")
    print("RUNNING ITE VALIDATION TEST")
    print(f"{'='*60}")
    
    try:
        validation_results = test_statin_ite_calculation(
            statin_results=statin_results,
            thetas=thetas,
            Y=Y,
            processed_ids=processed_ids,
            covariate_dicts=covariate_dicts
        )
        
        if validation_results is None:
            print("❌ ITE validation test failed")
            return None
            
        print(f"✅ ITE validation test completed")
        print(f"Status: {validation_results['validation_status']}")
        
        return {
            'statin_results': statin_results,
            'validation_results': validation_results,
            'data': {
                'thetas': thetas,
                'Y': Y,
                'processed_ids': processed_ids,
                'covariate_dicts': covariate_dicts
            }
        }
        
    except Exception as e:
        print(f"❌ Error in ITE validation: {e}")
        return None

def run_complete_validation_with_visualization():
    """
    Run the complete validation test with visualization.
    """
    
    print("COMPLETE ITE VALIDATION TEST")
    print("="*50)
    
    # Run the validation
    results = load_data_and_run_statin_analysis()
    
    if results is None:
        print("❌ Validation test failed")
        return None
        
    validation_results = results['validation_results']
    
    # Create and show visualization
    print(f"\n{'='*60}")
    print("CREATING VALIDATION VISUALIZATION")
    print(f"{'='*60}")
    
    try:
        fig = create_ite_validation_visualization(validation_results)
        
        if fig is not None:
            plt.show()
            print("✅ Visualization created and displayed")
        else:
            print("⚠️ Could not create visualization")
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    status = validation_results['validation_status']
    ite_effect = validation_results['ite_effect']
    
    print(f"Validation Status: {status}")
    print(f"ITE Effect: {ite_effect:.6f} hazard reduction per year")
    
    if status == 'PASSED':
        print("✅ SUCCESS: The updated ITE calculation now aligns with the successful HR methodology!")
        print("   The treatment effect is properly captured.")
        print("   You can now run differential analysis on the statin results.")
        
        # Mark todo as completed
        print(f"\n📋 TODO: Validation completed successfully")
        
    elif status == 'FAILED':
        print("❌ FAILURE: The ITE calculation is still not working properly.")
        print("   The treatment effect is not being captured correctly.")
        print("   Further debugging of the ITE methodology may be needed.")
        
    elif status == 'UNEXPECTED':
        print("⚠️ UNEXPECTED: The ITE shows treatment increases risk.")
        print("   This contradicts the successful HR=0.70 result.")
        print("   The calculation methodology may need review.")
        
    return results

if __name__ == "__main__":
    # Run the complete validation test
    final_results = run_complete_validation_with_visualization()
    
    if final_results is not None and final_results['validation_results']['validation_status'] == 'PASSED':
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Run differential analysis on the validated statin results")
        print(f"2. Compare findings with aspirin differential effects")
        print(f"3. Investigate biological mechanisms of differential signatures")
        
        # Show how to access the results for next steps
        statin_results = final_results['statin_results']
        print(f"\nYou can now use these results for differential analysis:")
        print(f"  - statin_results with {statin_results['cohort_sizes']['n_treated']:,} matched pairs")
        print(f"  - Comprehensive matching data and treatment times available")
        print(f"  - Ready for true_differential_treatment_effects.py analysis")