import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_signature_differences_with_variability(learner):
    """
    Analyze signature differences accounting for variability and effect sizes
    """
    
    # Get the comparison data
    comparison = learner.compare_treated_vs_untreated_patterns()
    if comparison is None:
        print("No comparison data available")
        return None
    
    # Extract treated and untreated patterns for more detailed analysis
    treated_patterns = learner.treatment_patterns['pre_treatment_signatures']
    
    # Re-sample untreated patterns for fair comparison
    never_treated_eids = learner.treatment_patterns['never_treated']
    untreated_patterns = []
    
    for eid in never_treated_eids[:len(treated_patterns)]:
        try:
            patient_idx = np.where(learner.processed_ids == eid)[0][0]
            sample_time = np.random.randint(12, min(40, learner.signatures.shape[2] - 12))
            pattern = learner.signatures[patient_idx, :, sample_time-12:sample_time]
            if pattern.shape == (learner.signatures.shape[1], 12):
                untreated_patterns.append(pattern)
        except:
            continue
    
    untreated_patterns = np.array(untreated_patterns)
    
    # Calculate comprehensive statistics for each signature
    signature_stats = []
    
    for s in range(min(treated_patterns.shape[1], untreated_patterns.shape[1])):
        # Final time point levels
        treated_levels = treated_patterns[:, s, -1]
        untreated_levels = untreated_patterns[:, s, -1]
        
        # Basic statistics
        treated_mean = np.mean(treated_levels)
        treated_std = np.std(treated_levels)
        untreated_mean = np.mean(untreated_levels)
        untreated_std = np.std(untreated_levels)
        
        # Effect size (Cohen's d) with numerical safeguards
        pooled_std = np.sqrt((treated_std**2 + untreated_std**2) / 2)
        if pooled_std > 1e-10:  # Avoid division by zero
            cohens_d = (treated_mean - untreated_mean) / pooled_std
        else:
            cohens_d = 0.0
        
        # Cap extreme values
        cohens_d = np.clip(cohens_d, -10, 10)
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(treated_levels, untreated_levels)
        
        # Variance ratio (measure of relative variability)
        variance_ratio = treated_std / untreated_std if untreated_std > 0 else np.inf
        
        signature_stats.append({
            'signature': s,
            'treated_mean': treated_mean,
            'treated_std': treated_std,
            'untreated_mean': untreated_mean,
            'untreated_std': untreated_std,
            'raw_difference': treated_mean - untreated_mean,
            'cohens_d': cohens_d,
            'p_value': p_val,
            'variance_ratio': variance_ratio,
            'significant': p_val < 0.05
        })
    
    return pd.DataFrame(signature_stats)

def plot_signature_evolution_over_time(learner, top_signatures=None, n_signatures=8):
    """
    Create stacked bar plot showing signature evolution relative to treatment start
    """
    
    treated_patterns = learner.treatment_patterns['pre_treatment_signatures']
    if len(treated_patterns) == 0:
        print("No treated patterns available")
        return
    
    # If no specific signatures provided, use the most variable ones
    if top_signatures is None:
        # Calculate which signatures change most over time
        signature_variability = []
        for s in range(treated_patterns.shape[1]):
            time_series = treated_patterns[:, s, :].mean(axis=0)  # Average across patients
            variability = np.std(time_series)  # Temporal variability
            signature_variability.append((s, variability))
        
        # Sort by variability and take top N
        signature_variability.sort(key=lambda x: x[1], reverse=True)
        top_signatures = [s[0] for s in signature_variability[:n_signatures]]
    
    # Create the evolution plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Signature Evolution Analysis', fontsize=16)
    
    # 1. Average signature levels over time (stacked)
    ax1 = axes[0, 0]
    
    time_points = range(-12, 0)  # 12 months before treatment
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_signatures)))
    
    # Calculate mean signature levels at each time point
    signature_means = np.zeros((len(top_signatures), 12))
    for i, sig_idx in enumerate(top_signatures):
        signature_means[i, :] = treated_patterns[:, sig_idx, :].mean(axis=0)
    
    # Create stacked area plot
    ax1.stackplot(time_points, signature_means, labels=[f'Sig {s}' for s in top_signatures], 
                  colors=colors, alpha=0.7)
    ax1.set_xlabel('Months Before Treatment')
    ax1.set_ylabel('Average Signature Loading')
    ax1.set_title('Signature Evolution (Stacked)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual signature trajectories
    ax2 = axes[0, 1]
    
    for i, sig_idx in enumerate(top_signatures):
        mean_traj = treated_patterns[:, sig_idx, :].mean(axis=0)
        std_traj = treated_patterns[:, sig_idx, :].std(axis=0)
        
        ax2.plot(time_points, mean_traj, color=colors[i], linewidth=2, 
                label=f'Sig {sig_idx}')
        ax2.fill_between(time_points, 
                        mean_traj - std_traj, 
                        mean_traj + std_traj, 
                        color=colors[i], alpha=0.2)
    
    ax2.set_xlabel('Months Before Treatment')
    ax2.set_ylabel('Signature Loading')
    ax2.set_title('Individual Signature Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Signature changes (slopes)
    ax3 = axes[1, 0]
    
    signature_slopes = []
    signature_names = []
    
    for sig_idx in top_signatures:
        # Calculate slope for each patient, then average
        slopes = []
        for p in range(treated_patterns.shape[0]):
            traj = treated_patterns[p, sig_idx, :]
            slope = np.polyfit(range(len(traj)), traj, 1)[0]
            slopes.append(slope)
        
        mean_slope = np.mean(slopes)
        signature_slopes.append(mean_slope)
        signature_names.append(f'Sig {sig_idx}')
    
    colors_slope = ['red' if slope > 0 else 'blue' for slope in signature_slopes]
    bars = ax3.bar(range(len(signature_slopes)), signature_slopes, 
                   color=colors_slope, alpha=0.7)
    ax3.set_xticks(range(len(signature_names)))
    ax3.set_xticklabels(signature_names, rotation=45)
    ax3.set_ylabel('Average Slope (Change per Month)')
    ax3.set_title('Signature Trends Before Treatment')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Variability analysis
    ax4 = axes[1, 1]
    
    within_patient_var = []
    between_patient_var = []
    
    for sig_idx in top_signatures:
        # Within-patient variability (temporal)
        within_var = np.mean([np.var(treated_patterns[p, sig_idx, :]) 
                             for p in range(treated_patterns.shape[0])])
        
        # Between-patient variability (at each time point)
        between_var = np.mean([np.var(treated_patterns[:, sig_idx, t]) 
                              for t in range(treated_patterns.shape[2])])
        
        within_patient_var.append(within_var)
        between_patient_var.append(between_var)
    
    x = np.arange(len(top_signatures))
    width = 0.35
    
    ax4.bar(x - width/2, within_patient_var, width, label='Within-Patient (Temporal)', alpha=0.7)
    ax4.bar(x + width/2, between_patient_var, width, label='Between-Patient', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Sig {s}' for s in top_signatures], rotation=45)
    ax4.set_ylabel('Variance')
    ax4.set_title('Signature Variability Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, top_signatures

def create_comprehensive_signature_report(learner):
    """
    Create a comprehensive report of signature analysis
    """
    print("=== COMPREHENSIVE SIGNATURE ANALYSIS ===\n")
    
    # 1. Get signature statistics with variability
    print("1. Calculating signature differences with effect sizes...")
    sig_stats = analyze_signature_differences_with_variability(learner)
    
    if sig_stats is not None:
        # Sort by effect size
        sig_stats_sorted = sig_stats.sort_values('cohens_d', key=abs, ascending=False)
        
        print(f"\nTop signatures by effect size (Cohen's d):")
        print("="*80)
        print(f"{'Sig':<4} {'Treated':<8} {'Untreated':<10} {'Raw Diff':<8} {'Cohens d':<8} {'P-value':<8} {'Significant'}")
        print("="*80)
        
        for _, row in sig_stats_sorted.head(10).iterrows():
            sig_marker = "***" if row['significant'] else "   "
            print(f"{row['signature']:<4} {row['treated_mean']:<8.4f} {row['untreated_mean']:<10.4f} "
                  f"{row['raw_difference']:<8.4f} {row['cohens_d']:<8.4f} {row['p_value']:<8.4f} {sig_marker}")
    
    # 2. Create evolution plots
    print(f"\n2. Creating signature evolution plots...")
    
    # Use top signatures by effect size for plotting
    if sig_stats is not None:
        top_sigs = sig_stats_sorted.head(8)['signature'].tolist()
    else:
        top_sigs = None
    
    fig, selected_sigs = plot_signature_evolution_over_time(learner, top_signatures=top_sigs)
    
    # 3. Summary insights
    print(f"\n3. Key Insights:")
    if sig_stats is not None:
        significant_sigs = sig_stats[sig_stats['significant']]
        print(f"   - {len(significant_sigs)} signatures show significant differences")
        
        if len(significant_sigs) > 0:
            top_sig = significant_sigs.loc[significant_sigs['cohens_d'].abs().idxmax()]
            print(f"   - Signature {top_sig['signature']} shows strongest effect (Cohen's d = {top_sig['cohens_d']:.3f})")
            
            increasing_sigs = significant_sigs[significant_sigs['raw_difference'] > 0]
            decreasing_sigs = significant_sigs[significant_sigs['raw_difference'] < 0]
            
            print(f"   - {len(increasing_sigs)} signatures higher in treated patients")
            print(f"   - {len(decreasing_sigs)} signatures lower in treated patients")
    
    return sig_stats, fig, selected_sigs

# Usage function
def run_advanced_signature_analysis(learner):
    """
    Run the complete advanced signature analysis
    """
    sig_stats, evolution_fig, top_signatures = create_comprehensive_signature_report(learner)
    
    return {
        'signature_statistics': sig_stats,
        'evolution_figure': evolution_fig,
        'top_signatures': top_signatures
    }

# Usage:
# advanced_results = run_advanced_signature_analysis(learner)
