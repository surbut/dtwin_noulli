def plot_signature_deviations_over_time(thetas, processed_ids, matched_results, 
                                       reference_thetas_path="reference_thetas.csv",
                                       sig_indices=None, years_before_event=5, 
                                       save_plots=True):
    """
    Plot signature deviations from baseline over time for treated vs controls
    
    Parameters:
    - thetas: Signature loadings (N x K x T)
    - processed_ids: Array of all processed patient IDs
    - matched_results: Results from treatment analysis containing matched patients
    - reference_thetas_path: Path to reference_thetas.csv file
    - sig_indices: List of signature indices to plot (if None, use first 10)
    - years_before_event: How many years before event to plot
    - save_plots: Whether to save plots to files
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Load reference thetas
    try:
        reference_theta = pd.read_csv(reference_thetas_path, header=0).values
        print(f"Loaded reference_theta from {reference_thetas_path}: {reference_theta.shape}")
    except Exception as e:
        print(f"Error loading reference thetas: {e}")
        return None
    
    # Extract matched patient information
    treated_eids = matched_results['matched_patients']['treated_eids']
    control_eids = matched_results['matched_patients']['control_eids']
    treated_indices = matched_results['matched_patients']['treated_indices']
    control_indices = matched_results['matched_patients']['control_indices']
    
    # Default signature indices
    if sig_indices is None:
        sig_indices = list(range(min(10, thetas.shape[1])))
    
    # Get the actual number of time points from theta data
    n_time_points = thetas.shape[2]
    print(f"Theta shape: {thetas.shape}, using last {n_time_points} time points")
    
    # Calculate time points (years before event) - match the actual data length
    time_points = np.linspace(-years_before_event, 0, n_time_points)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Colors for groups
    colors = {'treated': '#FF6B6B', 'control': '#4ECDC4'}
    
    # 1. BEFORE MATCHING PLOT (top)
    print("=== BEFORE MATCHING SIGNATURE DEVIATIONS ===")
    
    # Get trajectories for all patients in each group (before matching)
    treated_trajs = []
    control_trajs = []
    
    # Get trajectories for treated patients (all, not just matched)
    for i, eid in enumerate(treated_eids):
        global_idx = treated_indices[i]
        if global_idx < thetas.shape[0]:
            # Get the full trajectory
            traj = thetas[global_idx, :, :]  # All signatures, all time points
            if traj.shape[1] == n_time_points:
                treated_trajs.append(traj)
    
    # Get trajectories for control patients (all, not just matched)
    for i, eid in enumerate(control_eids):
        global_idx = control_indices[i]
        if global_idx < thetas.shape[0]:
            # Get the full trajectory
            traj = thetas[global_idx, :, :]  # All signatures, all time points
            if traj.shape[1] == n_time_points:
                control_trajs.append(traj)
    
    # Calculate average deviations for each group
    if len(treated_trajs) > 0:
        treated_avg = np.mean(treated_trajs, axis=0)  # Average across patients
        # Make sure reference_theta has the right shape
        if reference_theta.shape[1] >= n_time_points:
            ref_slice = reference_theta[:, -n_time_points:]
        else:
            ref_slice = reference_theta
        treated_dev = treated_avg - ref_slice  # Deviation from reference
        print(f"Treated trajectories: {len(treated_trajs)} patients")
    
    if len(control_trajs) > 0:
        control_avg = np.mean(control_trajs, axis=0)  # Average across patients
        # Make sure reference_theta has the right shape
        if reference_theta.shape[1] >= n_time_points:
            ref_slice = reference_theta[:, -n_time_points:]
        else:
            ref_slice = reference_theta
        control_dev = control_avg - ref_slice  # Deviation from reference
        print(f"Control trajectories: {len(control_trajs)} patients")
    
    # Plot treated deviations (top subplot)
    ax = axes[0]
    if len(treated_trajs) > 0:
        # Calculate mean deviation for each signature
        mean_props = np.abs(treated_dev).mean(axis=1)
        
        # Get top signatures by absolute mean deviation
        top_sig_idx = np.argsort(mean_props)[-len(sig_indices):][::-1]
        
        # Create stacked area plot
        bottom_pos = np.zeros(n_time_points)
        bottom_neg = np.zeros(n_time_points)
        colors_sig = plt.cm.tab20(np.linspace(0, 1, len(sig_indices)))
        
        for i, sig in enumerate(top_sig_idx):
            values = treated_dev[sig]
            # Split positive and negative deviations
            pos_values = np.maximum(values, 0)
            neg_values = np.minimum(values, 0)
            
            ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                           label=f'Sig {sig} (Δθ={mean_props[sig]:.3f})',
                           color=colors_sig[i])
            ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                           color=colors_sig[i], alpha=0.5)
            
            bottom_pos += pos_values
            bottom_neg += neg_values
        
        # Center y-axis around 0 for deviations
        max_dev = max(abs(bottom_pos.max()), abs(bottom_neg.min()))
        ax.set_ylim(-max_dev, max_dev)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    ax.set_title(f'Before Matching: Treated Group Signature Deviations from Baseline\n({len(treated_trajs)} patients)')
    ax.set_xlabel('Years Before Event')
    ax.set_ylabel('Δ Proportion (θ)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 2. AFTER MATCHING PLOT (bottom)
    print("=== AFTER MATCHING SIGNATURE DEVIATIONS ===")
    
    # Get trajectories for MATCHED patients only
    treated_trajs_matched = []
    control_trajs_matched = []
    
    # Get trajectories for matched treated patients
    for i, eid in enumerate(treated_eids):
        global_idx = treated_indices[i]
        if global_idx < thetas.shape[0]:
            traj = thetas[global_idx, :, :]
            if traj.shape[1] == n_time_points:
                treated_trajs_matched.append(traj)
    
    # Get trajectories for matched control patients
    for i, eid in enumerate(control_eids):
        global_idx = control_indices[i]
        if global_idx < thetas.shape[0]:
            traj = thetas[global_idx, :, :]
            if traj.shape[1] == n_time_points:
                control_trajs_matched.append(traj)
    
    # Calculate average deviations for matched groups
    if len(treated_trajs_matched) > 0:
        treated_avg_matched = np.mean(treated_trajs_matched, axis=0)
        # Make sure reference_theta has the right shape
        if reference_theta.shape[1] >= n_time_points:
            ref_slice = reference_theta[:, -n_time_points:]
        else:
            ref_slice = reference_theta
        treated_dev_matched = treated_avg_matched - ref_slice
        print(f"Matched treated trajectories: {len(treated_trajs_matched)} patients")
    
    if len(control_trajs_matched) > 0:
        control_avg_matched = np.mean(control_trajs_matched, axis=0)
        # Make sure reference_theta has the right shape
        if reference_theta.shape[1] >= n_time_points:
            ref_slice = reference_theta[:, -n_time_points:]
        else:
            ref_slice = reference_theta
        control_dev_matched = control_avg_matched - ref_slice
        print(f"Matched control trajectories: {len(control_trajs_matched)} patients")
    
    # Plot both groups on same subplot for comparison
    ax = axes[1]
    
    # Plot treated deviations
    if len(treated_trajs_matched) > 0:
        mean_props_treated = np.abs(treated_dev_matched).mean(axis=1)
        top_sig_idx_treated = np.argsort(mean_props_treated)[-len(sig_indices):][::-1]
        
        bottom_pos = np.zeros(n_time_points)
        bottom_neg = np.zeros(n_time_points)
        
        for i, sig in enumerate(top_sig_idx_treated):
            values = treated_dev_matched[sig]
            pos_values = np.maximum(values, 0)
            neg_values = np.minimum(values, 0)
            
            ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                           label=f'Treated Sig {sig} (Δθ={mean_props_treated[sig]:.3f})',
                           color=colors['treated'], alpha=0.7)
            ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                           color=colors['treated'], alpha=0.5)
            
            bottom_pos += pos_values
            bottom_neg += neg_values
    
    # Plot control deviations
    if len(control_trajs_matched) > 0:
        mean_props_control = np.abs(control_dev_matched).mean(axis=1)
        top_sig_idx_control = np.argsort(mean_props_control)[-len(sig_indices):][::-1]
        
        bottom_pos = np.zeros(n_time_points)
        bottom_neg = np.zeros(n_time_points)
        
        for i, sig in enumerate(top_sig_idx_control):
            values = control_dev_matched[sig]
            pos_values = np.maximum(values, 0)
            neg_values = np.minimum(values, 0)
            
            ax.fill_between(time_points, bottom_pos, bottom_pos + pos_values,
                           label=f'Control Sig {sig} (Δθ={mean_props_control[sig]:.3f})',
                           color=colors['control'], alpha=0.7)
            ax.fill_between(time_points, bottom_neg, bottom_neg + neg_values,
                           color=colors['control'], alpha=0.5)
            
            bottom_pos += pos_values
            bottom_neg += neg_values
    
    ax.set_title(f'After Matching: Treated vs Control Signature Deviations from Baseline\n(Treated: {len(treated_trajs_matched)}, Control: {len(control_trajs_matched)} patients)')
    ax.set_xlabel('Years Before Event')
    ax.set_ylabel('Δ Proportion (θ)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Overall title
    fig.suptitle('Signature Deviations from Baseline Over Time: Before vs After Matching', 
                 fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('signature_deviations_over_time.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'signature_deviations_over_time.png'")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Before matching:")
    print(f"  Treated patients: {len(treated_trajs)}")
    print(f"  Control patients: {len(control_trajs)}")
    
    print(f"\nAfter matching:")
    print(f"  Treated patients: {len(treated_trajs_matched)}")
    print(f"  Control patients: {len(control_trajs_matched)}")
    
    return fig