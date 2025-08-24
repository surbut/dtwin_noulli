import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.ensemble import RandomForestRegressor
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except (ImportError, AttributeError, KeyError) as e:
    print(f"Warning: PyMC not available ({type(e).__name__}). Bayesian functions will be skipped.")
    pm = None
    az = None
    PYMC_AVAILABLE = False

class ObservationalTreatmentPatternLearner:
    """
    Learn treatment patterns from observational data - no ground truth needed!
    """
    
    def __init__(self, signature_loadings, processed_ids, statin_prescriptions, 
                 covariates, time_grid_start_age=30, gp_scripts=None):
        self.signatures = signature_loadings  # N x K x T
        self.processed_ids = processed_ids
        self.prescriptions = statin_prescriptions
        self.covariates = covariates
        self.time_start_age = time_grid_start_age
        self.gp_scripts = gp_scripts
        
        # Process prescription data to find treatment initiation times
        self.treatment_patterns = self._extract_treatment_patterns()
        
    def _extract_treatment_patterns(self):
        """
        Extract when patients actually started treatment in observational data
        """
        patterns = {
            'treated_patients': [],
            'treatment_times': [],
            'pre_treatment_signatures': [],
            'post_treatment_outcomes': [],
            'never_treated': []
        }
        
        # Process prescriptions to find first statin prescription
        self.prescriptions['issue_date'] = pd.to_datetime(
            self.prescriptions['issue_date'], format='%d/%m/%Y', errors='coerce'
        )
        
        first_statins = (self.prescriptions
                        .groupby('eid')['issue_date']
                        .min()
                        .reset_index())
        
        # Only consider patients who have prescription data (if gp_scripts provided)
        if self.gp_scripts is not None:
            patients_to_check = set(self.gp_scripts['eid'].unique()).intersection(set(self.processed_ids))
        else:
            patients_to_check = self.processed_ids
        
        for eid in patients_to_check:
            try:
                patient_idx = np.where(self.processed_ids == eid)[0][0]
                
                # Check if patient received statins
                if eid in first_statins['eid'].values:
                    # Patient was treated
                    first_date = first_statins[first_statins['eid'] == eid]['issue_date'].iloc[0]
                    
                    # Get patient birth year and enrollment info
                    patient_cov = self.covariates[self.covariates['eid'] == eid]
                    if len(patient_cov) == 0:
                        continue
                        
                    birth_year = patient_cov['birth_year'].iloc[0]
                    age_at_treatment = first_date.year - birth_year
                    
                    # Convert to time index (CORRECTED: age 30 = index 0, age 81 = index 51)
                    time_idx = int(age_at_treatment - self.time_start_age)
                    
                    if 12 <= time_idx < self.signatures.shape[2] - 12:
                        patterns['treated_patients'].append(eid)
                        patterns['treatment_times'].append(time_idx)
                        
                        # Extract pre-treatment signature pattern
                        pre_pattern = self.signatures[patient_idx, :, time_idx-12:time_idx]
                        patterns['pre_treatment_signatures'].append(pre_pattern)
                        
                        # Extract post-treatment for outcome analysis (if available)
                        if time_idx + 24 < self.signatures.shape[2]:
                            post_pattern = self.signatures[patient_idx, :, time_idx:time_idx+24]
                            patterns['post_treatment_outcomes'].append(post_pattern)
                        else:
                            patterns['post_treatment_outcomes'].append(None)
                
                else:
                    # Patient never treated (at least not in our data)
                    patterns['never_treated'].append(eid)
                    
            except Exception as e:
                continue
        
        # Convert to arrays
        if patterns['pre_treatment_signatures']:
            patterns['pre_treatment_signatures'] = np.array(patterns['pre_treatment_signatures'])
        
        print(f"Found {len(patterns['treated_patients'])} treated patients")
        print(f"Found {len(patterns['never_treated'])} never-treated patients")
        
        return patterns
    
    def discover_treatment_initiation_patterns(self, n_clusters=3):
        """
        Discover common patterns in pre-treatment signatures using clustering
        This learns what signatures look like before people actually get treated
        """
        if len(self.treatment_patterns['pre_treatment_signatures']) == 0:
            return None
        
        pre_treatment = self.treatment_patterns['pre_treatment_signatures']
        n_patients, n_sigs, window = pre_treatment.shape
        
        # Feature engineering: extract meaningful patterns from pre-treatment period
        features = []
        for p in range(n_patients):
            patient_features = []
            for s in range(n_sigs):
                trajectory = pre_treatment[p, s, :]
                
                # Trend features
                trend_slope = np.polyfit(range(len(trajectory)), trajectory, 1)[0]
                trend_acceleration = np.polyfit(range(len(trajectory)), trajectory, 2)[0]
                
                # Level features  
                recent_level = np.mean(trajectory[-3:])  # Last 3 months
                early_level = np.mean(trajectory[:3])    # First 3 months
                max_level = np.max(trajectory)
                
                # Variability features
                volatility = np.std(trajectory)
                
                # Change features
                total_change = trajectory[-1] - trajectory[0]
                max_change = np.max(np.abs(np.diff(trajectory)))
                
                patient_features.extend([
                    trend_slope, trend_acceleration, recent_level, early_level, 
                    max_level, volatility, total_change, max_change
                ])
            
            features.append(patient_features)
        
        features = np.array(features)
        
        # Cluster the pre-treatment patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Analyze each cluster
        cluster_patterns = {}
        for c in range(n_clusters):
            cluster_mask = clusters == c
            cluster_signatures = pre_treatment[cluster_mask]
            cluster_times = np.array(self.treatment_patterns['treatment_times'])[cluster_mask]
            
            cluster_patterns[c] = {
                'n_patients': np.sum(cluster_mask),
                'mean_treatment_age': np.mean(cluster_times) + self.time_start_age,  # Convert back to age
                'signature_pattern': np.mean(cluster_signatures, axis=0),
                'signature_std': np.std(cluster_signatures, axis=0),
                'patient_indices': np.where(cluster_mask)[0]
            }
        
        return {
            'cluster_model': kmeans,
            'clusters': clusters,
            'cluster_patterns': cluster_patterns,
            'features': features,
            'n_clusters': n_clusters
        }
    
    def learn_treatment_responsive_patterns(self):
        """
        Learn what signature patterns predict "good" treatment timing
        by comparing outcomes of patients treated at different signature states
        """
        if len(self.treatment_patterns['pre_treatment_signatures']) == 0:
            return None
        
        pre_treatment = self.treatment_patterns['pre_treatment_signatures']
        treatment_times = np.array(self.treatment_patterns['treatment_times'])
        
        # Strategy 1: Early vs Late Treatment Analysis
        # Compare patients treated at different ages
        ages_at_treatment = treatment_times + self.time_start_age  # Convert back to actual ages
        median_age = np.median(ages_at_treatment)
        
        early_treaters = ages_at_treatment < median_age
        late_treaters = ages_at_treatment >= median_age
        
        # Compare signature patterns between early and late treaters
        early_patterns = pre_treatment[early_treaters]
        late_patterns = pre_treatment[late_treaters]
        
        # Statistical comparison
        signature_differences = []
        for s in range(pre_treatment.shape[1]):
            # Compare final pre-treatment levels
            early_levels = early_patterns[:, s, -1]  # Last month before treatment
            late_levels = late_patterns[:, s, -1]
            
            stat, p_val = ttest_ind(early_levels, late_levels)
            signature_differences.append({
                'signature': s,
                'early_mean': np.mean(early_levels),
                'late_mean': np.mean(late_levels),
                'difference': np.mean(early_levels) - np.mean(late_levels),
                'p_value': p_val,
                'effect_size': stat
            })
        
        # Strategy 2: Signature Trajectory Analysis
        # Find signatures that show concerning trends before treatment
        concerning_patterns = {}
        for s in range(pre_treatment.shape[1]):
            sig_trajectories = pre_treatment[:, s, :]
            
            # Calculate trends for each patient
            trends = []
            accelerations = []
            for traj in sig_trajectories:
                trend = np.polyfit(range(len(traj)), traj, 1)[0]
                if len(traj) >= 3:
                    accel = np.polyfit(range(len(traj)), traj, 2)[0]
                else:
                    accel = 0
                trends.append(trend)
                accelerations.append(accel)
            
            concerning_patterns[s] = {
                'mean_trend': np.mean(trends),
                'mean_acceleration': np.mean(accelerations),
                'trend_std': np.std(trends),
                'concerning_trend_fraction': np.mean(np.array(trends) > 0),  # Upward trends
                'accelerating_fraction': np.mean(np.array(accelerations) > 0)
            }
        
        return {
            'early_vs_late': {
                'early_treaters': early_treaters,
                'late_treaters': late_treaters,
                'signature_differences': signature_differences,
                'median_treatment_age': median_age
            },
            'concerning_patterns': concerning_patterns,
            'treatment_readiness_signatures': self._identify_readiness_signatures(concerning_patterns)
        }
    
    def _identify_readiness_signatures(self, concerning_patterns):
        """
        Identify which signatures most strongly indicate treatment readiness
        """
        readiness_scores = {}
        for s, patterns in concerning_patterns.items():
            # Signatures that consistently trend upward before treatment are "readiness indicators"
            trend_score = patterns['concerning_trend_fraction']  # Fraction showing upward trend
            acceleration_score = patterns['accelerating_fraction']  # Fraction accelerating
            consistency_score = 1 - patterns['trend_std']  # Lower std = more consistent
            
            readiness_score = (trend_score + acceleration_score + max(0, consistency_score)) / 3
            readiness_scores[s] = readiness_score
        
        # Sort by readiness score
        sorted_readiness = sorted(readiness_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_readiness
    
    def compare_treated_vs_untreated_patterns(self, window=12):
        """
        Compare signature patterns between treated and never-treated patients
        This helps identify what makes someone "treatment-worthy"
        """
        treated_eids = set(self.treatment_patterns['treated_patients'])
        never_treated_eids = [eid for eid in self.treatment_patterns['never_treated']]
        
        # Sample signature patterns from never-treated patients at various ages
        never_treated_patterns = []
        sample_size = min(len(treated_eids), len(never_treated_eids), 1000)  # Limit sample size
        
        for eid in never_treated_eids[:sample_size]:
            try:
                patient_idx = np.where(self.processed_ids == eid)[0][0]
                
                # Sample a random time point (ages 40-70 -> time indices 10-40)
                valid_times = list(range(window, min(40, self.signatures.shape[2] - window)))
                if len(valid_times) == 0:
                    continue
                    
                sample_time = np.random.choice(valid_times)
                pattern = self.signatures[patient_idx, :, sample_time-window:sample_time]
                
                # Check pattern shape matches expected
                if pattern.shape == (self.signatures.shape[1], window):
                    never_treated_patterns.append(pattern)
                    
            except Exception as e:
                continue
        
        if len(never_treated_patterns) == 0:
            return None
        
        # Convert to array safely
        try:
            never_treated_patterns = np.array(never_treated_patterns)
            treated_patterns = self.treatment_patterns['pre_treatment_signatures']
            
            # Ensure we have valid data
            if len(never_treated_patterns) == 0 or len(treated_patterns) == 0:
                return None
                
        except ValueError as e:
            print(f"Error creating arrays: {e}")
            return None
        
        # Compare the two groups
        comparisons = {}
        n_signatures = min(treated_patterns.shape[1], never_treated_patterns.shape[1])
        
        for s in range(n_signatures):
            try:
                # Compare final levels
                treated_levels = treated_patterns[:, s, -1]
                untreated_levels = never_treated_patterns[:, s, -1]
                
                # Only proceed if we have valid data
                if len(treated_levels) > 0 and len(untreated_levels) > 0:
                    stat, p_val = mannwhitneyu(treated_levels, untreated_levels, 
                                             alternative='two-sided')
                    
                    comparisons[s] = {
                        'treated_mean': np.mean(treated_levels),
                        'untreated_mean': np.mean(untreated_levels),
                        'difference': np.mean(treated_levels) - np.mean(untreated_levels),
                        'p_value': p_val,
                        'effect_size': (np.mean(treated_levels) - np.mean(untreated_levels)) / 
                                      np.sqrt((np.var(treated_levels) + np.var(untreated_levels)) / 2)
                    }
            except Exception as e:
                continue
        
        return comparisons
    
    def build_treatment_readiness_predictor(self):
        """
        Build a model that predicts treatment readiness based on learned patterns
        """
        # Learn patterns first
        responsive_patterns = self.learn_treatment_responsive_patterns()
        treated_vs_untreated = self.compare_treated_vs_untreated_patterns()
        
        if responsive_patterns is None or treated_vs_untreated is None:
            return None
        
        # Create features based on learned patterns
        def extract_readiness_features(signature_pattern):
            """Extract features that indicate treatment readiness"""
            features = []
            
            for s in range(signature_pattern.shape[0]):
                trajectory = signature_pattern[s, :]
                
                # Trend (from responsive patterns analysis)
                trend = np.polyfit(range(len(trajectory)), trajectory, 1)[0]
                
                # Level compared to typical treated patients
                level = trajectory[-1]
                treated_typical = responsive_patterns['concerning_patterns'][s]['mean_trend']
                level_vs_typical = level - treated_typical
                
                # Acceleration
                if len(trajectory) >= 3:
                    accel = np.polyfit(range(len(trajectory)), trajectory, 2)[0]
                else:
                    accel = 0
                
                features.extend([trend, level, level_vs_typical, accel])
            
            return features
        
        # Prepare training data
        X_treated = []
        y_treated = []
        
        # Positive examples: actual pre-treatment patterns
        for pattern in self.treatment_patterns['pre_treatment_signatures']:
            features = extract_readiness_features(pattern)
            X_treated.append(features)
            y_treated.append(1)  # Treatment indicated
        
        # Negative examples: patterns from never-treated patients
        sample_size = min(len(X_treated), len(self.treatment_patterns['never_treated']), 500)
        
        for eid in self.treatment_patterns['never_treated'][:sample_size]:
            try:
                patient_idx = np.where(self.processed_ids == eid)[0][0]
                
                # Sample random time (ages 40-70 -> time indices 10-40)
                valid_times = list(range(12, min(40, self.signatures.shape[2] - 12)))
                if len(valid_times) == 0:
                    continue
                    
                sample_time = np.random.choice(valid_times)
                pattern = self.signatures[patient_idx, :, sample_time-12:sample_time]
                
                # Check pattern shape
                if pattern.shape == (self.signatures.shape[1], 12):
                    features = extract_readiness_features(pattern)
                    X_treated.append(features)
                    y_treated.append(0)  # Treatment not indicated
                    
            except Exception as e:
                continue
        
        X = np.array(X_treated)
        y = np.array(y_treated)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Cross-validation score
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        return {
            'model': model,
            'feature_extractor': extract_readiness_features,
            'cv_auc': np.mean(cv_scores),
            'responsive_patterns': responsive_patterns,
            'treated_vs_untreated': treated_vs_untreated
        }
    
    def plot_learned_patterns(self):
        """
        Visualize the patterns learned from observational data
        """
        # Discover patterns
        cluster_analysis = self.discover_treatment_initiation_patterns()
        responsive_patterns = self.learn_treatment_responsive_patterns()
        treated_vs_untreated = self.compare_treated_vs_untreated_patterns()
        
        if any(x is None for x in [cluster_analysis, responsive_patterns, treated_vs_untreated]):
            print("Insufficient data for pattern analysis")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Learned Treatment Patterns from Observational Data', fontsize=16)
        
        # 1. Pre-treatment signature clusters
        ax1 = axes[0, 0]
        n_clusters = cluster_analysis['n_clusters']
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        
        for c in range(n_clusters):
            pattern = cluster_analysis['cluster_patterns'][c]
            n_patients = pattern['n_patients']
            mean_age = pattern['mean_treatment_age']
            
            # Plot average signature for top signature
            top_sig = pattern['signature_pattern'][0, :]  # First signature
            ax1.plot(range(-12, 0), top_sig, color=colors[c], linewidth=2,
                    label=f'Cluster {c} (n={n_patients}, age={mean_age:.1f})')
        
        ax1.set_xlabel('Months Before Treatment')
        ax1.set_ylabel('Signature Loading')
        ax1.set_title('Pre-Treatment Signature Clusters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Treatment readiness signatures
        ax2 = axes[0, 1]
        readiness_sigs = responsive_patterns['treatment_readiness_signatures']
        top_readiness = readiness_sigs[:5]  # Top 5
        
        signatures = [s[0] for s in top_readiness]
        scores = [s[1] for s in top_readiness]
        
        bars = ax2.bar(range(len(signatures)), scores, alpha=0.7)
        ax2.set_xticks(range(len(signatures)))
        ax2.set_xticklabels([f'Sig {s}' for s in signatures])
        ax2.set_ylabel('Treatment Readiness Score')
        ax2.set_title('Most Predictive Signatures')
        ax2.grid(True, alpha=0.3)
        
        # 3. Treated vs Untreated comparison
        ax3 = axes[0, 2]
        signatures = list(treated_vs_untreated.keys())[:8]  # First 8 signatures
        differences = [treated_vs_untreated[s]['difference'] for s in signatures]
        p_values = [treated_vs_untreated[s]['p_value'] for s in signatures]
        
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        bars = ax3.bar(range(len(signatures)), differences, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(signatures)))
        ax3.set_xticklabels([f'Sig {s}' for s in signatures])
        ax3.set_ylabel('Difference (Treated - Untreated)')
        ax3.set_title('Signature Differences: Treated vs Untreated')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # 4. Treatment timing distribution
        ax4 = axes[1, 0]
        treatment_ages = np.array(self.treatment_patterns['treatment_times']) + self.time_start_age
        ax4.hist(treatment_ages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Age at Treatment Initiation')
        ax4.set_ylabel('Number of Patients')
        ax4.set_title('Observed Treatment Initiation Ages')
        ax4.grid(True, alpha=0.3)
        
        # 5. Signature trends before treatment
        ax5 = axes[1, 1]
        concerning = responsive_patterns['concerning_patterns']
        signatures = list(concerning.keys())[:8]
        trend_fractions = [concerning[s]['concerning_trend_fraction'] for s in signatures]
        
        ax5.bar(range(len(signatures)), trend_fractions, alpha=0.7, color='orange')
        ax5.set_xticks(range(len(signatures)))
        ax5.set_xticklabels([f'Sig {s}' for s in signatures])
        ax5.set_ylabel('Fraction with Upward Trend')
        ax5.set_title('Concerning Trends Before Treatment')
        ax5.grid(True, alpha=0.3)
        
        # 6. Early vs Late treatment comparison
        ax6 = axes[1, 2]
        early_late = responsive_patterns['early_vs_late']['signature_differences']
        sig_diffs = early_late[:6]  # First 6 signatures
        
        differences = [d['difference'] for d in sig_diffs]
        p_values = [d['p_value'] for d in sig_diffs]
        signatures = [d['signature'] for d in sig_diffs]
        
        colors = ['green' if p < 0.05 else 'gray' for p in p_values]
        ax6.bar(range(len(signatures)), differences, color=colors, alpha=0.7)
        ax6.set_xticks(range(len(signatures)))
        ax6.set_xticklabels([f'Sig {s}' for s in signatures])
        ax6.set_ylabel('Difference (Early - Late Treaters)')
        ax6.set_title('Early vs Late Treatment Signatures')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Usage function
def learn_from_observational_data(signature_loadings, processed_ids, statin_prescriptions, 
                                covariates):
    """
    Complete workflow for learning treatment patterns from observational data
    """
    learner = ObservationalTreatmentPatternLearner(
        signature_loadings, processed_ids, statin_prescriptions, covariates
    )
    
    print("=== Learning Treatment Patterns from Observational Data ===\n")
    
    # Learn patterns
    print("1. Discovering pre-treatment signature clusters...")
    clusters = learner.discover_treatment_initiation_patterns()
    
    print("2. Learning treatment-responsive patterns...")
    responsive = learner.learn_treatment_responsive_patterns()
    
    print("3. Comparing treated vs never-treated patterns...")
    comparison = learner.compare_treated_vs_untreated_patterns()
    
    print("4. Building treatment readiness predictor...")
    predictor = learner.build_treatment_readiness_predictor()
    
    if predictor:
        print(f"   Treatment readiness model AUC: {predictor['cv_auc']:.3f}")
    
    # Visualize learned patterns
    print("5. Visualizing learned patterns...")
    fig = learner.plot_learned_patterns()
    
    return learner, {
        'clusters': clusters,
        'responsive_patterns': responsive,
        'treated_vs_untreated': comparison,
        'predictor': predictor
    }


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
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((treated_std**2 + untreated_std**2) / 2)
        cohens_d = (treated_mean - untreated_mean) / pooled_std if pooled_std > 0 else 0
        
        # Statistical test
        t_stat, p_val = ttest_ind(treated_levels, untreated_levels)
        
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

def bayesian_propensity_response_model(signature_trajectories, treatment_status, outcomes):
    """
    Two-stage model: signatures → propensity → response
    """
    if not PYMC_AVAILABLE:
        print("PyMC not available. Skipping Bayesian analysis.")
        return None, None
        
    n_patients, n_signatures = signature_trajectories.shape
    
    with pm.Model() as model:
        # Stage 1: Propensity model (who gets treated)
        propensity_signature_effects = pm.Normal('propensity_signature_effects', 
                                               mu=0, sigma=1, shape=n_signatures)
        propensity_baseline = pm.Normal('propensity_baseline', mu=0, sigma=1)
        
        # Propensity score
        propensity_logit = propensity_baseline + pm.math.dot(signature_trajectories, 
                                                            propensity_signature_effects)
        propensity = pm.math.sigmoid(propensity_logit)
        
        # Treatment assignment
        treatment = pm.Bernoulli('treatment', p=propensity, observed=treatment_status)
        
        # Stage 2: Response model (how you respond if treated)
        response_signature_effects = pm.Normal('response_signature_effects', 
                                             mu=0, sigma=1, shape=n_signatures)
        response_baseline = pm.Normal('response_baseline', mu=0, sigma=1)
        treatment_effect = pm.Normal('treatment_effect', mu=0, sigma=1)
        
        # Response (only for treated patients)
        response_logit = response_baseline + pm.math.dot(signature_trajectories, 
                                                        response_signature_effects) + treatment_status * treatment_effect
        response_prob = pm.math.sigmoid(response_logit)
        
        # Outcome
        outcome = pm.Bernoulli('outcome', p=response_prob, observed=outcomes)
        
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    
    return trace, model
