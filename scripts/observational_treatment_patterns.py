import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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
                        
                else:
                    # Patient never treated (at least not in our data)
                    patterns['never_treated'].append(eid)
                    
            except Exception as e:
                continue
        
        print(f"Found {len(patterns['treated_patients'])} treated patients")
        print(f"Found {len(patterns['never_treated'])} never-treated patients")
        
        return patterns