import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Union, Tuple

class FairnessChecker:
    def __init__(self, sensitive_attributes: List[str]):
        """
        Initialize the FairnessChecker with sensitive attributes to check for discrimination.
        
        Args:
            sensitive_attributes: List of column names that are considered sensitive (e.g., ['race', 'gender', 'age'])
        """
        self.sensitive_attributes = sensitive_attributes
        
    def check_discrimination(self, 
                           data: pd.DataFrame,
                           target_column: str,
                           model_predictions: Union[np.ndarray, List] = None,
                           threshold: float = 0.1) -> Dict:
        """
        Check for discrimination in the dataset by analyzing disparate impact and demographic parity.
        
        Args:
            data: DataFrame containing the dataset
            target_column: Name of the target/outcome column
            model_predictions: Optional model predictions to check for discrimination in model outputs
            threshold: Threshold for determining significant discrimination (default: 0.1)
            
        Returns:
            Dictionary containing discrimination analysis results
        """
        results = {}
        
        for attribute in self.sensitive_attributes:
            if attribute not in data.columns:
                continue
                
            # Calculate base rates for each group in the sensitive attribute
            groups = data[attribute].unique()
            group_stats = {}
            
            for group in groups:
                group_mask = data[attribute] == group
                group_data = data[group_mask]
                
                # Calculate positive outcome rate
                if model_predictions is not None:
                    positive_rate = np.mean(model_predictions[group_mask])
                else:
                    positive_rate = group_data[target_column].mean()
                
                group_stats[group] = {
                    'size': len(group_data),
                    'positive_rate': positive_rate
                }
            
            # Calculate disparate impact
            max_rate = max(stats['positive_rate'] for stats in group_stats.values())
            min_rate = min(stats['positive_rate'] for stats in group_stats.values())
            
            if max_rate > 0:
                disparate_impact = min_rate / max_rate
            else:
                disparate_impact = 1.0
            
            # Determine if discrimination exists
            has_discrimination = disparate_impact < (1 - threshold)
            
            results[attribute] = {
                'has_discrimination': has_discrimination,
                'disparate_impact': disparate_impact,
                'group_stats': group_stats
            }
            
        return results
    
    def get_discriminatory_cases(self,
                               data: pd.DataFrame,
                               target_column: str,
                               model_predictions: Union[np.ndarray, List] = None) -> pd.DataFrame:
        """
        Return specific cases where discrimination might be present.
        
        Args:
            data: DataFrame containing the dataset
            target_column: Name of the target/outcome column
            model_predictions: Optional model predictions to check
            
        Returns:
            DataFrame containing potentially discriminatory cases
        """
        discriminatory_cases = pd.DataFrame()
        
        for attribute in self.sensitive_attributes:
            if attribute not in data.columns:
                continue
                
            groups = data[attribute].unique()
            
            # Calculate the overall positive rate
            if model_predictions is not None:
                overall_positive_rate = np.mean(model_predictions)
            else:
                overall_positive_rate = data[target_column].mean()
            
            for group in groups:
                group_mask = data[attribute] == group
                group_data = data[group_mask]
                
                if model_predictions is not None:
                    group_predictions = model_predictions[group_mask]
                    group_positive_rate = np.mean(group_predictions)
                else:
                    group_positive_rate = group_data[target_column].mean()
                
                # If this group has a significantly different outcome rate
                if abs(group_positive_rate - overall_positive_rate) > 0.1:
                    cases = group_data.copy()
                    cases['discrimination_type'] = f'Different outcome rate for {attribute}={group}'
                    cases['group_positive_rate'] = group_positive_rate
                    cases['overall_positive_rate'] = overall_positive_rate
                    discriminatory_cases = pd.concat([discriminatory_cases, cases])
        
        return discriminatory_cases

def check_dataset_fairness(data: pd.DataFrame,
                         sensitive_attributes: List[str],
                         target_column: str,
                         model_predictions: Union[np.ndarray, List] = None) -> Tuple[Dict, pd.DataFrame]:
    """
    Convenience function to check dataset fairness and get discriminatory cases.
    
    Args:
        data: DataFrame containing the dataset
        sensitive_attributes: List of column names that are considered sensitive
        target_column: Name of the target/outcome column
        model_predictions: Optional model predictions to check
        
    Returns:
        Tuple containing:
        - Dictionary with discrimination analysis results
        - DataFrame with potentially discriminatory cases
    """
    checker = FairnessChecker(sensitive_attributes)
    results = checker.check_discrimination(data, target_column, model_predictions)
    cases = checker.get_discriminatory_cases(data, target_column, model_predictions)
    return results, cases
