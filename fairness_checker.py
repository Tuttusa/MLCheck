from mlCheck import Assume, Assert, propCheck
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)  # 13 features in Adult dataset
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)   # Binary classification

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def check_discrimination(dataset_path, model_path=None, iteration_no=5):
    """
    Check for discrimination in the dataset using MLCheck framework.
    
    Args:
        dataset_path: Path to the dataset CSV file
        model_path: Optional path to a pre-trained model
        iteration_no: Number of times to run each test case
    """
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Initialize the model
    if model_path:
        model = torch.load(model_path)
    else:
        model = Net()
        
    # Define sensitive attributes to check
    sensitive_attrs = {
        'race': list(range(5)),    # 0-4 for different races
        'sex': [0, 1],            # 0-1 for gender
        'age': list(range(17, 91, 10))  # Age groups
    }
    
    results = {}
    
    for attr, values in sensitive_attrs.items():
        cex_counts = []
        
        for box in ['Decision tree', 'DNN']:
            for i in range(iteration_no):
                # Initialize property checker
                checker = propCheck(
                    no_of_params=1,
                    max_samples=1500,
                    model_type='Pytorch',
                    model=model,
                    mul_cex=True,
                    xml_file='dataInput.xml',
                    no_of_class=2,
                    white_box_model=box,
                    no_of_layers=2,
                    layer_size=32,
                    no_EPOCHS=1
                )
                
                # Set assumptions based on attribute
                if attr == 'race':
                    for val in values:
                        Assume(f'x[8] == {val}')  # 8 is race index
                        Assert('model.predict(x) == model.predict(x_prime) where x_prime is x with different race')
                
                elif attr == 'sex':
                    for val in values:
                        Assume(f'x[9] == {val}')  # 9 is sex index
                        Assert('model.predict(x) == model.predict(x_prime) where x_prime is x with different sex')
                
                elif attr == 'age':
                    for val in values:
                        Assume(f'x[0] >= {val} and x[0] < {val + 10}')  # 0 is age index
                        Assert('abs(model.predict(x) - model.predict(x_prime)) < 0.1 where x_prime is x with age +/- 5')
                
                # Check for discriminatory cases
                dfCexSet = pd.read_csv('CexSet.csv')
                cex_counts.append(dfCexSet.shape[0])
        
        results[attr] = {
            'total_cases': sum(cex_counts),
            'avg_cases_per_iteration': np.mean(cex_counts),
            'max_cases': max(cex_counts),
            'min_cases': min(cex_counts)
        }
    
    return results

def get_discriminatory_cases():
    """
    Return the discriminatory cases found in the last run.
    """
    try:
        return pd.read_csv('CexSet.csv')
    except:
        return pd.DataFrame()
