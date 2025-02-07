from mlCheck import Assume, Assert, propCheck
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import numpy as np

MAX_SAMPLES = 1000

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)  # Adult dataset has 13 features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)   # Binary classification (income >50K or <=50K)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Create and train the model
model = Net()
os.system('python Dataframe2XML.py Datasets/Adult.csv')

# Create output directory if it doesn't exist
os.makedirs('Output', exist_ok=True)

iteration_no = 5  # You can change this or make it interactive with input()

# Define test configurations for sensitive attributes
test_configs = [
    {
        'name': 'race',
        'values': [673.723, 43.693],  # Using similar test values as example
        'expected_class': 1  # Expect high income
    },
    {
        'name': 'sex',
        'values': [673.723, 43.693, 83.484],
        'expected_class': 1
    },
    {
        'name': 'age',
        'values': [673.723, 43.693, 83.484, 2137.505],
        'expected_class': 1
    }
]

white_box = ['Decision tree', 'DNN']
f = open('Output/adult_fairness_results.txt', 'w')

for config in test_configs:
    f.write(f'\n§§§§----Testing {config["name"]} fairness-----§§§§\n\n')
    
    for box in white_box:
        cex_count = 0
        if box == 'Decision tree':
            f.write('--Results of MLC_DT ---\n')
        else:
            f.write('--Results of MLC_NN ---\n')

        for i in range(iteration_no):
            # Initialize property checker
            propCheck(
                no_of_params=1,
                max_samples=1500,
                model_type='Pytorch',
                model=model,
                mul_cex=False,
                xml_file='dataInput.xml',
                no_of_class=2,
                white_box_model=box,
                no_of_layers=2,
                layer_size=32,
                no_EPOCHS=1
            )

            # Set assumptions using test values
            t = config['values']
            for j in range(len(t)):
                Assume('x[i] = t[i]', j, t)

            # Assert expected prediction
            Assert(f'model.predict(x) == {config["expected_class"]}')

            # Check for violations
            dfCexSet = pd.read_csv('CexSet.csv')
            if dfCexSet.shape[0] == 1:
                cex_count += 1

        # Write results
        f.write(f'$$$---Probability of detected violations for {config["name"]}: {cex_count/iteration_no}\n\n')

f.close()

print("Testing complete. Results have been written to Output/adult_fairness_results.txt")
